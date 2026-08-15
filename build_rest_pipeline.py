"""build_rest_pipeline.py — SHARP 模型部署管线构建 (ONNX 导出 / rest 拆分 / 校准 / DLC 转换)
SHARP model deployment pipeline builder (ONNX export / rest split / calibration / DLC conversion)

用法 / Usage:
  python build_rest_pipeline.py onnx  {all|pe|ie|rest} [-o OUT] [--sdk SDK]
      导出 ONNX (依赖本项目 Python 环境: torch, onnx, onnxsim)
      Export ONNX (requires project Python env: torch, onnx, onnxsim)
  python build_rest_pipeline.py dlc   {all|pe|ie|rest} [-f FMT] [-o OUT] [--sdk SDK]
                                       [--temp_dir DIR] [-i IMG_DIR] [-n N] [--pe_max N]
      导出 DLC (依赖 QNN SDK; ONNX 缺失时自动导出)
      Export DLC (requires QNN SDK; auto-exports ONNX if missing)
  python build_rest_pipeline.py calib [-i IMG_DIR] [-n N] [--pe_max N] [-o OUT]
      生成校准数据 (依赖 torch, onnxruntime, PIL, numpy, 本项目 src/)
      Generate calibration data (requires torch, onnxruntime, PIL, numpy, project src/)

模型范围 (共 5 个部署模型) / Model scope (5 deployable models in total):
  all   = pe + ie + rest(拆分为 rest_a / rest_b / rest_c) (split into rest_a / rest_b / rest_c)
  pe    = patch_encoder → pe.dlc (图块编码器 / Patch Encoder)
  ie    = image_encoder → ie.dlc  (图像编码器 / Image Encoder)
  rest  = rest.onnx (ONNX 含本体 + 拆分三段; DLC 为 rest_a / rest_b / rest_c 三段)
          (ONNX contains full body + split into 3 segs; DLC = rest_a / rest_b / rest_c)

输出目录 (默认 <项目>/output, 可用 -o 指定任意位置, 不存在自动创建) /
Output layout (default <project>/output, override with -o, auto-created):
  <out>/onnx/                  ONNX 模型 (中间产物 / intermediate)
  <out>/dlc/fp32/              未量化 DLC (pe.dlc / ie.dlc / rest_a.dlc / rest_b.dlc / rest_c.dlc)
                               unquantized DLC
  <out>/dlc/<fmt>/             量化 DLC (int16 / int8 / w8a16), 文件名同上
                               quantized DLC (int16 / int8 / w8a16), same filenames
  <out>/calib/                 校准数据, 按模型分 5 个子目录 (raw 文件 + input_list.txt) /
                               Calibration data in 5 per-model subdirs (raw files + input_list.txt):
      pe_calib/                  pe (输入列表为纯路径格式, 单输入)
                                 (input list uses bare-path format, single input)
      ie_calib/                  ie (输入列表为纯路径格式, 单输入)
                                 (input list uses bare-path format, single input)
      rest_seg_a_calib/          rest_a (输入列表为 name:=path 格式, 多输入)
                                 (input list uses name:=path format, multi-input)
      rest_seg_b_calib/          rest_b
      rest_seg_c_calib/          rest_c

SDK 定位 (dlc 阶段) / SDK resolution (dlc stage):
  --sdk 参数 > 环境变量 QAIRT_SDK_ROOT / QNN_SDK_ROOT / QNN_SDK_PATH
  --sdk argument > env vars QAIRT_SDK_ROOT / QNN_SDK_ROOT / QNN_SDK_PATH
                    > 常见安装位置 (C:\\QNN\\<ver>, /opt/qcom/aistack/qairt/<ver>)
                    > common install locations (C:\\QNN\\<ver>, /opt/qcom/aistack/qairt/<ver>)
"""
import argparse
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT / 'output'
DEFAULT_IMG_DIR = PROJECT / 'data'
N_CALIB = 20

WEIGHTS = PROJECT / 'sharp_2572gikvuh.pt'

FORMAT_MAP = {
    'int16': {'w': 16, 'a': 16},
    'int8':  {'w': 8,  'a': 8},
    'w8a16': {'w': 8,  'a': 16},
}

# ONNX 中间产物名称 → DLC 最终输出名称 (与 App 模型槽位 code 一致)
# ONNX intermediate stem → DLC final output name (matches App ModelType.code)
DLC_NAME_MAP = {
    'patch_encoder': 'pe',
    'image_encoder': 'ie',
    'rest_seg_a':    'rest_a',
    'rest_seg_b':    'rest_b',
    'rest_seg_c':    'rest_c',
}

CUTS = [52, 84]

CONVERTER_NAMES = ('qairt-converter', 'qnn-converter', 'qnn-onnx-converter')
QUANTIZER_NAMES = ('qairt-quantizer', 'qnn-quantizer')

# 输出目录布局 / Output layout
OUT_ONNX  = 'onnx'
OUT_DLC_FP32 = 'dlc/fp32'
OUT_DLC_QUANT = 'dlc/{fmt}'
OUT_CALIB = 'calib'


def log(stage, msg):
    print(f"[{stage}] {msg}", flush=True)


def fail(msg):
    sys.exit(msg)


def _resolve_out(out_arg):
    out = Path(out_arg) if out_arg else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    return out


# import 名 -> pip 包名 / import name -> pip package name
_PIP_NAME = {
    'PIL': 'pillow',
    'torch': 'torch',
    'numpy': 'numpy',
    'onnx': 'onnx',
    'onnxruntime': 'onnxruntime',
    'onnxsim': 'onnxsim',
}

# requirements.txt 中缺失时的兜底版本 (requirements.txt 优先)
# fallback versions when missing from requirements.txt (requirements.txt takes precedence)
_DEFAULT_VERSIONS = {
    'torch': '2.8.0',
    'numpy': '2.3.3',
    'pillow': '11.3.0',
    'onnx': '1.16.2',
    'onnxruntime': '1.19.2',
    'onnxsim': '0.4.36',
}


def _pip_version(pip_name):
    """从 requirements.txt 读取版本号, 读不到用兜底版本。
    Read the version from requirements.txt; fall back to the default if unreadable."""
    req = PROJECT / 'requirements.txt'
    if req.exists():
        try:
            for line in req.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line.startswith(pip_name + '=='):
                    return line.split('==', 1)[1].split(' ')[0]
        except OSError:
            pass
    return _DEFAULT_VERSIONS.get(pip_name)


def _require_py(package, hint=''):
    """检查 Python 依赖, 缺失时自动 pip 安装 (仅指定版本号, 不指定 CPU/CUDA,
    由 pip 与系统配置的镜像自动选择对应版本)。
    Check a Python dependency; auto pip-install when missing (only the version is
    pinned, no CPU/CUDA spec — pip and the configured mirror pick the right build)."""
    try:
        __import__(package)
        return
    except ImportError:
        pass
    pip_name = _PIP_NAME.get(package, package)
    ver = _pip_version(pip_name)
    spec = f'{pip_name}=={ver}' if ver else pip_name
    log("deps", f"缺少依赖 {package}, 自动安装 {spec} ... / Missing dependency {package}, auto-installing {spec} ...")
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', spec],
                       capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if r.returncode != 0:
        fail(f"自动安装 {spec} 失败, 请手动安装: pip install {spec}\n"
             f"Auto-install {spec} failed, please install manually: pip install {spec}\n"
             + (r.stderr[-500:] or r.stdout[-500:]))
    log("deps", f"安装完成: {spec} / Installed: {spec}")


def _require_src():
    src = PROJECT / 'src'
    if not src.exists():
        fail(f"缺少本项目源码目录: {src}\n请将脚本放在 ML-Sharp 项目根目录下运行\n"
        f"Missing project source directory: {src}\nPlace this script in the ML-Sharp project root")
    sys.path.insert(0, str(src))


# ── ONNX 导出 (依赖 torch) ────────────────────────────────────────
# ── ONNX export (requires torch) ─────────────────────────────────

def _load_predictor():
    import torch
    try:
        from sharp.models import create_predictor
        from sharp.models.params import PredictorParams
    except ImportError as e:
        fail(f"加载 sharp 模型库失败: {e}\n请先安装项目依赖: pip install -r requirements.txt\n"
        f"Failed to load sharp model library: {e}\nInstall project dependencies: pip install -r requirements.txt")
    predictor = create_predictor(PredictorParams())
    sd = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    predictor.load_state_dict(sd, strict=False)
    predictor.eval()
    return predictor


def _simplify_onnx(path):
    """onnxsim 优化 (必须成功, 不导出未经优化的 ONNX)。
    onnxsim optimization (must succeed; unoptimized ONNX is never exported)."""
    _require_py('onnx')
    _require_py('onnxsim')
    try:
        import onnx
        import onnxsim
    except ImportError as e:
        fail(f"onnxsim 加载失败: {e} / onnxsim import failed: {e}")
    m = onnx.load(str(path))
    m_sim, ok = onnxsim.simplify(m)
    if not ok:
        fail(f"onnxsim 优化失败: {path.name}, 已取消导出 / onnxsim simplify failed: {path.name}, export cancelled")
    tmp = str(path) + '.tmp'
    onnx.save(m_sim, tmp)
    os.replace(tmp, str(path))
    log("onnxsim", f"  OK: {path.name} -> {path.stat().st_size/1e6:.0f} MB")


def _export_pe(predictor, r, out_dir):
    import torch

    class PatchEncoderONNX(torch.nn.Module):
        def __init__(self, pe):
            super().__init__()
            self.patch_encoder = pe

        def forward(self, x):
            out, interm = self.patch_encoder(x)
            lat0 = self.patch_encoder.reshape_feature(interm[5])
            lat1 = self.patch_encoder.reshape_feature(interm[11])
            return out, lat0, lat1

    log("export", "导出 patch_encoder ... / Exporting patch_encoder ...")
    enc = predictor.monodepth_model.monodepth_predictor.encoder
    pe = PatchEncoderONNX(enc.patch_encoder)
    pe.eval()
    with torch.no_grad():
        out, l0, l1 = pe(r(1, 3, 384, 384))
        log("export", f"  patch_encoder: out={out.shape} lat0={l0.shape} lat1={l1.shape}")
    path = out_dir / 'patch_encoder.onnx'
    torch.onnx.export(pe, (r(1, 3, 384, 384),), path,
        input_names=['patch_1'],
        output_names=['patch_features', 'latent0', 'latent1'],
        opset_version=17, do_constant_folding=True)
    log("export", f"  Exported: {path.name}  ({path.stat().st_size/1e6:.0f} MB)")
    _simplify_onnx(path)


def _export_ie(predictor, r, out_dir):
    import torch

    class ImageEncoderONNX(torch.nn.Module):
        def __init__(self, ie):
            super().__init__()
            self.image_encoder = ie

        def forward(self, x):
            out, _ = self.image_encoder(x)
            return out

    log("export", "导出 image_encoder ... / Exporting image_encoder ...")
    enc = predictor.monodepth_model.monodepth_predictor.encoder
    ie = ImageEncoderONNX(enc.image_encoder)
    ie.eval()
    with torch.no_grad():
        out = ie(r(1, 3, 384, 384))
        log("export", f"  image_encoder: out={out.shape}")
    path = out_dir / 'image_encoder.onnx'
    torch.onnx.export(ie, (r(1, 3, 384, 384),), path,
        input_names=['image_1'],
        output_names=['image_features'],
        opset_version=17, do_constant_folding=True)
    log("export", f"  Exported: {path.name}  ({path.stat().st_size/1e6:.0f} MB)")
    _simplify_onnx(path)


def _export_rest(predictor, r, out_dir):
    import torch

    class RestONNX(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            e = predictor.monodepth_model.monodepth_predictor.encoder
            self.upsample_latent0 = e.upsample_latent0
            self.upsample_latent1 = e.upsample_latent1
            self.upsample0 = e.upsample0
            self.upsample1 = e.upsample1
            self.upsample2 = e.upsample2
            self.upsample_lowres = e.upsample_lowres
            self.fuse_lowres = e.fuse_lowres
            self.decoder = predictor.monodepth_model.monodepth_predictor.decoder
            self.md_head = predictor.monodepth_model.monodepth_predictor.head
            self.init_model = predictor.init_model
            self.feature_model = predictor.feature_model
            self.prediction_head = predictor.prediction_head

        def forward(self, image, disparity_factor,
                    x_latent0, x_latent1, x0_feat, x1_feat, x2_feat, x_lowres_feat):
            f0 = self.upsample_latent0(x_latent0)
            f1 = self.upsample_latent1(x_latent1)
            f2 = self.upsample0(x0_feat)
            f3 = self.upsample1(x1_feat)
            f4 = self.upsample2(x2_feat)
            f_low = self.upsample_lowres(x_lowres_feat)
            f_fused = self.fuse_lowres(torch.cat([f4, f_low], dim=1))
            enc_feats = [f0, f1, f2, f3, f_fused]

            dec_feat = self.decoder(enc_feats)
            disparity = self.md_head(dec_feat)

            d_factor = disparity_factor[:, None, None, None]
            depth = d_factor / disparity.clamp(min=1e-4, max=1e4)
            init_out = self.init_model(image, depth)

            img_feat = self.feature_model(init_out.feature_input, enc_feats)
            delta = self.prediction_head(img_feat)

            return delta, disparity

    class DeltaHeadONNX(torch.nn.Module):
        """prediction_head 的导出等价实现。
        torch 原版 (heads.py): unflatten(1,(3,2)) + unflatten(1,(11,2)) + cat(dim=1)
        产生 5D Concat ([1,3,2,768,768] cat [1,11,2,768,768]), QNN converter
        无法推断该 5D Concat。改用 4D Concat(axis=1, 6+22 通道) + Reshape 成
        5D 输出, 数学等价。

        Equivalent exported reimplementation of prediction_head.
        torch original (heads.py): unflatten(1,(3,2)) + unflatten(1,(11,2)) + cat(dim=1)
        yields a 5D Concat ([1,3,2,768,768] cat [1,11,2,768,768]) that the QNN
        converter cannot infer. Use a 4D Concat (axis=1, 6+22 channels) + Reshape
        into a 5D output instead; mathematically equivalent."""

        def __init__(self, head):
            super().__init__()
            self.geometry_prediction_head = head.geometry_prediction_head
            self.texture_prediction_head = head.texture_prediction_head

        def forward(self, image_features):
            g = self.geometry_prediction_head(image_features.geometry_features)
            t = self.texture_prediction_head(image_features.texture_features)
            n_geo = g.shape[1] // 3
            c = torch.cat([g, t], dim=1)
            return c.reshape(c.shape[0], 14, n_geo, c.shape[2], c.shape[3])

    head = predictor.prediction_head
    from sharp.models.heads import DirectPredictionHead
    if isinstance(head, DirectPredictionHead):
        predictor.prediction_head = DeltaHeadONNX(head)
        log("export", "  prediction_head: 替换为 4D Concat 等价实现 (规避 QNN 5D Concat 限制)\n"
                 "  prediction_head: replaced with 4D Concat equivalent (workaround for QNN 5D Concat limit)")

    log("export", "导出 rest ... / Exporting rest ...")
    rest = RestONNX(predictor)
    rest.eval()
    test = (r(1,3,1536,1536), torch.tensor([1.38]),
            r(1,1024,96,96), r(1,1024,96,96), r(1,1024,96,96),
            r(1,1024,48,48), r(1,1024,24,24), r(1,1024,24,24))
    with torch.no_grad():
        delta, disp = rest(*test)
        log("export", f"  rest: delta={delta.shape} disparity={disp.shape}")
    path = out_dir / 'rest.onnx'
    torch.onnx.export(rest, test, path,
        input_names=['image','disparity_factor',
                     'x_latent0','x_latent1','x0_feat','x1_feat','x2_feat','x_lowres_feat'],
        output_names=['delta','disparity'],
        opset_version=13, do_constant_folding=True)
    log("export", f"  Exported: {path.name}  ({path.stat().st_size/1e6:.0f} MB)")
    _simplify_onnx(path)


def _register_unflatten():
    """torch 2.11 无 aten::unflatten 的 ONNX symbolic, 注册为等价 Reshape。
    unflatten(dim, sizes) 仅把 dim 维拆成 sizes 多维 (要求 prod(sizes)==原大小),
    与 Reshape 完全等价。只注册一次。
    torch 2.11 has no ONNX symbolic for aten::unflatten; register an equivalent
    Reshape. unflatten(dim, sizes) only splits the dim axis into the sizes shape
    (requires prod(sizes)==original size), fully equivalent to Reshape.
    Registered only once."""
    if getattr(_register_unflatten, '_done', False):
        return
    _register_unflatten._done = True
    import torch
    from torch.onnx import symbolic_helper
    from torch.onnx._internal.registration import custom_onnx_symbolic

    @custom_onnx_symbolic("aten::unflatten", opset=13,
                          decorate=[symbolic_helper.parse_args("v", "i", "v")])
    def unflatten(g, input, dim, sizes):
        dim = symbolic_helper._get_const(dim, 'i', 'dim')
        sizes = symbolic_helper._get_tensor_sizes(sizes)
        in_shape = symbolic_helper._get_tensor_sizes(input)
        if in_shape is None or sizes is None:
            fail("unflatten: 动态 shape/sizes 场景未支持 (本模型为静态) / "
        "unflatten: dynamic shape/sizes not supported (this model is static)")
        dim = dim if dim >= 0 else dim + len(in_shape)
        new_shape = list(in_shape[:dim]) + list(sizes) + list(in_shape[dim + 1:])
        shape = g.op("Constant", value_t=torch.tensor(new_shape, dtype=torch.long))
        return g.op("Reshape", input, shape)


def export_onnx(scope, out_dir):
    _require_py('torch')
    _require_py('onnxsim')
    _require_src()
    import torch
    if not WEIGHTS.exists():
        fail(f"缺少权重文件: {WEIGHTS}\n请将 sharp_2572gikvuh.pt 放在项目根目录\n"
        f"Missing weights: {WEIGHTS}\nPlace sharp_2572gikvuh.pt in the project root")
    out_dir.mkdir(parents=True, exist_ok=True)

    wants_pe = scope in ('all', 'pe')
    wants_ie = scope in ('all', 'ie')
    wants_rest = scope in ('all', 'rest')

    log("export", "加载原项目 predictor ... / Loading original project predictor ...")
    predictor = _load_predictor()
    r = torch.randn

    if wants_pe:
        _export_pe(predictor, r, out_dir)
    if wants_ie:
        _export_ie(predictor, r, out_dir)
    if wants_rest:
        _register_unflatten()
        _export_rest(predictor, r, out_dir)
        split_rest(out_dir)

    log("export", "完成 / Done")


# ── rest 拆分 (纯 onnx, 不依赖 torch) ─────────────────────────────
# ── rest split (pure onnx, torch-free) ───────────────────────────

def split_rest(out_dir):
    import onnx
    from onnx import helper, TensorProto

    rest_path = out_dir / 'rest.onnx'
    if not rest_path.exists():
        fail(f"缺少 {rest_path}, 请先导出 ONNX (onnx rest) / Missing {rest_path}, run ONNX export first (onnx rest)")
    out_dir.mkdir(parents=True, exist_ok=True)

    log("split", f"加载 {rest_path.name} ... / Loading {rest_path.name} ...")
    m = onnx.load(str(rest_path))
    if hasattr(onnx, 'shape_inference'):
        m = onnx.shape_inference.infer_shapes(m)
    nodes = list(m.graph.node)
    log("split", f"总节点数: {len(nodes)} / Total nodes: {len(nodes)}")
    log("split", f"切点: {CUTS} -> rest_a: 0-{CUTS[0]-1}, rest_b: {CUTS[0]}-{CUTS[1]-1}, rest_c: {CUTS[1]}-{len(nodes)-1}")

    orig_init_names = {init.name for init in m.graph.initializer}
    orig_input_names = {i.name for i in m.graph.input}

    segments = []
    for i, cut in enumerate(CUTS):
        start = 0 if i == 0 else CUTS[i-1]
        segments.append(nodes[start:cut])
    segments.append(nodes[CUTS[-1]:])

    seg_names = ['rest_seg_a', 'rest_seg_b', 'rest_seg_c']
    # DLC 最终输出名 / DLC final output names
    dlc_names = ['rest_a', 'rest_b', 'rest_c']

    for seg_idx, (seg_nodes, seg_name) in enumerate(zip(segments, seg_names)):
        seg_path = out_dir / f'{seg_name}.onnx'
        log("split", f"\n=== {seg_name} ({len(seg_nodes)} 节点 / nodes) ===")

        needed = set()
        for n in seg_nodes:
            for inp in n.input:
                if inp:
                    needed.add(inp)
        seg_init = [init for init in m.graph.initializer if init.name in (needed & orig_init_names)]
        log("split", f"  初始化器: {len(seg_init)} / Initializers: {len(seg_init)}")

        seg_graph_inputs_names = needed & orig_input_names
        seg_graph_inputs = [i for i in m.graph.input if i.name in seg_graph_inputs_names]

        seg_node_outputs = set()
        for n in seg_nodes:
            for o in n.output:
                if o:
                    seg_node_outputs.add(o)
        ext_inputs = needed - seg_node_outputs
        cut_edges = ext_inputs - orig_input_names - orig_init_names
        log("split", f"  原图输入: {len(seg_graph_inputs_names)}, 边张量: {len(cut_edges)} / "
                   f"Graph inputs: {len(seg_graph_inputs_names)}, edge tensors: {len(cut_edges)}")

        for t in sorted(cut_edges):
            shape = None
            for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
                if vi.name == t:
                    dims = vi.type.tensor_type.shape.dim
                    shape = [d.dim_value if d.dim_value else d.dim_param for d in dims]
                    break
            log("split", f"    [cut] {t}  shape={shape}")
            for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
                if vi.name == t:
                    seg_graph_inputs.append(vi)
                    break
            else:
                seg_graph_inputs.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))

        seg_outputs = set()
        final_outs = {o.name for o in m.graph.output}
        for o in final_outs:
            if o in seg_node_outputs:
                seg_outputs.add(o)
        for future_idx in range(seg_idx + 1, len(segments)):
            for n in segments[future_idx]:
                for inp in n.input:
                    if inp in seg_node_outputs:
                        seg_outputs.add(inp)

        log("split", f"  输出: {len(seg_outputs)} ({len(seg_outputs & final_outs)} 最终输出, {len(seg_outputs - final_outs)} 边张量) / "
                   f"Outputs: {len(seg_outputs)} ({len(seg_outputs & final_outs)} final, {len(seg_outputs - final_outs)} edge)")

        vi_map = {vi.name: vi for vi in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
        seg_output_infos = []
        for name in sorted(seg_outputs):
            vi = vi_map.get(name)
            if vi is None or not vi.type.HasField('tensor_type') or not vi.type.tensor_type.HasField('shape'):
                vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            seg_output_infos.append(vi)

        graph = helper.make_graph(
            [n for n in seg_nodes],
            seg_name,
            seg_graph_inputs,
            seg_output_infos,
            seg_init
        )
        model = helper.make_model(graph, producer_name='rest_3way_split')
        model.ir_version = 10
        while len(model.opset_import) > 0:
            model.opset_import.pop()
        for opset in m.opset_import:
            oi = model.opset_import.add()
            oi.domain = opset.domain
            oi.version = opset.version

        onnx.save(model, str(seg_path))
        log("split", f"  保存: {seg_path} / Saved: {seg_path}")

    log("split", "完成 / Done")


# ── 校准数据生成 (依赖 torch + onnxruntime) ───────────────────────
# ── Calibration data generation (requires torch + onnxruntime) ───

def _read_calib_lines(calib_dir):
    """读取 input_list.txt 的非空行; 无文件或读取失败返回 []。
    Read non-empty lines of input_list.txt; returns [] when missing or unreadable."""
    il = calib_dir / 'input_list.txt'
    if not il.exists():
        return []
    try:
        return [l.strip() for l in il.read_text(encoding='utf-8').splitlines() if l.strip()]
    except (OSError, UnicodeDecodeError):
        return []


def _trim_calib(calib_dir, target):
    """对齐已有校准样本数量: 行数超过 target 时保留前 target 行并删除多余 raw;
    行数不足则原样保留。返回当前行数。
    Align the existing calib sample count: when lines > target, keep the first
    target lines and delete extra raw files; when fewer, keep as-is. Returns the
    current line count."""
    lines = _read_calib_lines(calib_dir)
    if len(lines) <= target:
        return len(lines)
    keep = lines[:target]
    (calib_dir / 'input_list.txt').write_text('\n'.join(keep) + '\n', encoding='utf-8')
    used = set()
    for line in keep:
        for tok in line.split():
            used.add(os.path.normpath(tok.split('=', 1)[-1]))
    for f in calib_dir.glob('*.raw'):
        if os.path.normpath(str(f)) not in used:
            try:
                f.unlink()
            except OSError:
                pass
    log("calib", f"  已有校准 {len(lines)} 组 > 目标 {target}, 跳过 / "
               f"Existing {len(lines)} calib sets > target {target}, skipping")
    log("calib", f"  已抽取保留前 {target} 组并清理多余 raw / "
               f"Kept first {target} sets, cleaned up extra raw files")
    return target


def _used_img_idx(calib_dir):
    """从已有 raw 文件名解析图片序号 (REST: NNNN_xxx.raw, IE: ie_NNNN.raw)。
    PE 文件名不含图片序号, 不要对该目录使用。
    Parse image indexes from existing raw filenames (REST: NNNN_xxx.raw,
    IE: ie_NNNN.raw). PE filenames carry no image index; do not use this for PE."""
    idxs = set()
    for line in _read_calib_lines(calib_dir):
        for tok in line.split():
            base = Path(tok.split('=', 1)[-1]).stem
            m = re.match(r'^(\d{4})_[^_]', base) or re.match(r'^ie_(\d{4})$', base)
            if m:
                idxs.add(int(m.group(1)))
    return idxs


def _purge_calib(calib_dir):
    """清空该目录的校准样本 (强制重建用)。
    Clear the calib samples in this directory (used for forced rebuild)."""
    for f in calib_dir.glob('*.raw'):
        f.unlink()
    il = calib_dir / 'input_list.txt'
    if il.exists():
        try:
            il.unlink()
        except OSError:
            il.write_text('')


def _gen_pe_ie_calib(img_dir, n_calib, calib_dir, pe_max,
                     want_pe=True, want_ie=True, force_pe=False, force_ie=False):
    """生成/补齐 PE 与 IE 校准数据, 只按样本数量补齐缺失部分。
    Generate/top-up PE and IE calibration data, by sample count only.

    IE 目标组数 = 图片数 (每图 1 组); PE 目标样本数 = 35 × n_per_pos,
    n_per_pos = max(1, min(图片数, pe_max // 35)), 即 35 个 patch 位置
    均匀分配, 总数不超过 pe_max。PE/IE 共享图片池, 同一次前向同时采集
    两个模型的校准输入。只按样本数量补齐缺口: PE 样本无图号追踪,
    图可复用; IE 按图号避免重复组。want_* 控制是否采集对应模型
    (只生成任务内模型); force_* 为 True 时先清空对应子目录再重建。
    IE target = number of images (1 per image); PE target = 35 × n_per_pos,
    n_per_pos = max(1, min(imgs, pe_max // 35)) for even distribution across
    the 35 patch slots, capped at pe_max. PE/IE share the image pool and are
    captured in the same forward pass. Top-up by sample count only: PE samples
    carry no image index so images are reusable; IE avoids repeated groups by
    image index. want_* selects which model to collect; force_* purges the
    corresponding subdir before rebuilding."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    _require_src()
    from sharp.models.encoders.spn_encoder import split

    PE_DIR = calib_dir / 'pe_calib'
    IE_DIR = calib_dir / 'ie_calib'
    for d in (PE_DIR, IE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if force_pe:
        _purge_calib(PE_DIR)
    if force_ie:
        _purge_calib(IE_DIR)

    img_paths = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    np.random.seed(42)
    np.random.shuffle(img_paths)
    img_paths = img_paths[:n_calib]
    if not img_paths:
        fail("校准图片目录中没有可用图片 / No available images in the calibration image directory")

    ie_target = len(img_paths)
    n_per_pos = max(1, min(len(img_paths), pe_max // 35))
    pe_target = 35 * n_per_pos

    ie_k = _trim_calib(IE_DIR, ie_target)
    pe_k = _trim_calib(PE_DIR, pe_target)
    need_ie = max(0, ie_target - ie_k)
    need_pe = max(0, pe_target - pe_k)
    if need_ie == 0 and need_pe == 0:
        log("calib", f"PE/IE 校准数据已完备 (PE {pe_k} 样本, IE {ie_k} 组), 无需更新 / "
                   f"PE/IE calibration complete (PE {pe_k} samples, IE {ie_k} sets), no update needed")
        return

    log("calib", "加载原项目 predictor ... / Loading original project predictor ...")
    predictor = _load_predictor()
    enc = predictor.monodepth_model.monodepth_predictor.encoder
    normalizer = predictor.monodepth_model.monodepth_predictor.normalizer

    ie_used = _used_img_idx(IE_DIR)
    ie_proc = [(i, p) for i, p in enumerate(img_paths) if i not in ie_used][:need_ie]
    pe_proc = []
    if need_pe:
        pool = [(i, p) for i, p in enumerate(img_paths) if i not in ie_used]
        need_imgs = (need_pe + 34) // 35
        if len(pool) < need_imgs:
            pool = list(enumerate(img_paths))  # 只看数量, 图可复用 / count-only, images reusable
        pe_proc = pool[:need_imgs]
    proc_map = {i: p for i, p in (ie_proc + pe_proc)}
    proc = sorted(proc_map.items())
    ie_proc_idx = {i for i, _ in ie_proc}

    log("calib", f"PE/IE 校准: 已有 PE {pe_k}/{pe_target} 样本, IE {ie_k}/{ie_target} 组, "
                 f"补缺 PE {need_pe} / IE {need_ie}, 需前向 {len(proc)} 张图 / "
                 f"PE/IE calib: have PE {pe_k}/{pe_target} samples, IE {ie_k}/{ie_target} sets, "
                 f"need PE {need_pe} / IE {need_ie}, will forward {len(proc)} images")
    patches = {}
    ie_new = []
    n_skip = 0

    for r, (i, p) in enumerate(proc):
        try:
            img_pil, w, h, f_px = _load_rgb_and_fpx(p)
            arr = np.asarray(img_pil, dtype=np.float32) / 255.0
            image_pt = torch.from_numpy(arr).permute(2, 0, 1)
            img = F.interpolate(image_pt[None], size=(1536, 1536), mode="bilinear", align_corners=True)
            x = normalizer(img)
            x0, x1, x2 = enc._create_pyramid(x)
            x0p = split(x0, overlap_ratio=0.25, patch_size=enc.patch_size)
            x1p = split(x1, overlap_ratio=0.50, patch_size=enc.patch_size)
            pe_in = torch.cat([x0p, x1p, x2], dim=0)
            if i in ie_proc_idx:
                ie_new.append((i, x2))
            if want_pe:
                for j in range(35):
                    patches[(i, j)] = pe_in[j:j+1]
            if (r+1) % 5 == 0:
                log("calib", f"  PE/IE 处理 {r+1}/{len(proc)} / Processing {r+1}/{len(proc)}")
        except Exception as e:
            n_skip += 1
            log("calib", f"  skip {p.name}: {e}")

    if n_skip:
        log("calib", f"  warn: {n_skip}/{len(proc)} 张图处理失败, 实际成功 {len(proc) - n_skip} 张 / "
                   f"warn: {n_skip}/{len(proc)} images failed, {len(proc) - n_skip} succeeded")

    if want_ie and ie_new:
        ie_start = max(ie_used) + 1 if ie_used else 0
        with open(IE_DIR / 'input_list.txt', 'a') as f:
            for t, (i, x2) in enumerate(sorted(ie_new)):
                path = IE_DIR / f'ie_{ie_start + t:04d}.raw'
                x2.numpy().astype(np.float32).tofile(path)
                f.write(f'{path}\n')
        if len(ie_new) < need_ie:
            log("calib", f"  warn: IE 补缺 {len(ie_new)}/{need_ie} 组, 不达目标 / "
                   f"warn: IE only filled {len(ie_new)}/{need_ie} sets, below target")

    if want_pe and patches:
        keys = sorted(patches.keys())
        n_new = min(need_pe, len(keys))
        with open(PE_DIR / 'input_list.txt', 'a') as f:
            for s in range(n_new):
                i, j = keys[s]
                path = PE_DIR / f'pe_b1_{pe_k + s:04d}.raw'
                patches[(i, j)].numpy().astype(np.float32).tofile(path)
                f.write(f'{path}\n')
        if n_new < need_pe:
            log("calib", f"  warn: PE 补缺 {n_new}/{need_pe} 样本, 不达目标 / "
                   f"warn: PE only filled {n_new}/{need_pe} samples, below target")

    pe_total = len(_read_calib_lines(PE_DIR))
    ie_total = len(_read_calib_lines(IE_DIR))
    log("calib", f"  PE 校准: 共 {pe_total}/{pe_target} 样本  IE 校准: 共 {ie_total}/{ie_target} 组 / "
                   f"PE calib: {pe_total}/{pe_target} samples  IE calib: {ie_total}/{ie_target} sets")


def _load_rgb_and_fpx(path):
    from PIL import Image
    img_pil = Image.open(path)

    img_exif = img_pil.getexif()
    exif_ifd = img_exif.get_ifd(0x8769)
    for tag_id, value in exif_ifd.items():
        img_exif[tag_id] = value

    orientation = img_exif.get(0x0112, 1)
    if orientation == 3:
        img_pil = img_pil.transpose(Image.ROTATE_180)
    elif orientation == 6:
        img_pil = img_pil.transpose(Image.ROTATE_270)
    elif orientation == 8:
        img_pil = img_pil.transpose(Image.ROTATE_90)
    elif orientation != 1:
        log("calib", f"    warn: ignoring orientation {orientation}")

    img = img_pil.convert("RGB")
    w, h = img.size

    f_35mm = exif_ifd.get(0xA405, exif_ifd.get(0xA40C, None))
    if f_35mm is None or f_35mm < 1:
        f_35mm = exif_ifd.get(0x920A, None)
        if f_35mm is None:
            f_35mm = 30.0
        if f_35mm < 10.0:
            f_35mm *= 8.4

    f_px = f_35mm * math.sqrt(w**2 + h**2) / math.sqrt(36**2 + 24**2)
    return img, w, h, f_px


def _preprocess(path):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    img_pil, w, h, f_px = _load_rgb_and_fpx(path)
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    image_pt = torch.from_numpy(arr).permute(2, 0, 1)
    disparity_factor = torch.tensor([f_px / w], dtype=torch.float32)
    image_resized = F.interpolate(
        image_pt[None], size=(1536, 1536), mode="bilinear", align_corners=True
    )[0]
    return image_resized[None], disparity_factor, f_px, w, h


def _safe(name):
    return name.replace('/', '_').strip('_')


def _gen_rest_calib(img_dir, n_calib, calib_dir, onnx_dir, force=False):
    """生成/补齐 REST 校准数据 (rest_a / rest_b / rest_c), 只补缺失组数。
    Generate/top-up REST calibration data (rest_a / rest_b / rest_c); only fill the missing count.

    目标组数 = 图片数 (每个模型组数一致, 以 rest_a 行数为准)。
    只使用未被已有校准占用的图片, 若可用图片不足则 fail
    (提示补充图片或清空校准目录重建)。force=True 时清空三个
    子目录后重建 (校准数据按家族整体维护)。
    Target = number of images (same for all segs, rest_a line count is the
    authority). Only unused images are picked; fail when not enough unused
    images (suggest adding images or purging calib to rebuild). force=True
    purges all three subdirs first (calib is maintained per family)."""
    import numpy as np
    import torch
    _require_py('onnxruntime')
    import onnxruntime as ort

    seg_paths = [onnx_dir / f'rest_seg_{s}.onnx' for s in ('a', 'b', 'c')]
    for p in seg_paths:
        if not p.exists():
            fail(f"[calib] 缺少 {p}, 请先导出并拆分 rest (onnx rest) / "
        f"[calib] Missing {p}, export and split rest first (onnx rest)")

    CALIB_A = calib_dir / 'rest_seg_a_calib'
    CALIB_B = calib_dir / 'rest_seg_b_calib'
    CALIB_C = calib_dir / 'rest_seg_c_calib'
    for d in (CALIB_A, CALIB_B, CALIB_C):
        d.mkdir(parents=True, exist_ok=True)
    if force:
        for d in (CALIB_A, CALIB_B, CALIB_C):
            _purge_calib(d)

    img_paths = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    np.random.seed(42)
    np.random.shuffle(img_paths)
    img_paths = img_paths[:n_calib]
    if not img_paths:
        fail("校准图片目录中没有可用图片 / No available images in the calibration image directory")

    target = len(img_paths)
    a_k = _trim_calib(CALIB_A, target)
    b_k = _trim_calib(CALIB_B, target)
    c_k = _trim_calib(CALIB_C, target)
    k = min(a_k, b_k, c_k)
    need = max(0, target - k)
    if need == 0:
        log("calib", f"REST 校准数据已完备 ({k} 组), 无需更新 / "
                   f"REST calibration complete ({k} sets), no update needed")
        return
    if k < target:
        for d in (CALIB_A, CALIB_B, CALIB_C):
            if len(_read_calib_lines(d)) > k:
                _trim_calib(d, k)

    _require_src()
    predictor = _load_predictor()
    enc = predictor.monodepth_model.monodepth_predictor.encoder
    normalizer = predictor.monodepth_model.monodepth_predictor.normalizer

    captured = {}
    def _hook(name):
        def fn(_m, inp, _out):
            captured[name] = inp[0].detach().clone()
        return fn
    handles = [
        enc.upsample_latent0.register_forward_hook(_hook('x_latent0')),
        enc.upsample_latent1.register_forward_hook(_hook('x_latent1')),
        enc.upsample0.register_forward_hook(_hook('x0_feat')),
        enc.upsample1.register_forward_hook(_hook('x1_feat')),
        enc.upsample2.register_forward_hook(_hook('x2_feat')),
        enc.upsample_lowres.register_forward_hook(_hook('x_lowres_feat')),
    ]

    log("calib", "加载 rest_a / rest_b / rest_c ONNX ... / Loading rest_a / rest_b / rest_c ONNX ...")
    sess_a = ort.InferenceSession(str(seg_paths[0]), providers=['CPUExecutionProvider'])
    sess_b = ort.InferenceSession(str(seg_paths[1]), providers=['CPUExecutionProvider'])
    sess_c = ort.InferenceSession(str(seg_paths[2]), providers=['CPUExecutionProvider'])
    a_in_names  = [i.name for i in sess_a.get_inputs()]
    a_out_names = [o.name for o in sess_a.get_outputs()]
    b_in_names  = [i.name for i in sess_b.get_inputs()]
    b_out_names = [o.name for o in sess_b.get_outputs()]
    c_in_names  = [i.name for i in sess_c.get_inputs()]

    used = _used_img_idx(CALIB_A)
    pool = [(i, p) for i, p in enumerate(img_paths) if i not in used]
    new_imgs = pool[:need]
    if len(new_imgs) < need:
        fail(f"可用的未使用图片不足: 需补 {need} 组, 仅 {len(new_imgs)} 张未用图片。"
             f"请补充图片到 {img_dir} 或清空校准目录重建 / "
             f"Not enough unused images: need {need} more sets, {len(new_imgs)} unused."
             f" Add images to {img_dir} or purge calib to rebuild")
    log("calib", f"REST 校准: 已有 {k}/{target} 组, 补缺 {len(new_imgs)} 组 "
                 f"(图片: {', '.join(p.name for _, p in new_imgs[:5])}{'...' if len(new_imgs) > 5 else ''}) / "
                 f"REST calib: have {k}/{target} sets, filling {len(new_imgs)} sets "
                 f"(images: {', '.join(p.name for _, p in new_imgs[:5])}{'...' if len(new_imgs) > 5 else ''})")

    a_lines, b_lines, c_lines = [], [], []
    n_skip = 0

    for r, (idx, p) in enumerate(new_imgs):
        try:
            img_t, d_factor_t, f_px, w, h = _preprocess(p)
            d_factor = d_factor_t.numpy().astype(np.float32)
            image_np = img_t.numpy().astype(np.float32)

            with torch.no_grad():
                x_norm = normalizer(img_t)
                enc(x_norm)

            feats = {name: captured[name].numpy().astype(np.float32) for name in
                     ['x_latent0', 'x_latent1', 'x0_feat', 'x1_feat', 'x2_feat', 'x_lowres_feat']}

            a_line = []
            for name in a_in_names:
                raw = CALIB_A / f'{idx:04d}_{_safe(name)}.raw'
                feats[name].tofile(raw)
                a_line.append(f'{name}:={raw}')
            a_lines.append(' '.join(a_line))

            feeds_a = {n: feats[n] for n in a_in_names}
            outputs_a = sess_a.run(None, feeds_a)
            edge_a = dict(zip(a_out_names, outputs_a))

            b_line = []
            for name in b_in_names:
                raw = CALIB_B / f'{idx:04d}_{_safe(name)}.raw'
                edge_a[name].astype(np.float32).tofile(raw)
                b_line.append(f'{name}:={raw}')
            b_lines.append(' '.join(b_line))

            feeds_b = {n: edge_a[n] for n in b_in_names}
            outputs_b = sess_b.run(None, feeds_b)
            edge_b = dict(zip(b_out_names, outputs_b))

            c_line = []
            for name in c_in_names:
                if name == 'image':
                    raw = CALIB_C / f'{idx:04d}_image.raw'
                    image_np.tofile(raw)
                elif name == 'disparity_factor':
                    raw = CALIB_C / f'{idx:04d}_disparity_factor.raw'
                    d_factor.tofile(raw)
                elif name in edge_a:
                    raw = CALIB_C / f'{idx:04d}_{_safe(name)}.raw'
                    edge_a[name].astype(np.float32).tofile(raw)
                elif name in edge_b:
                    raw = CALIB_C / f'{idx:04d}_{_safe(name)}.raw'
                    edge_b[name].astype(np.float32).tofile(raw)
                else:
                    raise KeyError(f'rest_c 输入 {name} 无来源')
                c_line.append(f'{name}:={raw}')
            c_lines.append(' '.join(c_line))

            disp = edge_b['disparity']
            log("calib", f"  [{r+1:2d}/{len(new_imgs)}] {p.name[:35]:35s} "
                         f"{w}x{h} f_px={f_px:.1f} d_fac={d_factor[0]:.4f} "
                         f"disp=[{disp.min():.3f},{disp.max():.3f}]")
        except Exception as e:
            n_skip += 1
            log("calib", f"  skip {p.name}: {e}")

    if n_skip:
        log("calib", f"  warn: {n_skip}/{len(new_imgs)} 张图处理失败, 实际新增 {len(new_imgs) - n_skip} 组 / "
                   f"warn: {n_skip}/{len(new_imgs)} images failed, actually added {len(new_imgs) - n_skip} sets")
    if len(a_lines) < need:
        log("calib", f"  warn: 新增 {len(a_lines)} 组 < 需求 {need}, 校准代表性不足, 建议检查图片目录 / "
                   f"warn: added {len(a_lines)} sets < needed {need}, calibration may be unrepresentative")

    for d, lines in ((CALIB_A, a_lines), (CALIB_B, b_lines), (CALIB_C, c_lines)):
        il = d / 'input_list.txt'
        old = _read_calib_lines(d)
        il.write_text('\n'.join(old + lines) + '\n', encoding='utf-8')

    for h in handles:
        h.remove()

    log("calib", f"  rest_a / rest_b / rest_c: 共 {len(_read_calib_lines(CALIB_A))} 组 (目标 {target}) / "
                   f"rest_a / rest_b / rest_c: {len(_read_calib_lines(CALIB_A))} total sets (target {target})")


def gen_calib(img_dir, n_calib, out, pe_max):
    _require_py('numpy')
    _require_py('PIL')
    if not img_dir.exists():
        fail(f"校准图片目录不存在: {img_dir}\n请用 -i 指定含 jpg/png 的目录 / "
        f"Calibration image directory not found: {img_dir}\nUse -i to specify a directory with jpg/png files")
    if not len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))):
        fail(f"校准图片目录 {img_dir} 中没有 jpg/png 图片 / "
        f"No jpg/png images found in {img_dir}")

    _check_disk('all', n_calib, out, None, 'calib')

    calib_dir = out / OUT_CALIB
    onnx_dir = out / OUT_ONNX

    log("calib", "生成 PE/IE 校准数据 (pe.dlc / ie.dlc) ... / Generating PE/IE calibration data (pe.dlc / ie.dlc) ...")
    _gen_pe_ie_calib(img_dir, n_calib, calib_dir, pe_max)

    log("calib", "生成 REST 校准数据 (rest_a / rest_b / rest_c) ... / Generating REST calibration data (rest_a / rest_b / rest_c) ...")
    _gen_rest_calib(img_dir, n_calib, calib_dir, onnx_dir)

    log("calib", "完成 / Done")


# ── DLC 转换 (依赖 QNN SDK) ───────────────────────────────────────
# ── DLC conversion (requires QNN SDK) ────────────────────────────

def _host_bin_dir():
    if platform.system() == 'Windows':
        return 'x86_64-windows-msvc'
    return 'x86_64-linux-clang'


def _resolve_sdk(sdk_arg):
    if sdk_arg:
        p = Path(sdk_arg)
        if p.exists():
            return p
    for var in ('QAIRT_SDK_ROOT', 'QNN_SDK_ROOT', 'QNN_SDK_PATH'):
        v = os.environ.get(var, '').strip()
        if v:
            p = Path(v)
            if p.exists():
                return p
    for root in (Path(os.environ.get('SystemDrive', 'C:') + '\\QNN') if platform.system() == 'Windows'
                 else Path('/opt/qcom/aistack/qairt'),):
        try:
            cands = [root] + sorted(root.iterdir()) if root.exists() else []
        except OSError:
            cands = []
        for p in cands:
            if p.exists() and (p / 'bin').exists():
                return p
    return None


def _find_tool(sdk, name_candidates):
    if not sdk:
        return None
    bin_dirs = []
    for base in (sdk / 'bin', sdk / 'qairt' / 'bin'):
        bin_dirs.append(base / _host_bin_dir())
        bin_dirs.append(base)
    for name in name_candidates:
        for d in bin_dirs:
            for cand in (d / name, d / (name + '.exe')):
                if cand.exists():
                    return cand
    return None


def _sdk_env(sdk, temp_dir=None):
    env = os.environ.copy()
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        env['TEMP'] = str(temp_dir)
        env['TMP'] = str(temp_dir)
    py_lib = sdk / 'lib' / 'python'
    if py_lib.exists():
        sep = ';' if platform.system() == 'Windows' else ':'
        parts = [p for p in env.get('PYTHONPATH', '').split(sep) if p and Path(p) != py_lib]
        parts.append(str(py_lib))
        env['PYTHONPATH'] = sep.join(parts)
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def _run_cmd(cmd, env=None):
    log("convert", f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            encoding='utf-8', errors='replace')
    if result.returncode != 0:
        if result.stderr:
            log("convert", f"  STDERR: {result.stderr[-800:]}")
        if result.stdout:
            log("convert", f"  STDOUT: {result.stdout[-800:]}")
        fail(f"命令失败 (exit {result.returncode}) / Command failed (exit {result.returncode})")
    return result.stdout


def _ensure_onnx(scope, out_dir):
    """确保 ONNX 已导出; 缺失时自动导出 (需要 torch 环境)。
    Ensure the ONNX is exported; auto-export when missing (requires torch env)."""
    need = []
    if scope in ('all', 'pe') and not (out_dir / 'patch_encoder.onnx').exists():
        need.append('pe')
    if scope in ('all', 'ie') and not (out_dir / 'image_encoder.onnx').exists():
        need.append('ie')
    if scope in ('all', 'rest'):
        if not (out_dir / 'rest_seg_a.onnx').exists():
            need.append('rest')
    if need:
        log("dlc", f"ONNX 缺失 ({', '.join(need)}), 自动导出 ... / "
            f"ONNX missing ({', '.join(need)}), auto-exporting ...")
        export_onnx(scope, out_dir)


def _convert_one(onnx_path, fmt, sdk, out, temp_dir):
    fmt_cfg = FORMAT_MAP[fmt]
    converter = _find_tool(sdk, CONVERTER_NAMES)
    if not converter:
        fail(f"找不到 qairt-converter, SDK: {sdk} / qairt-converter not found, SDK: {sdk}")
    quantizer = _find_tool(sdk, QUANTIZER_NAMES)
    if not quantizer:
        fail(f"找不到 qairt-quantizer, SDK: {sdk} / qairt-quantizer not found, SDK: {sdk}")

    env = _sdk_env(sdk, temp_dir)
    py = sys.executable

    fp32_dir = out / OUT_DLC_FP32
    quant_dir = out / OUT_DLC_QUANT.format(fmt=fmt)
    fp32_dir.mkdir(parents=True, exist_ok=True)
    quant_dir.mkdir(parents=True, exist_ok=True)

    # 使用 App 期望的短名称 / Use short name expected by the App
    dlc_stem = DLC_NAME_MAP.get(onnx_path.stem, onnx_path.stem)
    fp32_dlc = fp32_dir / f'{dlc_stem}.dlc'
    quant_dlc = quant_dir / f'{dlc_stem}.dlc'

    input_list = _calib_dir_for(onnx_path, out) / 'input_list.txt'
    if not input_list.exists():
        fail(f"校准数据不存在: {input_list}\n请先运行: python build_rest_pipeline.py calib\n"
        f"Calibration data missing: {input_list}\nRun first: python build_rest_pipeline.py calib")

    log("convert", f"  [1/2] converter: {onnx_path.name} -> {fp32_dlc.name}")
    _run_cmd([py, str(converter), '--input_network', str(onnx_path),
              '--output_path', str(fp32_dlc)], env=env)
    log("convert", f"    FP32 DLC: {fp32_dlc.stat().st_size/1e6:.0f} MB")

    log("convert", f"  [2/2] quantizer: {fmt} (W{fmt_cfg['w']}A{fmt_cfg['a']})")
    _run_cmd([py, str(quantizer), '--input_dlc', str(fp32_dlc),
              '--output_dlc', str(quant_dlc),
              '--weights_bitwidth', str(fmt_cfg['w']),
              '--act_bitwidth', str(fmt_cfg['a']),
              '--input_list', str(input_list)], env=env)
    log("convert", f"    {fmt} DLC: {quant_dlc.stat().st_size/1e6:.0f} MB")


def _calib_dir_for(onnx_path, out):
    """根据 ONNX 文件名推断对应校准数据目录。
    Infer the calibration data directory from the ONNX filename.
    ONNX 中间名 → 校准目录映射 / ONNX stem → calib dir mapping:
      patch_encoder → pe_calib
      image_encoder → ie_calib
      rest_seg_a/b/c → rest_seg_a/b/c_calib"""
    stem = onnx_path.stem
    if stem == 'patch_encoder':
        return out / OUT_CALIB / 'pe_calib'
    if stem == 'image_encoder':
        return out / OUT_CALIB / 'ie_calib'
    return out / OUT_CALIB / f'{stem}_calib'


def _calib_raw_valid(onnx_path, out):
    """校验该模型的校准数据是否完整有效:
    input_list.txt 存在, 且每个 raw 文件大小 == 模型输入元素数×4 (float32)。
    任何一项不符视为无效 (会重新生成)。
    输入列表支持两种格式: PE/IE 单输入模型的纯路径格式 (每行一个路径),
    REST 多输入模型的 name:=path 格式 (多输入模型不接受纯路径格式)。
    Validate the model's calibration data completeness:
    input_list.txt exists and every raw file size == input element count × 4
    (float32). Any mismatch marks the data invalid (it will be regenerated).
    Two input-list formats: bare-path (one path per line, PE/IE single-input
    only) and name:=path (REST multi-input; bare-path not accepted)."""
    import onnx
    d = _calib_dir_for(onnx_path, out)
    il = d / 'input_list.txt'
    if not il.exists():
        return False
    try:
        m = onnx.load(str(onnx_path))
    except Exception:
        return False
    expect = {}
    for inp in m.graph.input:
        t = inp.type.tensor_type
        if not t.HasField('shape'):
            return False
        dims = [dim.dim_value for dim in t.shape.dim]
        if any(x <= 0 for x in dims):
            continue
        expect[inp.name] = math.prod(dims)
    try:
        lines = [l.strip() for l in il.read_text(encoding='utf-8').splitlines() if l.strip()]
    except OSError:
        return False
    if not lines:
        return False
    covered = set()
    for line in lines:
        toks = line.split()
        if not toks:
            continue
        if '=' in line:
            for tok in toks:
                name, path = tok.split('=', 1)
                name = name.rstrip(':')
                if name not in expect:
                    return False
                rp = Path(path)
                try:
                    if not rp.exists() or rp.stat().st_size != expect[name] * 4:
                        return False
                except OSError:
                    return False
                covered.add(name)
        else:
            if len(expect) != 1:
                return False
            (name,) = list(expect)   # PE/IE 纯路径格式 (单输入模型)
                                     # bare-path format (single-input model)
            rp = Path(toks[0])
            try:
                if not rp.exists() or rp.stat().st_size != expect[name] * 4:
                    return False
            except OSError:
                return False
            covered.add(name)
    if expect and covered != set(expect):
        return False
    return True


def _disk_free(path):
    p = Path(path)
    while not p.exists():
        pp = p.parent
        if pp == p:
            break
        p = pp
    return shutil.disk_usage(str(p)).free


def _estimate_space(scope, n_calib):
    """预估本次指令在 <out> 下需要的字节数 (ONNX + 校准 + fp32/量化 DLC, 未扣已有文件)。
    校准量最大, 按每张图的 raw 体积估算。
    Estimate bytes needed under <out> for this run (ONNX + calib + fp32/quantized
    DLC, not subtracting existing files). Calib dominates; estimated per-image raw size.
    DLC 输出: pe.dlc / ie.dlc / rest_a.dlc / rest_b.dlc / rest_c.dlc
    DLC output: pe.dlc / ie.dlc / rest_a.dlc / rest_b.dlc / rest_c.dlc"""
    est = 0.0
    if scope in ('all', 'pe'):
        est += 0.40e9 + 45e6 * n_calib   # pe: onnx+dlc + pe/ie 校准
                                         # pe: onnx+dlc + pe/ie calib
    if scope in ('all', 'ie'):
        est += 0.40e9 + 45e6 * n_calib   # ie: onnx+dlc + pe/ie 校准
                                         # ie: onnx+dlc + pe/ie calib
    if scope in ('all', 'rest'):
        est += 0.45e9 + 600e6 * n_calib  # rest: onnx+seg dlc + rest_a/b/c 校准 (~600MB/图)
                                         # rest: onnx+seg dlc + rest_a/b/c calib (~600MB per image)
    return est


def _check_disk(scope, n_calib, out, temp_dir, stage):
    need = _estimate_space(scope, n_calib)
    free = _disk_free(out)
    log(stage, f"空间预估: 需要 ~{need/1e9:.1f} GB, {out} 所在盘剩余 {free/1e9:.1f} GB / "
            f"Space estimate: need ~{need/1e9:.1f} GB, {free/1e9:.1f} GB free on {out}")
    if free < need * 1.15:
        fail(f"磁盘空间不足: {out} 所在盘剩余 {free/1e9:.1f} GB, "
             f"本次任务预估需要 {need/1e9:.1f} GB (含 15% 余量)。\n"
             f"请清理磁盘或改用 -o 指定空间充足的目录 / "
             f"Insufficient disk space: {free/1e9:.1f} GB free on {out}, "
             f"estimated need {need/1e9:.1f} GB (with 15% margin).\n"
             f"Free up disk space or use -o to specify a directory with enough space")
    if temp_dir:
        tfree = _disk_free(temp_dir)
        log(stage, f"临时目录 {temp_dir} 所在盘剩余 {tfree/1e9:.1f} GB / "
                f"Temp dir {temp_dir}: {tfree/1e9:.1f} GB free")
        if tfree < 10e9:
            fail(f"临时目录 {temp_dir} 所在盘剩余 {tfree/1e9:.1f} GB < 10 GB, "
                 f"converter 需要 8-10 GB 临时空间。请更换 --temp_dir / "
                 f"Temp dir {temp_dir} has only {tfree/1e9:.1f} GB < 10 GB, "
                 f"converter needs 8-10 GB temp space. Use --temp_dir to change")


def _gen_calib_for(tasks, img_dir, n_calib, out, pe_max, force=frozenset()):
    """按需生成校准数据: 只生成 tasks 中模型对应的部分。
    force: 无效需强制重建的模型 stem 集合 (重建整个对应家族)。
    Generate calib data on demand: only the parts needed by the models in tasks.
    force: set of model stems whose invalid calib must be force-rebuilt (the whole
    corresponding family is rebuilt)."""
    calib_dir = out / OUT_CALIB
    onnx_dir = out / OUT_ONNX
    stems = {t.stem for t in tasks}
    force = set(force)
    want_pe = 'patch_encoder' in stems
    want_ie = 'image_encoder' in stems
    if want_pe or want_ie:
        log("calib", "生成 PE/IE 校准数据 (pe.dlc / ie.dlc) ... / Generating PE/IE calibration data (pe.dlc / ie.dlc) ...")
        _gen_pe_ie_calib(img_dir, n_calib, calib_dir, pe_max,
                         want_pe=want_pe, want_ie=want_ie,
                         force_pe='patch_encoder' in force,
                         force_ie='image_encoder' in force)
    if stems & {'rest_seg_a', 'rest_seg_b', 'rest_seg_c'}:
        log("calib", "生成 REST 校准数据 (rest_a.dlc / rest_b.dlc / rest_c.dlc) ... / Generating REST calibration data (rest_a.dlc / rest_b.dlc / rest_c.dlc) ...")
        _gen_rest_calib(img_dir, n_calib, calib_dir, onnx_dir,
                        force=bool(force & {'rest_seg_a', 'rest_seg_b', 'rest_seg_c'}))
    log("calib", "完成 / Done")


def export_dlc(scope, fmt, sdk_arg, out, temp_dir, img_dir, n_calib, pe_max):
    _require_py('onnx')
    out.mkdir(parents=True, exist_ok=True)
    onnx_dir = out / OUT_ONNX

    sdk = _resolve_sdk(sdk_arg)
    if not sdk:
        fail("找不到 QNN SDK。请用 --sdk <PATH> 指定, 或设置环境变量 QAIRT_SDK_ROOT / QNN_SDK_ROOT / QNN_SDK_PATH / "
        "QNN SDK not found. Use --sdk <PATH> or set env var QAIRT_SDK_ROOT / QNN_SDK_ROOT / QNN_SDK_PATH")
    log("dlc", f"QNN SDK: {sdk}")
    log("dlc", f"输出目录: {out} / Output dir: {out}")

    _check_disk(scope, n_calib, out, temp_dir, 'dlc')

    _ensure_onnx(scope, onnx_dir)

    tasks = []
    if scope in ('all', 'pe'):
        p = onnx_dir / 'patch_encoder.onnx'
        if not p.exists():
            fail(f"缺少 {p}, 请先导出 ONNX / Missing {p}, run ONNX export first")
        tasks.append(p)
    if scope in ('all', 'ie'):
        p = onnx_dir / 'image_encoder.onnx'
        if not p.exists():
            fail(f"缺少 {p}, 请先导出 ONNX / Missing {p}, run ONNX export first")
        tasks.append(p)
    if scope in ('all', 'rest'):
        for seg in ('a', 'b', 'c'):
            p = onnx_dir / f'rest_seg_{seg}.onnx'
            if not p.exists():
                fail(f"缺少 {p}, 请先导出并拆分 rest (onnx rest) / "
                f"Missing {p}, export and split rest first (onnx rest)")
            tasks.append(p)

    missing = {p.stem for p in tasks if not _calib_raw_valid(p, out)}
    if missing:
        log("dlc", f"校准数据缺失或无效 ({', '.join(sorted(missing))}), 自动生成 "
                   f"(img_dir={img_dir}, n_calib={n_calib}, pe_max={pe_max}) ... / "
                   f"Calibration missing or invalid ({', '.join(sorted(missing))}), "
                   f"auto-generating (img_dir={img_dir}, n_calib={n_calib}, pe_max={pe_max}) ...")
        _gen_calib_for(tasks, img_dir, n_calib, out, pe_max, force=missing)
        still = [p.stem for p in tasks if not _calib_raw_valid(p, out)]
        if still:
            fail(f"校准数据生成后仍无效 ({', '.join(still)}), 请检查图片目录与磁盘空间 / "
                f"Calibration still invalid after generation ({', '.join(still)}), check image dir and disk space")

    for onnx_path in tasks:
        dlc_stem = DLC_NAME_MAP.get(onnx_path.stem, onnx_path.stem)
        log("dlc", f"\n--- {dlc_stem}.dlc ({onnx_path.stem}) ---")
        _simplify_onnx(onnx_path)
        _convert_one(onnx_path, fmt, sdk, out, temp_dir)

    log("dlc", "完成 / Done")


# ── 主入口 ────────────────────────────────────────────────────────
# ── Main entry ───────────────────────────────────────────────────

class _WideHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """加宽 help 列, 让参数与说明显示在同一行。
    Widen the help columns so arguments and their descriptions fit on one line.
    支持 help 文本内换行 (中英分行), 每段再按宽度折行。
    Keep explicit newlines in help text (CN/EN on separate lines)."""

    def __init__(self, prog, **kwargs):
        kwargs.setdefault('max_help_position', 60)
        kwargs.setdefault('width', 100)
        super().__init__(prog, **kwargs)

    def _split_lines(self, text, width):
        import textwrap
        lines = []
        for seg in text.split('\n'):
            t = self._whitespace_matcher.sub(' ', seg).strip()
            if t:
                lines.extend(textwrap.wrap(t, width))
        return lines


def main():
    parser = argparse.ArgumentParser(
        description='SHARP 模型部署管线构建: ONNX 导出 / rest 拆分 / 校准 / DLC 转换\n'
                    'SHARP model deployment pipeline builder: ONNX export / rest split /\n'
                    'calibration / DLC conversion',
        formatter_class=_WideHelpFormatter)
    parser.add_argument('-t', '--task', default='dlc', metavar='<task>',
                        help='任务: onnx=导出ONNX / dlc=导出DLC / calib=生成校准数据 (默认: dlc)\n'
                             'Task: onnx=export ONNX / dlc=export DLC / calib=generate calib data (default: dlc)')
    parser.add_argument('-a', '--scope', default='all', metavar='<scope>',
                        help='模型: all=全部 / pe / ie / rest=含拆分三段 (默认: all)\n'
                             'Models: all / pe / ie / rest=with 3 seg split (default: all)')
    parser.add_argument('-o', '--out', default=str(DEFAULT_OUT), metavar='<out>',
                        help='导出根目录 (默认: output/, 自动创建, 分onnx/dlc/calib子目录)\n'
                             'Export root (default: output/, auto-created; onnx/dlc/calib subdirs)')
    parser.add_argument('-f', '--format', choices=list(FORMAT_MAP.keys()), default='w8a16', metavar='<fmt>',
                        help='量化格式: int16 / int8 / w8a16 (默认: w8a16)\n'
                             'Quantization format: int16 / int8 / w8a16 (default: w8a16)')
    parser.add_argument('--sdk', default='', metavar='<sdk_dir>',
                        help='QNN SDK根目录 (默认自动探测环境变量)\n'
                             'QNN SDK root (auto-detected via env by default)')
    parser.add_argument('--temp_dir', default=None, metavar='<dir>',
                        help='converter临时目录 (默认: 系统TEMP, 需8-10GB)\n'
                             'Converter temp dir (default: system TEMP, 8-10GB needed)')
    parser.add_argument('-i', '--img_dir', default=str(DEFAULT_IMG_DIR), metavar='<img_dir>',
                        help='校准图片目录 (默认: data/)\n'
                             'Calibration image dir (default: data/)')
    parser.add_argument('-n', '--n_calib', type=int, default=N_CALIB, metavar='<n_calib>',
                        help='校准图片数量 (默认: 20)\n'
                             'Calibration image count (default: 20)')
    parser.add_argument('--pe_max', type=int, default=70, metavar='<n>',
                        help='PE校准样本总数上限, 35个位置均匀分配 (默认: 70)\n'
                             'PE calib sample cap, evenly over 35 slots (default: 70)')
    args = parser.parse_args()

    if args.task not in ('onnx', 'dlc', 'calib'):
        fail(f"未知任务: {args.task} (可选: onnx / dlc / calib) / "
        f"Unknown task: {args.task} (valid: onnx / dlc / calib)")
    if args.scope not in ('all', 'pe', 'ie', 'rest'):
        fail(f"未知模型范围: {args.scope} (可选: all / pe / ie / rest) / "
        f"Unknown scope: {args.scope} (valid: all / pe / ie / rest)")

    out = _resolve_out(args.out)
    t0 = time.perf_counter()
    print("=" * 60)
    print(f"SHARP 模型部署管线  (task={args.task}, scope={args.scope}, out={out}) / "
          f"SHARP deployment pipeline  (task={args.task}, scope={args.scope}, out={out})")
    print("=" * 60)

    if args.task == 'onnx':
        export_onnx(args.scope, out / OUT_ONNX)
    elif args.task == 'dlc':
        export_dlc(args.scope, args.format, args.sdk, out, args.temp_dir,
                   Path(args.img_dir), args.n_calib, args.pe_max)
    elif args.task == 'calib':
        gen_calib(Path(args.img_dir), args.n_calib, out, args.pe_max)
    else:
        fail(f"未知任务: {args.task} / Unknown task: {args.task}")

    print(f"\n完成! 总耗时 {time.perf_counter() - t0:.1f} s / "
          f"Done! Total time {time.perf_counter() - t0:.1f} s")
    print("=" * 60)


if __name__ == '__main__':
    main()