// qnn_runtime.h — QNN HTP 运行时接口
// qnn_runtime.h — QNN HTP runtime interface
// 封装 QNN HTP backend 的加载、context binary 加载、graph 推理
// Wraps QNN HTP backend loading, context binary loading, and graph inference
#pragma once

#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"

namespace qnn {

// HTP 架构版本
// HTP architecture versions
enum class HtpArch {
    V68 = 68,
    V69 = 69,
    V73 = 73,
    V75 = 75,
    V79 = 79,
    V81 = 81
};

// 推理输入输出张量
// Inference input/output tensor
struct Tensor {
    std::string name;
    std::vector<uint32_t> dims;    // 维度 (NHWC 或 NCHW, 取决于模型) / dims (NHWC or NCHW, model-dependent)
    float* data;                   // 数据指针: 默认指向 float32; quantized=true 时指向量化原始字节 (uint16/uint8)
                                   // data pointer: float32 by default; raw quantized bytes (uint16/uint8) when quantized=true
    size_t count;                  // 元素数 / element count
    bool quantized;                // data 指向量化原始数据 (uint16/uint8, 按 dims/模型 info)
                                   // data points to raw quantized data (uint16/uint8, per dims/model info)
};

// 进度回调
// Progress callback
using ProgressCallback = void (*)(int current, int total, long elapsedMs, const char* detail);

// 4KB 对齐缓冲: QNN clientBuf 官方建议对齐分配 (HTP DMA 传输最稳妥)
// 4KB-aligned buffer: QNN clientBuf is recommended to be aligned (safest for HTP DMA transfer)
// 替代 std::vector<uint8_t> (默认 16 字节对齐, 不满足 QNN 对齐建议)
// Replaces std::vector<uint8_t> (default 16-byte alignment, below the QNN recommendation)
class AlignedBuffer {
public:
    AlignedBuffer() : m_data(nullptr), m_size(0) {}
    ~AlignedBuffer() { deallocate(); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept : m_data(other.m_data), m_size(other.m_size) {
        other.m_data = nullptr;
        other.m_size = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    uint8_t* data() { return m_data; }
    const uint8_t* data() const { return m_data; }
    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }

    // 保证至少 bytes 字节容量 (4KB 对齐); 原缓冲足够大则复用
    // Ensures at least `bytes` of capacity (4KB aligned); reuses the existing buffer when large enough
    void ensure(size_t bytes) {
        if (bytes <= m_size && m_data) return;
        deallocate();
        if (bytes == 0) return;
        void* p = nullptr;
        if (posix_memalign(&p, 4096, bytes) != 0) {
            throw std::bad_alloc();
        }
        m_data = static_cast<uint8_t*>(p);
        m_size = bytes;
    }

    void clear() { deallocate(); }

private:
    uint8_t* m_data;
    size_t m_size;

    void deallocate() {
        if (m_data) {
            ::free(m_data);
            m_data = nullptr;
            m_size = 0;
        }
    }
};

// HTP 性能配置 (由调用方经 JNI 下发, nativeInit 时随共享状态创建生效)
// HTP performance config (sent down via JNI by the caller, applied when the shared state is created in nativeInit)
// type: 0 = 锁角模式 (min=target=max=lockedCorner, 频率恒定);
//       0 = locked-corner mode (min=target=max=lockedCorner, fixed frequency);
//       1 = 自动调角模式 (DCVS 在 [minCorner, maxCorner] 区间内按 dcvsMode 策略动态调节)
//       1 = adaptive mode (DCVS adjusts dynamically in [minCorner, maxCorner] per the dcvsMode policy)
// 电压角取值见 QnnHtpPerfInfrastructure_VoltageCorner_t (0x20 MIN ~ 0xA0 MAX, 不含 DISABLE 0x10)
// Voltage corners per QnnHtpPerfInfrastructure_VoltageCorner_t (0x20 MIN ~ 0xA0 MAX, excluding DISABLE 0x10)
// dcvsMode 取值见 QnnHtpPerfInfrastructure_PowerMode_t (0x1 ADJUST_UP_DOWN ~ 0x20 DUTY_CYCLE)
// dcvsMode per QnnHtpPerfInfrastructure_PowerMode_t (0x1 ADJUST_UP_DOWN ~ 0x20 DUTY_CYCLE)
struct PerfConfig {
    int type = 0;                            // 0=锁角, 1=自动调角 / 0=locked corner, 1=adaptive
    uint32_t lockedCorner = 0xA0;            // 锁角模式: 锁定的电压角 (默认最高档) / locked corner: the corner to lock (max by default)
    uint32_t minCorner = 0x60;               // 自动调角: 允许 DCVS 下调的最低角 (NOM) / adaptive: lowest corner DCVS may drop to (NOM)
    uint32_t targetCorner = 0xA0;            // 自动调角: 投票目标角 (初始请求点, 默认最高) / adaptive: voting target corner (initial request, max by default)
    uint32_t maxCorner = 0xA0;               // 自动调角: 允许 DCVS 上调的最高角 (MAX) / adaptive: highest corner DCVS may raise to (MAX)
    uint32_t dcvsMode = 0x1;                 // 自动调角: DCVS 调节策略 (默认 ADJUST_UP_DOWN) / adaptive: DCVS policy (ADJUST_UP_DOWN by default)
};

// QNN 共享状态: log/backend/device 在多个 HtpRuntime 间共享
// QNN shared state: log/backend/device are shared across HtpRuntime instances
// (官方标准: 单 backend + 单 device + 多 context; 官方示例即此结构)
// (official pattern: one backend + one device + multiple contexts; the official samples use this layout)
// 由调用方 (sharp_jni.cpp) 持有全局实例; HtpRuntime 通过引用计数共享生命周期
// Owned globally by the caller (sharp_jni.cpp); HtpRuntime shares the lifetime via reference counting
struct QnnSharedState {
    void* libHtpHandle = nullptr;            // dlopen 的 libQnnHtp.so (shared, 引用计数管理) / dlopen'd libQnnHtp.so (shared, ref-counted)
    void* libSystemHandle = nullptr;         // dlopen 的 libQnnSystem.so (shared, 引用计数管理) / dlopen'd libQnnSystem.so (shared, ref-counted)
    void* logHandle = nullptr;               // QnnLog handle
    void* backendHandle = nullptr;           // QnnBackend handle
    void* deviceHandle = nullptr;            // QnnDevice handle
    QnnInterface_t qnnInterface{};           // 版本匹配后的 QNN core interface / QNN core interface after version matching
    QnnSystemInterface_t qnnSystemInterface{};  // 版本匹配后的 QNN system interface / QNN system interface after version matching
    QnnHtpDevice_PerfInfrastructure_t perfInfra = QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT;
    uint32_t perfConfigId = 0;               // 0 = 未创建 power config id / 0 = no power config id created yet
    int refCount = 0;                        // 引用计数 (本进程内共享的 HtpRuntime 数) / refcount (number of HtpRuntime instances sharing this state)
};

// QNN HTP 运行时
// QNN HTP runtime
class HtpRuntime {
public:
    // shared: 本进程共享的 QnnSharedState (由调用方持有, 生命周期长于所有 runtime)
    // shared: process-wide QnnSharedState (owned by the caller, outlives every runtime)
    explicit HtpRuntime(QnnSharedState* shared);
    ~HtpRuntime();

    // 初始化: 若共享状态尚未创建, 则 dlopen libQnnHtp.so + libQnnSystem.so,
    // init: if the shared state does not exist yet, dlopen libQnnHtp.so + libQnnSystem.so,
    // 创建 log/backend/device + HTP 性能配置; 之后引用计数 +1
    // create log/backend/device + HTP performance config; then bump the refcount
    // libDir: QNN .so 所在目录 (如 /data/data/com.sharp.qnn/lib/arm64)
    // libDir: directory of the QNN .so files (e.g. /data/data/com.sharp.qnn/lib/arm64)
    // skelDir: Skel .so 所在目录 (设置 ADSP_LIBRARY_PATH)
    // skelDir: directory of the Skel .so files (set as ADSP_LIBRARY_PATH)
    // arch: HTP 架构版本 (如 V79, 由 QnnDevice_getPlatformInfo 探测得到)
    // arch: HTP architecture version (e.g. V79, probed via QnnDevice_getPlatformInfo)
    // perf: HTP 性能配置 (锁角模式/自动调角模式, 详见 PerfConfig)
    // perf: HTP performance config (locked-corner/adaptive, see PerfConfig)
    // 返回 0 成功
    // Returns 0 on success
    int init(const std::string& libDir, const std::string& skelDir, HtpArch arch,
             const PerfConfig& perf = PerfConfig());

    // 从 context binary (.bin) 加载模型
    // Loads a model from a context binary (.bin)
    // binPath: .bin 文件路径 / path to the .bin file
    // graphName: 输出 graph 名称 (从 binary 元数据读取) / output graph name (read from the binary metadata)
    // 返回 0 成功
    // Returns 0 on success
    int loadFromBinary(const std::string& binPath, std::string& graphName);

    // 执行推理
    // Runs inference
    // inputs: 输入张量数组
    // inputs: input tensor array
    // outputs: 输出张量数组 (输出, 调用方分配 data 内存;
    // outputs: output tensor array (out; the caller allocates `data`;
    //           keepOutputQuantized=true 时 data 可为 null, 结果保留在内部量化缓冲)
    //           with keepOutputQuantized=true, data may be null and results stay in the internal quantized buffer)
    // keepOutputQuantized: true 时不反量化输出, 原始量化数据保留在内部缓冲,
    // keepOutputQuantized: when true, outputs are not dequantized; raw quantized data stays in an internal buffer,
    //                      由 getQuantizedOutputData() 读取、releaseQuantizedOutput() 释放
    //                      read via getQuantizedOutputData() and released via releaseQuantizedOutput()
    // 返回 0 成功
    // Returns 0 on success
    int execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                bool keepOutputQuantized = false);

    // keepOutputQuantized 模式下读取第 i 个输出的原始量化数据 (字节指针)
    // Reads the raw quantized data of output i in keepOutputQuantized mode (byte pointer)
    const uint8_t* getQuantizedOutputData(size_t i) const {
        return (i < m_outQuantBuf.size() && !m_outQuantBuf[i].empty())
                   ? m_outQuantBuf[i].data() : nullptr;
    }

    // 释放第 i 个输出的量化缓冲 (已落盘后调用, 降低写文件阶段内存占用)
    // Releases the quantized buffer of output i (call after persisting, lowers the write-phase memory footprint)
    void releaseQuantizedOutput(size_t i) {
        if (i < m_outQuantBuf.size()) m_outQuantBuf[i].clear();
    }

    // 释放本 runtime 的 context+graph (共享的 device/backend/log 保留, 供其他 runtime 使用)
    // Frees this runtime's context+graph (shared device/backend/log stay alive for other runtimes)
    void freeGraph();

    // 释放本 runtime 的 context+graph, 并释放共享状态引用
    // Frees this runtime's context+graph and releases the shared-state reference
    // (共享 backend/device/log 在最后一个引用释放时销毁)
    // (shared backend/device/log are destroyed when the last reference is released)
    void freeContext();

    // 是否已初始化
    // Whether initialized
    bool isReady() const { return m_ready; }

    // 获取输入输出张量的元信息 (dims, dataType, quantParams)
    // Gets input/output tensor metadata (dims, dataType, quantParams)
    // 从 binary 元数据读取, 在 loadFromBinary 后可用
    // Read from the binary metadata, available after loadFromBinary
    struct TensorInfo {
        std::string name;
        std::vector<uint32_t> dims;
        uint32_t dataType;       // QNN_DATATYPE_*
        float scale;             // 量化 scale / quantization scale
        int32_t offset;          // 量化 offset / quantization offset
        uint32_t quantEncoding;  // 原始量化编码 (QNN_QUANTIZATION_ENCODING_*) / raw quantization encoding
        uint32_t bitwidth;       // BW_SCALE_OFFSET 的 bitwidth (16 或 8) / BW_SCALE_OFFSET bitwidth (16 or 8)
    };
    const std::vector<TensorInfo>& getInputInfos() const { return m_inputInfos; }
    const std::vector<TensorInfo>& getOutputInfos() const { return m_outputInfos; }

    friend class DlcCompiler;

private:
    // 创建共享状态 (dlopen + log/backend/device + HTP 性能配置); 仅 refCount==0 时执行
    // Creates the shared state (dlopen + log/backend/device + HTP performance config); runs only when refCount==0
    int createSharedState(const std::string& libDir, HtpArch arch, const PerfConfig& perf);
    void releaseSharedState();
    void destroySharedState();

    QnnSharedState* m_shared;   // 共享状态 (不拥有, 生命周期由调用方保证) / shared state (not owned; lifetime guaranteed by the caller)
    void* m_contextHandle;
    void* m_graphHandle;
    bool m_ready;

    std::vector<TensorInfo> m_inputInfos;
    std::vector<TensorInfo> m_outputInfos;
    std::string m_graphName;

    // context binary 数据与 tensor id (属于本 runtime)
    // Context binary data and tensor ids (owned by this runtime)
    struct RuntimeData {
        std::vector<uint8_t> binaryBuffer;
        std::vector<uint32_t> inputIds;
        std::vector<uint32_t> outputIds;
    };
    RuntimeData m_data;

    // 持久化的输入/输出缓冲区 (HTP context 在多次 graphExecute 间
    // Persistent input/output buffers (the HTP context binds the clientBuf.data
    // 会绑定 clientBuf.data 的指针地址, 必须保持指针稳定; 4KB 对齐)
    // pointer address across graphExecute calls, so pointers must stay stable; 4KB aligned)
    std::vector<AlignedBuffer> m_inQuantBuf;
    std::vector<AlignedBuffer> m_outQuantBuf;
    std::vector<std::vector<uint32_t>> m_inDims;
    std::vector<std::vector<uint32_t>> m_outDims;
};

} // namespace qnn