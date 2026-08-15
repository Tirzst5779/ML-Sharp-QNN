// qnn_dlc_compiler.h — DLC 设备端编译器
// qnn_dlc_compiler.h — on-device DLC compiler
// 从 DLC 文件编译生成 context binary (.bin)
// Compiles a DLC file into a context binary (.bin)
#pragma once

#include <atomic>
#include <string>
#include "qnn_runtime.h"
#include "System/QnnSystemInterface.h"

namespace qnn {

// DLC 编译器: 将 DLC 在设备端编译为 context binary
// DLC compiler: compiles a DLC into a context binary on-device
// 流程: systemDlcCreateFromFile → systemDlcComposeGraphs(单次, 返回 graph 信息)
// Flow: systemDlcCreateFromFile -> systemDlcComposeGraphs (once, returns graph info)
//       → 每 graph: graphRetrieve → graphSetConfig(HTP 优化) → graphFinalize
//       -> per graph: graphRetrieve -> graphSetConfig (HTP optimization) -> graphFinalize
//       → contextGetBinary
//       -> contextGetBinary
// (官方 SampleApp 同款流程: compose 一次 + HTP config 经 graphSetConfig 下发)
// (same flow as the official SampleApp: compose once, HTP config pushed via graphSetConfig)
class DlcCompiler {
public:
    DlcCompiler();
    ~DlcCompiler();

    // 编译 DLC 为 context binary
    // Compiles a DLC into a context binary
    // runtime: 已初始化的 HtpRuntime (提供 backend/device/context)
    // runtime: an initialized HtpRuntime (provides backend/device/context)
    // dlcPath: DLC 文件路径
    // dlcPath: path to the DLC file
    // outBinPath: 编译后 .bin 保存路径
    // outBinPath: path to save the compiled .bin
    // progressCb: 编译进度回调 (可选)
    // progressCb: compilation progress callback (optional)
    // 返回 0 成功; -100 表示被取消 (已在内部清理部分资源)
    // Returns 0 on success; -100 means cancelled (partial resources already cleaned up internally)
    int compile(HtpRuntime& runtime, const std::string& dlcPath,
                const std::string& outBinPath, ProgressCallback progressCb = nullptr);

    // 请求取消当前编译 (线程安全, 编译在步骤间隙检查标志后中止)
    // Requests cancellation of the current compile (thread-safe; the compile checks the flag between steps and aborts)
    // 注意: QNN graphFinalize 为阻塞调用不可被打断, 取消在该调用返回后立即生效
    // Note: QNN graphFinalize is blocking and cannot be interrupted; cancellation takes effect right after it returns
    void requestCancel() { m_cancelRequested.store(true, std::memory_order_relaxed); }

    // 是否已被取消
    // Whether a cancellation was requested
    bool isCancelRequested() const { return m_cancelRequested.load(std::memory_order_relaxed); }

private:
    void* m_dlcHandle;
    // 保存最近一次 compile 使用的 QNN system interface 副本,
    // Keeps a copy of the QNN system interface used by the last compile,
    // 供析构函数在异常残留 m_dlcHandle 时调用 systemDlcFree (不依赖 runtime 存活)
    // so the destructor can call systemDlcFree on a leftover m_dlcHandle (independent of runtime lifetime)
    QnnSystemInterface_t m_sysInterface;
    bool m_sysInterfaceValid;
    // 编译取消标志 (由 JNI cancelCompile 设置)
    // Cancellation flag (set by the JNI cancelCompile entry point)
    std::atomic<bool> m_cancelRequested;
};

} // namespace qnn