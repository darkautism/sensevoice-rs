use std::path::PathBuf;

/// 設定檔結構，用於手動載入模型
/// Configuration struct for manual model loading
#[derive(Debug, Clone)]
pub struct SenseVoiceConfig {
    /// 主要模型路徑 (ONNX 或 RKNN)
    /// Path to the main model (ONNX or RKNN)
    pub model_path: PathBuf,

    /// Tokenizer 模型路徑 (sentencepiece)
    /// Path to the tokenizer model
    pub tokenizer_path: PathBuf,

    /// ASR CMVN 檔案路徑
    /// Path to the ASR CMVN file
    pub cmvn_path: Option<PathBuf>,
}
