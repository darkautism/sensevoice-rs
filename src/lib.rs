pub mod fsmn_vad;
pub mod config;

#[cfg(feature = "stream")]
pub mod silero_vad;
pub mod wavfrontend;

use core::fmt;
use std::sync::Mutex;
#[cfg(feature = "rknpu")]
use std::{fs::File, io::BufReader};

use fsmn_vad::{FSMNVad, VADXOptions};
use hf_hub::api::sync::Api;
use hound::WavReader;
use ndarray::parallel::prelude::*;
#[cfg(feature = "rknpu")]
use ndarray::{Array2, Array3};
use ndarray::{s, ArrayView3, Axis};
#[cfg(feature = "rknpu")]
use ndarray_npy::ReadNpyExt;
use rayon::iter::IntoParallelIterator;
use regex::Regex;
#[cfg(feature = "rknpu")]
use rknn_rs::prelude::{Rknn, RknnInput, RknnTensorFormat, RknnTensorType};
use sentencepiece::SentencePieceProcessor;
use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "stream")]
use silero_vad::{VadConfig, VadProcessor};
use wavfrontend::{WavFrontend, WavFrontendConfig};
use config::SenseVoiceConfig;

#[cfg(feature = "stream")]
use async_stream::stream;
#[cfg(feature = "stream")]
use futures::stream::Stream;
#[cfg(feature = "stream")]
use futures::StreamExt;

/// Represents supported languages for speech recognition.
///
/// This enum defines the languages supported by the `SenseVoiceSmall` model.
#[derive(Debug, Copy, Clone)]
pub enum SenseVoiceLanguage {
    /// English
    En,
    /// Chinese (Mandarin)
    Zh,
    /// Cantonese
    Yue,
    /// Japanese
    Ja,
    /// Korean
    Ko,
}

/// Implementation of methods for `SenseVoiceLanguage`.
impl SenseVoiceLanguage {
    /// Converts a string to a `SenseVoiceLanguage` variant.
    ///
    /// This method parses a string (case-insensitive) and returns the corresponding language variant.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse (e.g., "en", "ZH").
    ///
    /// # Returns
    ///
    /// An `Option<SenseVoiceLanguage>` where `None` indicates an unrecognized language string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "en" => Some(SenseVoiceLanguage::En),
            "zh" => Some(SenseVoiceLanguage::Zh),
            "yue" => Some(SenseVoiceLanguage::Yue),
            "ja" => Some(SenseVoiceLanguage::Ja),
            "ko" => Some(SenseVoiceLanguage::Ko),
            _ => None,
        }
    }
}

/// Represents possible emotions detected in speech.
///
/// This enum defines the emotional states that can be identified in audio segments.
#[derive(Debug, Copy, Clone)]
pub enum SenseVoiceEmo {
    /// Happy emotion
    Happy,
    /// Sad emotion
    Sad,
    /// Angry emotion
    Angry,
    /// Neutral emotion
    Neutral,
    /// Fearful emotion
    Fearful,
    /// Disgusted emotion
    Disgusted,
    /// Surprised emotion
    Surprised,
}

/// Implementation of methods for `SenseVoiceEmo`.
impl SenseVoiceEmo {
    /// Converts a string to a `SenseVoiceEmo` variant.
    ///
    /// This method parses a string (case-insensitive) and returns the corresponding emotion variant.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse (e.g., "HAPPY", "sad").
    ///
    /// # Returns
    ///
    /// An `Option<SenseVoiceEmo>` where `None` indicates an unrecognized emotion string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "HAPPY" => Some(SenseVoiceEmo::Happy),
            "SAD" => Some(SenseVoiceEmo::Sad),
            "ANGRY" => Some(SenseVoiceEmo::Angry),
            "NEUTRAL" => Some(SenseVoiceEmo::Neutral),
            "FEARFUL" => Some(SenseVoiceEmo::Fearful),
            "DISGUSTED" => Some(SenseVoiceEmo::Disgusted),
            "SURPRISED" => Some(SenseVoiceEmo::Surprised),
            _ => None,
        }
    }
}

/// Represents types of audio events detected in speech.
///
/// This enum defines the categories of events that can occur within audio segments.
#[derive(Debug, Copy, Clone)]
pub enum SenseVoiceEvent {
    /// Background music
    Bgm,
    /// Speech content
    Speech,
    /// Applause sound
    Applause,
    /// Laughter sound
    Laughter,
    /// Crying sound
    Cry,
    /// Sneezing sound
    Sneeze,
    /// Breathing sound
    Breath,
    /// Coughing sound
    Cough,
}

/// Implementation of methods for `SenseVoiceEvent`.
impl SenseVoiceEvent {
    /// Converts a string to a `SenseVoiceEvent` variant.
    ///
    /// This method parses a string (case-insensitive) and returns the corresponding event variant.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse (e.g., "BGM", "laughter").
    ///
    /// # Returns
    ///
    /// An `Option<SenseVoiceEvent>` where `None` indicates an unrecognized event string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "BGM" => Some(SenseVoiceEvent::Bgm),
            "SPEECH" => Some(SenseVoiceEvent::Speech),
            "APPLAUSE" => Some(SenseVoiceEvent::Applause),
            "LAUGHTER" => Some(SenseVoiceEvent::Laughter),
            "CRY" => Some(SenseVoiceEvent::Cry),
            "SNEEZE" => Some(SenseVoiceEvent::Sneeze),
            "BREATH" => Some(SenseVoiceEvent::Breath),
            "COUGH" => Some(SenseVoiceEvent::Cough),
            _ => None,
        }
    }
}

/// Represents options for punctuation normalization in transcribed text.
///
/// This enum defines whether punctuation is included or excluded in the output text.
#[derive(Debug, Copy, Clone)]
pub enum SenseVoicePunctuationNormalization {
    /// Include punctuation in the text
    With,
    /// Exclude punctuation from the text
    Woitn,
}

/// Implementation of methods for `SenseVoicePunctuationNormalization`.
impl SenseVoicePunctuationNormalization {
    /// Converts a string to a `SenseVoicePunctuationNormalization` variant.
    ///
    /// This method parses a string (case-insensitive) and returns the corresponding normalization variant.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse (e.g., "with", "WOITN").
    ///
    /// # Returns
    ///
    /// An `Option<SenseVoicePunctuationNormalization>` where `None` indicates an unrecognized normalization string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "with" => Some(SenseVoicePunctuationNormalization::With),
            "woitn" => Some(SenseVoicePunctuationNormalization::Woitn),
            _ => None,
        }
    }
}

/// Represents a segment of audio with its transcribed text and associated metadata.
///
/// This structure holds the transcription result of an audio segment, including timing, language, emotion, event, and normalization details.
#[derive(Debug)]
pub struct VoiceText {
    /// The language of the transcribed text.
    pub language: SenseVoiceLanguage,
    /// The detected emotion in the audio segment.
    pub emotion: SenseVoiceEmo,
    /// The type of audio event in the segment.
    pub event: SenseVoiceEvent,
    /// Indicates whether punctuation is included in the transcribed text.
    pub punctuation_normalization: SenseVoicePunctuationNormalization,
    /// The transcribed text of the audio segment.
    pub content: String,
}

/// Parses a string line into a `VoiceText` instance based on a specific format.
///
/// The expected format is: `<|language|><|emotion|><|event|><|punctuation|><content>`
///
/// # Arguments
///
/// * `line` - The string to parse (e.g., "<|zh|><|HAPPY|><|BGM|><|woitn|>Hello").
/// * `start_ms` - Start time of the segment in milliseconds.
/// * `end_ms` - End time of the segment in milliseconds.
///
/// # Returns
///
/// An `Option<VoiceText>` where `None` indicates parsing failure due to invalid format or unrecognized tags.
fn parse_line(line: &str) -> Option<VoiceText> {
    let re = Regex::new(r"^<\|(.*?)\|><\|(.*?)\|><\|(.*?)\|><\|(.*?)\|>(.*)$").unwrap();
    if let Some(caps) = re.captures(line) {
        let lang_str = &caps[1];
        let emo_str = &caps[2];
        let event_str = &caps[3];
        let punct_str = &caps[4];
        let content = &caps[5];

        let language = SenseVoiceLanguage::from_str(lang_str)?;
        let emotion = SenseVoiceEmo::from_str(emo_str)?;
        let event = SenseVoiceEvent::from_str(event_str)?;
        let punctuation_normalization = SenseVoicePunctuationNormalization::from_str(punct_str)?;

        Some(VoiceText {
            language,
            emotion,
            event,
            punctuation_normalization,
            content: content.to_string(),
        })
    } else {
        None
    }
}

/// Represents an error specific to the `SenseVoiceSmall` system.
///
/// This structure encapsulates error messages related to initialization, inference, or resource management.
#[derive(Debug)]
struct SenseVoiceSmallError {
    /// The error message describing the issue.
    message: String,
}

/// Implements `Display` trait for `SenseVoiceSmallError` to format error messages.
impl fmt::Display for SenseVoiceSmallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SenseVoiceSmallError: {}", self.message)
    }
}

/// Implements `Error` trait for `SenseVoiceSmallError` to integrate with Rust's error handling system.
impl std::error::Error for SenseVoiceSmallError {}

/// Implementation of methods for `SenseVoiceSmallError`.
impl SenseVoiceSmallError {
    /// Creates a new `SenseVoiceSmallError` instance with the given message.
    ///
    /// # Arguments
    ///
    /// * `message` - The error message to encapsulate.
    ///
    /// # Returns
    ///
    /// A new `SenseVoiceSmallError` instance.
    pub fn new(message: &str) -> Self {
        SenseVoiceSmallError {
            message: message.to_owned(),
        }
    }
}

/// Represents the core structure for the SenseVoiceSmall speech recognition system.
///
/// This structure manages components such as voice activity detection (VAD), automatic speech recognition (ASR),
/// and inference (RKNN or ONNX) for processing audio data.
#[derive(Debug)]
pub struct SenseVoiceSmall {
    vad_frontend: WavFrontend,
    asr_frontend: WavFrontend,
    #[cfg(feature = "rknpu")]
    n_seq: usize,
    spp: SentencePieceProcessor,

    // RKNN specific fields
    #[cfg(feature = "rknpu")]
    rknn: Option<Rknn>,
    #[cfg(feature = "rknpu")]
    embedding: Option<ndarray::Array2<f32>>,

    // ONNX specific fields
    // Session is thread-safe in ORT 2.0? Usually yes. But we might need Mutex if run needs mut?
    // According to docs, Session::run takes &self, not &mut self.
    // Wait, the error said "cannot borrow `*session` as mutable, as it is behind a `&` reference".
    // This implies `session.run` takes `&mut self`?
    // Let's check ORT 2.0 docs or source.
    // If it takes &mut self, we need Mutex or RefCell.
    // But typically inference sessions are shared.
    // If it needs mutable, maybe I should wrap it in Mutex.
    ort_session: Option<Mutex<ort::session::Session>>,

    fsmn: Mutex<Option<FSMNVad>>, // Mutex wrapping Option to allow skipping VAD? No, VAD struct is needed but maybe unused.
    // Wait, fsmn_vad creates an ORT session, so it's heavy. We should make it optional if VAD is optional.
    // However, existing code uses Mutex<FSMNVad>. Let's stick to Mutex<FSMNVad> for now and assume it's always initialized if VAD is needed.
    // If VAD is optional for ONNX, we might skip initializing it.
    // Let's use Option<FSMNVad> inside Mutex to allow it to be None.

    #[cfg(feature = "stream")]
    silero_vad: VadProcessor,

    use_rknn: bool,
}

/// Implementation of methods for `SenseVoiceSmall`.
impl SenseVoiceSmall {
    /// Initializes a new `SenseVoiceSmall` instance.
    ///
    /// If the `rknpu` feature is enabled, it initializes the RKNN backend using the default RKNN model.
    /// Otherwise, it initializes the ONNX backend using the default ONNX model.
    ///
    /// # Arguments
    ///
    /// * `vadconfig` - Configuration for VAD.
    ///
    /// # Errors
    ///
    /// Returns an error if model files cannot be loaded.
    pub fn init(
        vadconfig: VADXOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(feature = "rknpu")]
        {
            println!("RKNPU feature enabled, initializing RKNN backend...");
            // RKNN Path
            let model_path = "happyme531/SenseVoiceSmall-RKNN2";

            let api = Api::new().unwrap();
            let repo = api.model(model_path.to_string());

            let fsmn_path = repo.get("fsmnvad-offline.onnx")?;
            let embedding_path = repo.get("embedding.npy")?;
            let rknn_path = repo.get("sense-voice-encoder.rknn")?;
            let sentence_path = repo.get("chn_jpn_yue_eng_ko_spectok.bpe.model")?;
            let fsmn_am_path = repo.get("fsmn-am.mvn")?;
            let am_path = repo.get("am.mvn")?;

            let config = SenseVoiceConfig {
                model_path: rknn_path,
                tokenizer_path: sentence_path,
                vad_model_path: Some(fsmn_path),
                vad_cmvn_path: Some(fsmn_am_path),
                cmvn_path: Some(am_path),
            };

            let fsmn = Mutex::new(Some(FSMNVad::new(config.vad_model_path.as_ref().unwrap(), vadconfig)?));

            let embedding_file = File::open(embedding_path)?;
            let embedding_reader = BufReader::new(embedding_file);
            let embedding: Array2<f32> = Array2::read_npy(embedding_reader)?;
            assert_eq!(embedding.shape()[1], 560, "Embedding dimension must be 560");

            let rknn = Rknn::rknn_init(config.model_path)?;
            let spp = SentencePieceProcessor::open(config.tokenizer_path)?;

            let n_seq = 171;

            let vad_frontend = WavFrontend::new(WavFrontendConfig {
                lfr_m: 5,
                cmvn_file: Some(config.vad_cmvn_path.as_ref().unwrap().to_str().unwrap().to_owned()),
                ..Default::default()
            })?;

            let asr_frontend = WavFrontend::new(WavFrontendConfig {
                lfr_m: 7,
                cmvn_file: Some(config.cmvn_path.as_ref().unwrap().to_str().unwrap().to_owned()),
                ..Default::default()
            })?;

            #[cfg(feature = "stream")]
            let silero_vad = VadProcessor::new(VadConfig::default())?;

            Ok(SenseVoiceSmall {
                vad_frontend,
                asr_frontend,
                n_seq,
                spp,
                rknn: Some(rknn),
                embedding: Some(embedding),
                ort_session: None,
                fsmn,
                #[cfg(feature = "stream")]
                silero_vad,
                use_rknn: true,
            })
        }
        #[cfg(not(feature = "rknpu"))]
        {
            println!("RKNPU feature disabled, initializing ONNX backend...");
            // ONNX Path
            // Use haixuantao/SenseVoiceSmall-onnx
            let model_repo = "haixuantao/SenseVoiceSmall-onnx";
            let api = Api::new().unwrap();
            let repo = api.model(model_repo.to_string());

            let model_path = repo.get("model_quant.onnx")?;
            let tokenizer_path = repo.get("chn_jpn_yue_eng_ko_spectok.bpe.model")?;
            let am_path = repo.get("am.mvn")?;

            let vad_repo_name = "happyme531/SenseVoiceSmall-RKNN2";
            let vad_repo = api.model(vad_repo_name.to_string());
            let vad_model_path = vad_repo.get("fsmnvad-offline.onnx").ok();
            let vad_cmvn_path = vad_repo.get("fsmn-am.mvn").ok();

            let config = SenseVoiceConfig {
                model_path,
                tokenizer_path,
                vad_model_path,
                vad_cmvn_path,
                cmvn_path: Some(am_path),
            };

            Self::init_with_config(config, vadconfig)
        }
    }

    /// Initializes a new `SenseVoiceSmall` instance with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration containing file paths.
    /// * `vadconfig` - Configuration for VAD.
    ///
    /// # Errors
    ///
    /// Returns an error if model files cannot be loaded.
    pub fn init_with_config(
        config: SenseVoiceConfig,
        vadconfig: VADXOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(feature = "rknpu")]
        {
             // If rknpu feature is enabled, we check if it is an RKNN model
             let is_rknn_model = config.model_path.extension().map_or(false, |ext| ext == "rknn");
             if is_rknn_model {
                 return Err("Manual loading of RKNN models via init_with_config is not fully supported yet (missing embedding path). Use init() for default RKNN model.".into());
             }
             // If not RKNN model (e.g. ONNX), we could technically support it even with rknpu feature enabled,
             // assuming dependencies allow it. However, `rknn` field in struct is Option, so we can set it to None.
             // But existing struct has `#[cfg(feature = "rknpu")]` fields.
             // We can proceed to ONNX loading below if we decide so, OR restrict it.
             // Given "assume it is rknpu", maybe we should strictly expect RKNN?
             // But the user might want to run ONNX on a device that also supports RKNN.
             // Let's allow ONNX loading here if the model is not .rknn
        }

        // ONNX Loading
        let spp = SentencePieceProcessor::open(&config.tokenizer_path)?;

        // Load ONNX model
        let ort_session = Mutex::new(ort::session::Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&config.model_path)?);

        // VAD Setup
        let fsmn = if let Some(vad_path) = config.vad_model_path {
             Mutex::new(Some(FSMNVad::new(vad_path, vadconfig)?))
        } else {
             Mutex::new(None)
        };

        let vad_frontend = WavFrontend::new(WavFrontendConfig {
            lfr_m: 5,
            cmvn_file: config.vad_cmvn_path.map(|p| p.to_string_lossy().to_string()),
            ..Default::default()
        })?;

        let asr_frontend = WavFrontend::new(WavFrontendConfig {
            lfr_m: 7,
            cmvn_file: config.cmvn_path.map(|p| p.to_string_lossy().to_string()),
            ..Default::default()
        })?;

        #[cfg(feature = "stream")]
        let silero_vad = VadProcessor::new(VadConfig::default())?;

        // n_seq is 171 for RKNN, but for ONNX it might differ or be dynamic.
        // However, standard SenseVoice ONNX might accept dynamic shapes.
        // We set n_seq to 0 or ignore it for ONNX.
        #[cfg(feature = "rknpu")]
        let n_seq = 0;

        Ok(SenseVoiceSmall {
            vad_frontend,
            asr_frontend,
            #[cfg(feature = "rknpu")]
            n_seq,
            spp,
            #[cfg(feature = "rknpu")]
            rknn: None,
            #[cfg(feature = "rknpu")]
            embedding: None,
            ort_session: Some(ort_session),
            fsmn,
            #[cfg(feature = "stream")]
            silero_vad,
            use_rknn: false,
        })
    }

    /// Performs speech recognition on a vector of audio samples.
    pub fn infer_vec(
        &self,
        content: Vec<i16>,
        sample_rate: u32,
    ) -> Result<Vec<VoiceText>, Box<dyn std::error::Error>> {
        // 提取特徵
        let audio_feats = self.vad_frontend.extract_features(&content)?;

        // 進行 VAD 推理 (如果啟用)
        let segments = {
            let mut fsmn_lock = self.fsmn.lock().unwrap();
            if let Some(fsmn) = fsmn_lock.as_mut() {
                fsmn.infer_vad(audio_feats, &content, true)?
            } else {
                // 如果沒有 VAD，則將整個音頻視為一個片段
                // Treat the whole audio as one segment if VAD is disabled
                vec![(0, (content.len() as f32 / sample_rate as f32 * 1000.0) as i32)]
            }
        };

        // 處理語音片段
        let mut ret = Vec::new();
        for (start_ms, end_ms) in segments {
            let start_sample = (start_ms as f32 / 1000.0 * sample_rate as f32) as usize;
            let end_sample = (end_ms as f32 / 1000.0 * sample_rate as f32) as usize;
            // Bound checking
            let start_sample = start_sample.min(content.len());
            let end_sample = end_sample.min(content.len());

            if start_sample >= end_sample { continue; }

            let segment = &content[start_sample..end_sample];
            let vt = self.recognition(segment)?;
            ret.push(vt);
        }
        Ok(ret)
    }

    pub fn recognition(&self, segment: &[i16]) -> Result<VoiceText, Box<dyn std::error::Error>> {
        // 提取特徵
        let audio_feats = self.asr_frontend.extract_features(segment)?;

        if self.use_rknn {
            #[cfg(feature = "rknpu")]
            {
                if let Some(rknn) = &self.rknn {
                    // 準備 RKNN 輸入
                    self.prepare_rknn_input_advanced(&audio_feats, 0, false)?;
                    rknn.run()?;
                    let mut asr_output = rknn.outputs_get_raw::<f32>()?;
                    let asr_text = self.decode_asr_output(&asr_output.data)?;
                    rknn.outputs_release(&mut asr_output)?;
                    return match parse_line(&asr_text) {
                        Some(vt) => Ok(vt),
                        None => Err(format!("Parse line failed, text is:{}, If u still get empty text, please check your vad config. This model only can infer 9 secs voice.", asr_text).into()),
                    };
                }
            }
            return Err("RKNN is enabled but model is not initialized".into());
        } else {
            // ONNX Inference
            if let Some(session_mutex) = &self.ort_session {
                let mut session = session_mutex.lock().unwrap();
                // Prepare inputs for ONNX
                // Inputs: speech (1, T, 560), language (1), textnorm (1)
                // Assuming dynamic shape for T

                let _seq_len = audio_feats.shape()[0];
                let speech = audio_feats.view().insert_axis(Axis(0)); // [1, T, 560]

                // Language: 0 (auto/zn), TextNorm: 15 (woitn/none) or 14 (with itn)
                // Existing code: "language=0 (auto), use_itn=false" -> text_norm_idx = 15
                let language_val = 0i32;
                let textnorm_val = 15i32; // 15 means 'woitn' (without ITN/punctuation?) based on prepare_rknn_input_advanced logic

                let language = ndarray::arr1(&[language_val]).insert_axis(Axis(0));
                let textnorm = ndarray::arr1(&[textnorm_val]).insert_axis(Axis(0));

                let speech_tensor = ort::value::Tensor::from_array(speech.to_owned())?;
                let language_tensor = ort::value::Tensor::from_array(language.to_owned())?;
                let textnorm_tensor = ort::value::Tensor::from_array(textnorm.to_owned())?;

                let inputs = ort::inputs![
                    "speech" => speech_tensor,
                    "language" => language_tensor,
                    "textnorm" => textnorm_tensor,
                ];

                let outputs = session.run(inputs)?;

                // Output handling
                // Usually output name is "logits" or similar.
                // SenseVoice ONNX output is usually [1, V, T] or [1, T, V]?
                // Let's assume [1, V, T] similar to RKNN or check dimensions.

                // Try to get the first output
                let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()?;

                // The decoding logic `decode_asr_output` expects `[n_vocab, n_seq]` (flattened or reshaped)
                // The RKNN code does: `ArrayView3::from_shape((1, n_vocab, self.n_seq), output)`
                // So RKNN output is [1, 25055, 171].
                // We need to know the shape of ONNX output.
                // Typically: [Batch, Time, Vocab] or [Batch, Vocab, Time].
                // If it is [Batch, Time, Vocab], we need to permute or adjust decoding.
                // SenseVoice (FunASR) usually outputs logits.

                // Debug/Verify shape if possible.
                // Assuming [1, Time, Vocab] for standard Transformers, but SenseVoice might be different.
                // If shape is [1, Time, Vocab], we need to adjust decode_asr_output.

                // Let's implement a generic decoder for ONNX output.
                // Assuming output_shape dereferences to &[i64] or is compatible
                let asr_text = self.decode_onnx_output(output_data, &output_shape)?;

                return match parse_line(&asr_text) {
                    Some(vt) => Ok(vt),
                    None => Err(format!("Parse line failed, text is:{}", asr_text).into()),
                };
            }
            return Err("ONNX session is not initialized".into());
        }
    }

    #[cfg(feature = "stream")]
    pub fn infer_stream<S>(
        &mut self,
        input_stream: S,
    ) -> impl Stream<Item = Result<VoiceText, Box<dyn std::error::Error>>>
    where
        S: Stream<Item = Vec<i16>> + Unpin,
    {
        stream! {
        let mut stream = input_stream;
        while let Some(chunk) = stream.next().await {
        if let Some(segment) = self.silero_vad.process_chunk(&chunk) {
        yield self.recognition(&segment);
        }
        }
        if let Some(segment) = self.silero_vad.finish() {
        yield self.recognition(&segment);
        }
        }
    }

    /// Performs speech recognition on an audio file.
    pub fn infer_file<P: AsRef<std::path::Path>>(
        &self,
        wav_path: P,
    ) -> Result<Vec<VoiceText>, Box<dyn std::error::Error>> {
        let mut wav_reader = WavReader::open(wav_path)?;
        match wav_reader.spec().sample_rate {
            8000 => (),
            16000 => (),
            _ => {
                return Err(Box::new(SenseVoiceSmallError::new(
                    "Unsupported sample rate. Expect 8 kHz or 16 kHz.",
                )))
            }
        };
        if wav_reader.spec().sample_format != hound::SampleFormat::Int {
            return Err(Box::new(SenseVoiceSmallError::new(
                "Unsupported sample format. Expect Int.",
            )));
        }

        let content = wav_reader
            .samples()
            .filter_map(|x| x.ok())
            .collect::<Vec<i16>>();
        if content.is_empty() {
            return Err(Box::new(SenseVoiceSmallError::new(
                "content is empty, check your audio file",
            )));
        }

        self.infer_vec(content, wav_reader.spec().sample_rate)
    }

    /// Decodes RKNN output into a transcribed text string.
    #[cfg(feature = "rknpu")]
    fn decode_asr_output(&self, output: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        // 解析為 [1, n_vocab, n_seq]
        let n_vocab = self.spp.len();
        // RKNN n_seq is fixed 171
        let output_array = ArrayView3::from_shape((1, n_vocab, self.n_seq), output)?;

        // 在 n_vocab 維度（Axis(1)）上取 argmax
        let token_ids: Vec<i32> = output_array
            .axis_iter(Axis(2)) // 沿著 n_seq=171 維度迭代
            .into_par_iter()
            .map(|slice| {
                slice
                    .iter()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |(idx, max_val), (i, &val)| {
                        if val > max_val {
                            (i, val)
                        } else {
                            (idx, max_val)
                        }
                    })
                    .0 as i32 // 提取最大值的索引
            })
            .collect();

        self.ids_to_text(token_ids)
    }

    /// Helper to convert token IDs to text
    fn ids_to_text(&self, token_ids: Vec<i32>) -> Result<String, Box<dyn std::error::Error>> {
         // 移除連續重複的 token 和 blank_id=0
        let mut unique_ids = Vec::new();
        let mut prev_id = None;
        for &id in token_ids.iter() {
            if Some(id) != prev_id && id != 0 {
                unique_ids.push(id as u32);
                prev_id = Some(id);
            } else if Some(id) != prev_id {
                prev_id = Some(id);
            }
        }

        // 解碼為文本
        let decoded_text = self.spp.decode_piece_ids(&unique_ids)?;
        Ok(decoded_text)
    }

    /// Decodes ONNX output.
    fn decode_onnx_output(&self, output: &[f32], shape: &[i64]) -> Result<String, Box<dyn std::error::Error>> {
        // Shape is likely [1, T, Vocab] or [1, Vocab, T].
        // If T is dynamic, we use it.
        // Assuming [1, T, Vocab] which is common for CTC/Frame-level outputs from generic inference.
        // But RKNN was [1, Vocab, T].
        // Let's assume standard SenseVoice ONNX matches pytorch output: [Batch, Time, Vocab].

        let batch_size = shape[0] as usize;
        if batch_size != 1 {
            return Err("Batch size must be 1".into());
        }

        // Guessing layout based on dimensions. Vocab size is ~25055.
        // If dim 1 is ~25000, then it is [B, V, T].
        // If dim 2 is ~25000, then it is [B, T, V].

        let n_vocab = self.spp.len(); // ~25055
        let dim1 = shape[1] as usize;
        let dim2 = shape[2] as usize;

        let output_array = ArrayView3::from_shape((shape[0] as usize, shape[1] as usize, shape[2] as usize), output)?;
        let mut token_ids = Vec::new();

        if dim1 == n_vocab {
             // [B, V, T] - iterate over T (dim 2)
             for t in 0..dim2 {
                 // slice at time t: [1, V]
                 let col = output_array.slice(s![0, .., t]);
                 // argmax over V
                 let (best_idx, _) = col.iter().enumerate().fold((0, f32::NEG_INFINITY), |(acc_idx, acc_val), (i, &val)| {
                     if val > acc_val { (i, val) } else { (acc_idx, acc_val) }
                 });
                 token_ids.push(best_idx as i32);
             }
        } else if dim2 == n_vocab {
             // [B, T, V] - iterate over T (dim 1)
             for t in 0..dim1 {
                 let row = output_array.slice(s![0, t, ..]);
                 let (best_idx, _) = row.iter().enumerate().fold((0, f32::NEG_INFINITY), |(acc_idx, acc_val), (i, &val)| {
                     if val > acc_val { (i, val) } else { (acc_idx, acc_val) }
                 });
                 token_ids.push(best_idx as i32);
             }
        } else {
             return Err(format!("Unexpected output shape: {:?}, expected one dimension to be vocab size {}", shape, n_vocab).into());
        }

        self.ids_to_text(token_ids)
    }

    /// Destroys the `SenseVoiceSmall` instance, releasing associated resources.
    ///
    /// This method ensures that the RKNN model resources are properly cleaned up.
    ///
    /// # Errors
    ///
    /// Returns an error if the RKNN model destruction fails.
    ///
    /// # Example
    ///
    /// ```
    /// use sensevoice_rs::SenseVoiceSmall;
    ///
    /// let svs = SenseVoiceSmall::init().expect("Failed to initialize");
    /// svs.destroy().expect("Failed to destroy SenseVoiceSmall");
    /// ```
    pub fn destroy(&self) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "rknpu")]
        if let Some(rknn) = &self.rknn {
            rknn.destroy()?;
        }
        Ok(())
    }

    /// Prepares input data for RKNN inference with advanced configuration.
    ///
    /// This method constructs the input tensor by combining language embeddings, event/emotion embeddings,
    /// text normalization embeddings, and scaled audio features, then pads or truncates it to match the expected shape.
    ///
    /// # Arguments
    ///
    /// * `feats` - A 2D array of audio features.
    /// * `language` - Index of the language embedding to use (0 for auto).
    /// * `use_itn` - Whether to use inverse text normalization (true) or not (false).
    ///
    /// # Errors
    ///
    /// Returns an error if tensor concatenation, padding, or RKNN input setting fails.
    #[cfg(feature = "rknpu")]
    fn prepare_rknn_input_advanced(
        &self,
        feats: &Array2<f32>,
        language: usize,
        use_itn: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 提取嵌入向量
        let embedding = self.embedding.as_ref().ok_or("Embedding not loaded")?;

        let language_query = embedding.slice(s![language, ..]).insert_axis(Axis(0));
        let text_norm_idx = if use_itn { 14 } else { 15 };
        let text_norm_query = embedding
            .slice(s![text_norm_idx, ..])
            .insert_axis(Axis(0));
        let event_emo_query = embedding.slice(s![1..=2, ..]).to_owned();

        // 縮放語音特徵
        let speech = feats.mapv(|x| x * 0.5);

        // 沿著幀軸串接
        let input_content = ndarray::concatenate(
            Axis(0),
            &[
                language_query.view(),
                event_emo_query.view(),
                text_norm_query.view(),
                speech.view(),
            ],
        )?;

        // 填充或截斷至 [n_seq , 560]
        let total_frames = input_content.shape()[0];
        let padded_input = if total_frames < self.n_seq {
            let mut padded = Array2::zeros((self.n_seq, 560));
            padded
                .slice_mut(s![..total_frames, ..])
                .assign(&input_content);
            padded
        } else {
            input_content.slice(s![..self.n_seq, ..]).to_owned()
        };
        // Add batch dimension
        let input_3d: Array3<f32> = padded_input.insert_axis(Axis(0)); // [1, n_seq , 560]

        // Ensure contiguous memory and flatten to 1D
        let contiguous_input = input_3d.as_standard_layout(); // Row-major contiguous
        let flattened_input: Vec<f32> = contiguous_input
            .into_shape_with_order(1 * self.n_seq * 560)? // Flatten to [95760]
            .to_vec(); // Owned Vec<f32>

        if let Some(rknn) = &self.rknn {
            rknn.input_set(&mut RknnInput {
                index: 0,             // 根據您的輸入索引設定
                buf: flattened_input, /* 您的數據 */
                pass_through: false,  // 通常設為 false，除非模型需要特殊處理
                type_: RknnTensorType::Float32,
                fmt: RknnTensorFormat::NCHW,
            })?;
        }
        Ok(())
    }
}
