use std::collections::{HashMap, VecDeque};

use candle::{DType, Device, Tensor};
use hf_hub::api::sync::Api;

use crate::wavfrontend::{WavFrontend, WavFrontendConfig};

pub const CHUNK_SIZE: usize = 512;
const FSMN_VAD_REPO: &str = "funasr/fsmn-vad-onnx";

struct FsmnVadModel {
    model: candle_onnx::onnx::ModelProto,
    frontend: WavFrontend,
    cache0: Tensor,
    cache1: Tensor,
    cache2: Tensor,
    cache3: Tensor,
}

impl std::fmt::Debug for FsmnVadModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsmnVadModel").finish()
    }
}

impl FsmnVadModel {
    fn new(config: &VadConfig) -> Result<Self, Box<dyn std::error::Error>> {
        if config.sample_rate != 16000 {
            return Err(std::io::Error::other("FSMN-VAD currently supports 16k sample rate").into());
        }

        let api = Api::new()?;
        let repo = api.model(FSMN_VAD_REPO.to_owned());
        let model_path = repo.get("model.onnx")?;
        let cmvn_path = repo.get("vad.mvn")?;

        let frontend = WavFrontend::new(WavFrontendConfig {
            sample_rate: config.sample_rate as i32,
            lfr_m: 5,
            lfr_n: 1,
            cmvn_file: Some(cmvn_path.to_string_lossy().to_string()),
            ..Default::default()
        })?;

        let model = candle_onnx::read_file(model_path)?;
        let cache = Tensor::zeros((1usize, 128usize, 19usize, 1usize), DType::F32, &Device::Cpu)?;

        Ok(Self {
            model,
            frontend,
            cache0: cache.clone(),
            cache1: cache.clone(),
            cache2: cache.clone(),
            cache3: cache,
        })
    }

    fn predict_probability(
        &mut self,
        chunk: &[i16; CHUNK_SIZE],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let feats = self.frontend.extract_features(chunk)?;
        let t = feats.shape()[0];
        let d = feats.shape()[1];
        if d != 400 {
            return Err(std::io::Error::other(format!(
                "FSMN-VAD expects feature dim 400, got {d}"
            ))
            .into());
        }

        let speech = Tensor::from_vec(
            feats.iter().copied().collect::<Vec<f32>>(),
            (1usize, t, d),
            &Device::Cpu,
        )?;

        let mut inputs = HashMap::new();
        inputs.insert("speech".to_string(), speech);
        inputs.insert("in_cache0".to_string(), self.cache0.clone());
        inputs.insert("in_cache1".to_string(), self.cache1.clone());
        inputs.insert("in_cache2".to_string(), self.cache2.clone());
        inputs.insert("in_cache3".to_string(), self.cache3.clone());

        let mut outputs = candle_onnx::simple_eval(&self.model, inputs)?;
        self.cache0 = outputs
            .remove("out_cache0")
            .ok_or_else(|| std::io::Error::other("Missing out_cache0 from FSMN-VAD output"))?;
        self.cache1 = outputs
            .remove("out_cache1")
            .ok_or_else(|| std::io::Error::other("Missing out_cache1 from FSMN-VAD output"))?;
        self.cache2 = outputs
            .remove("out_cache2")
            .ok_or_else(|| std::io::Error::other("Missing out_cache2 from FSMN-VAD output"))?;
        self.cache3 = outputs
            .remove("out_cache3")
            .ok_or_else(|| std::io::Error::other("Missing out_cache3 from FSMN-VAD output"))?;

        let logits = outputs
            .remove("logits")
            .ok_or_else(|| std::io::Error::other("Missing logits from FSMN-VAD output"))?;

        let dims = logits.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] == 0 {
            return Err(std::io::Error::other(format!(
                "Unexpected FSMN-VAD logits shape: {:?}",
                dims
            ))
            .into());
        }

        let frames = dims[1];
        let classes = dims[2];
        if frames == 0 {
            return Ok(0.0);
        }

        let data = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut speech_prob_sum = 0f32;
        for frame_idx in 0..frames {
            // class-0 is silence in FunASR FSMN-VAD config (sil_pdf_ids: [0])
            let silence_prob = data[frame_idx * classes];
            speech_prob_sum += (1.0 - silence_prob).clamp(0.0, 1.0);
        }

        Ok(speech_prob_sum / frames as f32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    pub sample_rate: u32,            // 採樣率，例如 16000 Hz
    pub speech_threshold: f32,       // 語音概率閾值，例如 0.5
    pub silence_duration_ms: u32,    // 靜音持續時間（毫秒），例如 500 ms
    pub max_speech_duration_ms: u32, // 最大語音段長（毫秒），例如 10000 ms
    pub rollback_duration_ms: u32,   // 剪斷後回退時間（毫秒），例如 200 ms
    pub min_speech_duration_ms: u32, // 最小語音段長（毫秒），小於此長度視為噪音，例如 250 ms
    pub notify_silence_after_ms: Option<u32>, // 如果處於等待狀態超過此時間，發出靜音通知
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            speech_threshold: 0.5,
            silence_duration_ms: 500,     // 500 ms 靜音算結束
            max_speech_duration_ms: 9000, // 9 秒最大語音段
            rollback_duration_ms: 200,    // 回退 200 ms
            min_speech_duration_ms: 250,  // 最小 250 ms
            notify_silence_after_ms: None,
        }
    }
}

#[derive(Debug)]
enum VadState {
    Waiting,
    Recording,
}

/// Enum to distinguish between a speech segment and a silence notification
#[derive(Debug)]
pub enum VadOutput {
    Segment(Vec<i16>),
    SilenceNotification,
}

#[derive(Debug)]
pub struct VadProcessor {
    vad: FsmnVadModel,
    config: VadConfig,
    state: VadState,
    current_segment: Vec<i16>,
    history_buffer: VecDeque<i16>, // 用於保留語音開始前的上下文
    silence_chunks: u32,           // 連續靜音塊數 (Recording 狀態下)
    speech_chunks: u32,            // 當前語音段的塊數
    waiting_dropped_chunks: u32,   // Waiting 狀態下已丟棄的塊數
    notified_silence: bool,        // 是否已經發出過靜音通知 (One-shot)
}

impl VadProcessor {
    pub fn new(config: VadConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let vad = FsmnVadModel::new(&config)?;
        Ok(Self {
            vad,
            config,
            state: VadState::Waiting,
            current_segment: Vec::new(),
            history_buffer: VecDeque::new(),
            silence_chunks: 0,
            speech_chunks: 0,
            waiting_dropped_chunks: 0,
            notified_silence: false,
        })
    }

    /// 更新通知靜音的設定
    pub fn set_notify_silence_after_ms(&mut self, ms: Option<u32>) {
        self.config.notify_silence_after_ms = ms;
        if ms.is_none() {
            self.notified_silence = false;
        }
    }

    pub fn process_chunk(&mut self, chunk: &[i16; CHUNK_SIZE]) -> Option<VadOutput> {
        let chunk_duration_ms = (CHUNK_SIZE as f32 / self.config.sample_rate as f32) * 1000.0;
        let probability = self
            .vad
            .predict_probability(chunk)
            .expect("FSMN VAD inference failed");

        match self.state {
            VadState::Waiting => {
                self.history_buffer.extend(chunk.iter().copied());

                let rollback_samples = ((self.config.rollback_duration_ms as f32 / 1000.0)
                    * self.config.sample_rate as f32) as usize;
                while self.history_buffer.len() > rollback_samples {
                    self.history_buffer.pop_front();
                }

                if probability > self.config.speech_threshold {
                    self.state = VadState::Recording;
                    self.current_segment.extend(self.history_buffer.iter());
                    self.history_buffer.clear();
                    self.silence_chunks = 0;
                    self.speech_chunks = 0;
                    self.waiting_dropped_chunks = 0;
                    self.notified_silence = false;
                } else if let Some(limit_ms) = self.config.notify_silence_after_ms {
                    self.waiting_dropped_chunks += 1;
                    let dropped_duration = self.waiting_dropped_chunks as f32 * chunk_duration_ms;
                    if dropped_duration >= limit_ms as f32 && !self.notified_silence {
                        self.notified_silence = true;
                        return Some(VadOutput::SilenceNotification);
                    }
                }
                None
            }
            VadState::Recording => {
                self.current_segment.extend(chunk);
                self.speech_chunks += 1;

                if probability > self.config.speech_threshold {
                    self.silence_chunks = 0;
                    let speech_duration_ms = self.speech_chunks as f32 * chunk_duration_ms;
                    if speech_duration_ms >= self.config.max_speech_duration_ms as f32 {
                        return self.finalize_segment(false);
                    }
                } else {
                    self.silence_chunks += 1;
                    let silence_duration_ms = self.silence_chunks as f32 * chunk_duration_ms;
                    if silence_duration_ms >= self.config.silence_duration_ms as f32 {
                        return self.finalize_segment(true);
                    }
                }
                None
            }
        }
    }

    fn finalize_segment(&mut self, trim_tail: bool) -> Option<VadOutput> {
        if self.current_segment.is_empty() {
            self.reset();
            return None;
        }

        let mut segment = if trim_tail {
            let chunk_len = CHUNK_SIZE;
            let silence_len = (self.silence_chunks as usize) * chunk_len;
            let valid_len = self.current_segment.len().saturating_sub(silence_len);
            if valid_len == 0 {
                Vec::new()
            } else {
                self.current_segment[..valid_len].to_vec()
            }
        } else {
            self.current_segment.clone()
        };

        let duration_ms = (segment.len() as f32 / self.config.sample_rate as f32) * 1000.0;
        if duration_ms < self.config.min_speech_duration_ms as f32 {
            segment.clear();
        }

        self.reset();

        if segment.is_empty() {
            None
        } else {
            Some(VadOutput::Segment(segment))
        }
    }

    fn reset(&mut self) {
        self.current_segment.clear();
        self.history_buffer.clear();
        self.silence_chunks = 0;
        self.speech_chunks = 0;
        self.state = VadState::Waiting;
        self.waiting_dropped_chunks = 0;
        self.notified_silence = false;
    }

    pub fn finish(&mut self) -> Option<VadOutput> {
        if !self.current_segment.is_empty() {
            let duration_ms =
                (self.current_segment.len() as f32 / self.config.sample_rate as f32) * 1000.0;
            if duration_ms < self.config.min_speech_duration_ms as f32 {
                self.reset();
                return None;
            }

            let segment = self.current_segment.clone();
            self.reset();
            Some(VadOutput::Segment(segment))
        } else {
            None
        }
    }
}
