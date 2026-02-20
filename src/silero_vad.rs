use std::collections::VecDeque;

use candle::{DType, Device, Tensor};
use candle_nn::{linear, linear_no_bias, ops::softmax_last_dim, Linear, Module, VarBuilder};
use hf_hub::api::sync::Api;

use crate::wavfrontend::{WavFrontend, WavFrontendConfig};

pub const CHUNK_SIZE: usize = 512;
const FSMN_VAD_REPO: &str = "funasr/fsmn-vad";
const FSMN_VAD_LAYERS: usize = 4;
const FSMN_VAD_PROJ_DIM: usize = 128;
const FSMN_VAD_CACHE_FRAMES: usize = 19;

struct FsmnVadBlock {
    linear: Linear,
    affine: Linear,
    conv_left_weight: Tensor,
}

impl FsmnVadBlock {
    fn new(vb: VarBuilder) -> Result<Self, Box<dyn std::error::Error>> {
        let linear_layer = linear_no_bias(250, FSMN_VAD_PROJ_DIM, vb.pp("linear").pp("linear"))?;
        let affine = linear(FSMN_VAD_PROJ_DIM, 250, vb.pp("affine").pp("linear"))?;
        let conv_left_weight = vb
            .pp("fsmn_block")
            .pp("conv_left")
            .get((FSMN_VAD_PROJ_DIM, 1, 20, 1), "weight")?;
        Ok(Self {
            linear: linear_layer,
            affine,
            conv_left_weight,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        cache: &Tensor,
    ) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
        let x = self.linear.forward(input)?; // [B, T, 128]
        let x_per = x.unsqueeze(1)?.permute((0, 3, 2, 1))?; // [B, 128, T, 1]

        let y_left = Tensor::cat(&[cache, &x_per], 2)?;
        let y_left_t = y_left.dim(2)?;
        let new_cache =
            y_left.narrow(2, y_left_t - FSMN_VAD_CACHE_FRAMES, FSMN_VAD_CACHE_FRAMES)?;

        let y_left = y_left.conv2d(&self.conv_left_weight, 0, 1, 1, FSMN_VAD_PROJ_DIM)?;
        let out = x_per.add(&y_left)?;
        let out = out.permute((0, 3, 2, 1))?.squeeze(1)?; // [B, T, 128]
        let out = self.affine.forward(&out)?.relu()?; // [B, T, 250]
        Ok((out, new_cache))
    }
}

struct FsmnVadModel {
    frontend: WavFrontend,
    in_linear1: Linear,
    in_linear2: Linear,
    fsmn_blocks: Vec<FsmnVadBlock>,
    out_linear1: Linear,
    out_linear2: Linear,
    caches: Vec<Tensor>,
}

impl std::fmt::Debug for FsmnVadModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsmnVadModel").finish()
    }
}

impl FsmnVadModel {
    fn new(config: &VadConfig) -> Result<Self, Box<dyn std::error::Error>> {
        if config.sample_rate != 16000 {
            return Err(
                std::io::Error::other("FSMN-VAD currently supports 16k sample rate").into(),
            );
        }

        let api = Api::new()?;
        let repo = api.model(FSMN_VAD_REPO.to_owned());
        let model_path = repo.get("model.pt")?;
        let cmvn_path = repo.get("am.mvn")?;

        let frontend = WavFrontend::new(WavFrontendConfig {
            sample_rate: config.sample_rate as i32,
            lfr_m: 5,
            lfr_n: 1,
            cmvn_file: Some(cmvn_path.to_string_lossy().to_string()),
            ..Default::default()
        })?;

        let vb = VarBuilder::from_pth(model_path, DType::F32, &Device::Cpu)?;
        let encoder_vb = vb.pp("encoder");
        let in_linear1 = linear(400, 140, encoder_vb.pp("in_linear1").pp("linear"))?;
        let in_linear2 = linear(140, 250, encoder_vb.pp("in_linear2").pp("linear"))?;
        let out_linear1 = linear(250, 140, encoder_vb.pp("out_linear1").pp("linear"))?;
        let out_linear2 = linear(140, 248, encoder_vb.pp("out_linear2").pp("linear"))?;

        let mut fsmn_blocks = Vec::with_capacity(FSMN_VAD_LAYERS);
        for i in 0..FSMN_VAD_LAYERS {
            fsmn_blocks.push(FsmnVadBlock::new(encoder_vb.pp("fsmn").pp(i))?);
        }

        let cache = Tensor::zeros(
            (1usize, FSMN_VAD_PROJ_DIM, FSMN_VAD_CACHE_FRAMES, 1usize),
            DType::F32,
            &Device::Cpu,
        )?;
        let caches = vec![cache; FSMN_VAD_LAYERS];

        Ok(Self {
            frontend,
            in_linear1,
            in_linear2,
            fsmn_blocks,
            out_linear1,
            out_linear2,
            caches,
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

        let mut x = self.in_linear1.forward(&speech)?;
        x = self.in_linear2.forward(&x)?;
        x = x.relu()?;
        for (idx, block) in self.fsmn_blocks.iter().enumerate() {
            let (new_x, new_cache) = block.forward(&x, &self.caches[idx])?;
            x = new_x;
            self.caches[idx] = new_cache;
        }
        x = self.out_linear1.forward(&x)?;
        x = self.out_linear2.forward(&x)?;
        let logits = softmax_last_dim(&x)?;

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
    pub sample_rate: u32,                     // 採樣率，例如 16000 Hz
    pub speech_threshold: f32,                // 語音概率閾值，例如 0.5
    pub silence_duration_ms: u32,             // 靜音持續時間（毫秒），例如 500 ms
    pub max_speech_duration_ms: u32,          // 最大語音段長（毫秒），例如 10000 ms
    pub rollback_duration_ms: u32,            // 剪斷後回退時間（毫秒），例如 200 ms
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
                    * self.config.sample_rate as f32)
                    as usize;
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
