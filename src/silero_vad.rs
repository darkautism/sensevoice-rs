use std::collections::{HashMap, VecDeque};

use candle_core::{DType, Device, Error, Tensor};
use candle_onnx::onnx::ModelProto;
use hf_hub::api::sync::Api;

/// 支援的取樣率。
pub const SAMPLE_RATES: [usize; 2] = [8000, 16000];
/// 8kHz 取樣率的區塊大小。
pub const CHUNKS_SR8K: usize = 256;
/// 16kHz 取樣率的區塊大小。
pub const CHUNKS_SR16K: usize = 512;

// Re-export or define constants expected by lib.rs
pub const CHUNK_SIZE: usize = 512; // Assuming 16k for now

static DEFAULT_MODEL_NAME: &str = "onnx-community/silero-vad";
static DEFAULT_MODEL_FILE: &str = "onnx/model.onnx";

// --- Original Public API Structures ---

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

/// Enum to distinguish between a speech segment and a silence notification
#[derive(Debug)]
pub enum VadOutput {
    Segment(Vec<i16>),
    SilenceNotification,
}

pub struct VadProcessor {
    inner: InnerVad,
    config: VadConfig,
    // State for tracking silence notification
    waiting_dropped_samples: usize,
    notified_silence: bool,
    // Buffer for accumulating segments from InnerVad
    pending_segments: VecDeque<Vec<i16>>,
    // Audio buffer to retrieve segment data
    audio_buffer: AudioBuffer,
    // Flag to ensure flush is called only once
    flushed: bool,
}

impl std::fmt::Debug for VadProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VadProcessor")
            .field("config", &self.config)
            .field("waiting_dropped_samples", &self.waiting_dropped_samples)
            .field("notified_silence", &self.notified_silence)
            .field("pending_segments", &self.pending_segments)
            .finish()
    }
}

impl VadProcessor {
    pub fn new(config: VadConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut inner_config = InnerVadConfig::new(
            (config.silence_duration_ms as usize * config.sample_rate as usize) / 1000,
            config.sample_rate as usize,
        );

        // Map VadConfig to InnerVadConfig
        inner_config.threshold = config.speech_threshold;
        inner_config.min_speech =
            (config.min_speech_duration_ms as usize * config.sample_rate as usize) / 1000;
        inner_config.max_speech =
            (config.max_speech_duration_ms as usize * config.sample_rate as usize) / 1000;
        inner_config.speech_pad =
            (config.rollback_duration_ms as usize * config.sample_rate as usize) / 1000;

        let mut inner = InnerVad::new(inner_config);

        // Load model
        inner.load("")?; // Empty string triggers default download

        // Calculate max capacity for AudioBuffer: max_speech_duration + 1000ms buffer
        let max_samples =
            ((config.max_speech_duration_ms + 1000) as usize * config.sample_rate as usize) / 1000;

        Ok(Self {
            inner,
            config,
            waiting_dropped_samples: 0,
            notified_silence: false,
            pending_segments: VecDeque::new(),
            audio_buffer: AudioBuffer::new(config.sample_rate as usize, max_samples),
            flushed: false,
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
        // Convert i16 to f32
        let chunk_f32: Vec<f32> = chunk.iter().map(|&x| x as f32 / 32768.0).collect();

        // Feed to Audio Buffer
        self.audio_buffer.input(&chunk_f32);

        // Feed to InnerVad
        let _ = self.inner.feed_chunk(chunk_f32).ok()?;

        // Check for new segments FIRST, before pruning
        while let Some((start, end)) = self.inner.yield_segment() {
            if let Some(segment_data) = self.audio_buffer.output(start, end) {
                // Convert back to i16 with safe casting
                let segment_i16: Vec<i16> = segment_data
                    .audio
                    .iter()
                    .map(|&x| (x * 32768.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();
                self.pending_segments.push_back(segment_i16);
            }
        }

        // Prune AudioBuffer logic
        // We must keep data for any pending segments in `inner.segments` + current active segment
        let min_required = self.inner.min_required_history();
        // min_required includes speech_pad adjustment internally if needed, or we apply it here?
        // Let's implement min_required_history to return the absolute index we need to keep.
        let keep_threshold = if min_required > self.inner.speech_pad {
            min_required - self.inner.speech_pad
        } else {
            0
        };

        self.audio_buffer.prune(keep_threshold);

        // Return pending segment if any (Prioritize segments over silence notification)
        if let Some(seg) = self.pending_segments.pop_front() {
            return Some(VadOutput::Segment(seg));
        }

        // Handle Silence Notification
        if self.inner.is_idle() {
            self.waiting_dropped_samples += CHUNK_SIZE;
            if let Some(limit_ms) = self.config.notify_silence_after_ms {
                let limit_samples = (limit_ms as usize * self.config.sample_rate as usize) / 1000;
                if self.waiting_dropped_samples >= limit_samples && !self.notified_silence {
                    self.notified_silence = true;
                    return Some(VadOutput::SilenceNotification);
                }
            }
        } else {
            self.waiting_dropped_samples = 0;
            self.notified_silence = false;
        }

        None
    }

    pub fn finish(&mut self) -> Option<VadOutput> {
        if !self.flushed {
            let _ = self.inner.flush();
            self.flushed = true;
        }

        while let Some((start, end)) = self.inner.yield_segment() {
            if let Some(segment_data) = self.audio_buffer.output(start, end) {
                let segment_i16: Vec<i16> = segment_data
                    .audio
                    .iter()
                    .map(|&x| (x * 32768.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();
                self.pending_segments.push_back(segment_i16);
            }
        }

        if let Some(seg) = self.pending_segments.pop_front() {
            return Some(VadOutput::Segment(seg));
        }
        None
    }
}

// --- Inner Implementation (Ported from Crane) ---

#[derive(Debug, Clone)]
struct InnerVadConfig {
    // pub use_cpu: bool, // 移除未使用的欄位
    pub sample_rate: usize,
    pub min_speech: usize,
    pub max_speech: usize,
    pub min_silence: usize,
    pub min_silence_at_max_speech: usize,
    pub speech_pad: usize,
    pub threshold: f32,
    pub hysteresis: f32,
    // pub timestamp_offset: bool, // 移除未使用的欄位
    pub context_size: usize,
}

impl InnerVadConfig {
    fn new(min_silence: usize, sample_rate: usize) -> Self {
        let context_size = if sample_rate == 8000 { 32 } else { 64 };
        InnerVadConfig {
            min_silence,
            sample_rate,
            context_size,
            // use_cpu: false,
            threshold: 0.5,
            hysteresis: 0.15,
            min_speech: 250,
            max_speech: 60_000,
            min_silence_at_max_speech: 98,
            speech_pad: min_silence,
            // timestamp_offset: false,
        }
    }
}

struct InnerVad {
    // config: InnerVadConfig, // 移除未使用的欄位
    sample_rate: usize,
    chunk_size: usize,
    min_speech: usize,
    max_speech: usize,
    min_silence: usize,
    min_silence_at_max_speech: usize,
    speech_pad: usize,
    threshold: f32,
    // hysteresis: f32, // 移除未使用的欄位
    // timestamp_offset: bool, // 移除未使用的欄位
    context_size: usize,
    neg_threshold: f32,
    model: Option<Box<ModelProto>>,
    state: Vec<Tensor>,
    triggered: bool,
    head: usize,
    tail: usize,
    temp_end: usize,
    prev_end: usize,
    next_start: usize,
    current_start: usize,
    current_end: usize,
    padded: bool,
    segments: Vec<(usize, usize)>,
    // buffer: Vec<f32>, // 移除未使用的欄位 (InnerVad 自己的 buffer 未被使用，我們用 AudioBuffer)
    input_key: String,
    sr_key: String,
    state_key: String,
}

impl InnerVad {
    fn new(config: InnerVadConfig) -> Self {
        let sr = config.sample_rate;
        let chunk_size = if sr == 8000 {
            CHUNKS_SR8K
        } else {
            CHUNKS_SR16K
        };

        let min_speech = config.min_speech;
        let speech_pad = config.speech_pad;
        let max_speech = config.max_speech - chunk_size - 2 * speech_pad;
        let min_silence = config.min_silence;
        let min_silence_at_max_speech = config.min_silence_at_max_speech;

        let neg_threshold = config.threshold - config.hysteresis;

        InnerVad {
            sample_rate: sr,
            chunk_size,
            min_speech,
            max_speech,
            min_silence,
            min_silence_at_max_speech,
            speech_pad,
            neg_threshold,
            threshold: config.threshold,
            // hysteresis: config.hysteresis,
            context_size: config.context_size,
            // timestamp_offset: config.timestamp_offset,
            // config,
            model: None,
            state: vec![],
            triggered: false,
            head: 0,
            tail: 0,
            temp_end: 0,
            prev_end: 0,
            next_start: 0,
            current_start: 0,
            current_end: 0,
            padded: true,
            segments: vec![],
            // buffer: vec![],
            input_key: "input".to_string(),
            sr_key: "sr".to_string(),
            state_key: "state".to_string(),
        }
    }

    fn load(&mut self, model_file: impl AsRef<str>) -> candle_core::Result<()> {
        self.reset()?;
        let model_file = model_file.as_ref();
        let model = if model_file.is_empty() {
            let api = Api::new().map_err(|e| Error::Msg(format!("Api Error: {}", e)))?;
            let model_path = api
                .model(DEFAULT_MODEL_NAME.into())
                .get(DEFAULT_MODEL_FILE)
                .map_err(|e| Error::Msg(format!("Download Error: {}", e)))?;
            candle_onnx::read_file(model_path)?
        } else {
            candle_onnx::read_file(model_file)?
        };
        self.model = Some(Box::new(model));
        Ok(())
    }

    fn reset(&mut self) -> candle_core::Result<()> {
        let device = Device::Cpu;
        let sr = Tensor::new(self.sample_rate as i64, &device)?;
        let previous = Tensor::zeros((2, 1, 128), DType::F32, &device)?;
        let context = Tensor::zeros((1, self.context_size), DType::F32, &device)?;
        self.state = vec![sr, previous, context];
        self.triggered = false;
        self.head = 0;
        self.tail = 0;
        self.temp_end = 0;
        self.prev_end = 0;
        self.next_start = 0;
        self.current_start = 0;
        self.current_end = 0;
        self.padded = true;
        self.segments.clear();
        // self.buffer.clear();
        Ok(())
    }

    fn yield_segment(&mut self) -> Option<(usize, usize)> {
        if self.segments.is_empty() {
            return None;
        }
        if self.segments.len() == 1 && !self.padded {
            return None;
        }
        let segment = self.segments.remove(0);
        self.tail = segment.1;
        Some(segment)
    }

    fn is_idle(&self) -> bool {
        self.segments.is_empty() && !self.triggered
    }

    fn inputs(&self, chunk: Tensor) -> HashMap<String, Tensor> {
        HashMap::from_iter([
            (self.input_key.clone(), chunk),
            (self.sr_key.clone(), self.state[0].clone()),
            (self.state_key.clone(), self.state[1].clone()),
        ])
    }

    fn update_state(&mut self, output: &Tensor, context: Tensor) {
        self.state[1] = output.detach();
        self.state[2] = context.detach();
    }

    fn feed_chunk(&mut self, mut chunk: Vec<f32>) -> candle_core::Result<f32> {
        let device = Device::Cpu;
        self.head += chunk.len();

        if chunk.len() < self.chunk_size {
            chunk.resize(self.chunk_size, 0f32);
        } else {
            chunk.truncate(self.chunk_size);
        }

        let chunk_size = self.chunk_size;
        let context_size = self.context_size;

        let next_context = Tensor::from_slice(
            &chunk[chunk_size - context_size..],
            (1, context_size),
            &device,
        )?;

        let chunk_tensor = Tensor::from_vec(chunk, (1, chunk_size), &device)?;
        let chunk_input = Tensor::cat(&[&self.state[2], &chunk_tensor], 1)?;

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::Msg("Model not loaded".into()))?;

        let out = candle_onnx::simple_eval(model, self.inputs(chunk_input))?;
        let out_names = &model.graph.as_ref().unwrap().output;
        let output = out.get(&out_names[0].name).unwrap().clone();
        self.update_state(out.get(&out_names[1].name).unwrap(), next_context);

        let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
        let prob = output_vec[0];

        self.make_segment(prob);
        Ok(prob)
    }

    fn make_segment(&mut self, prob: f32) {
        let offset = self.head;
        if prob >= self.threshold {
            if self.temp_end > 0 {
                self.temp_end = 0;
                if self.next_start < self.prev_end {
                    self.next_start = offset;
                }
            }
            if !self.triggered {
                self.finish_padding(true);
                self.triggered = true;
                self.current_start = offset;
                return;
            }
        }

        if self.triggered && offset - self.current_start > self.max_speech {
            if self.prev_end > 0 {
                self.current_end = self.prev_end;
                self.push_segment();
                if self.next_start < self.prev_end {
                    self.triggered = false;
                } else {
                    self.current_start = self.next_start;
                }
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
            } else {
                self.current_end = offset;
                self.push_segment();
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.triggered = false;
                return;
            }
        }

        if self.triggered && prob < self.neg_threshold {
            if self.temp_end == 0 {
                self.temp_end = offset;
            }
            if offset - self.temp_end > self.min_silence_at_max_speech {
                self.prev_end = self.temp_end;
            }
            if offset - self.temp_end < self.min_silence {
                return;
            } else {
                self.current_end = self.temp_end;
                if self.current_end - self.current_start > self.min_speech {
                    self.push_segment();
                }
                self.current_start = 0;
                self.current_end = 0;
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.triggered = false;
                return;
            }
        }

        self.finish_padding(false);
    }

    fn finish_padding(&mut self, triggering: bool) {
        if self.padded || (self.triggered && !triggering) {
            return;
        }
        if let Some(s) = self.segments.last_mut() {
            let silence = self.head - s.1;
            let pad = if silence > self.speech_pad * 2 {
                self.speech_pad
            } else if triggering {
                silence / 2
            } else {
                return;
            };
            s.1 += pad;
            self.padded = true;
        }
    }

    fn push_segment(&mut self) {
        let mut start = self.current_start;
        if self.segments.is_empty() {
            start = if start > self.tail + self.speech_pad {
                start - self.speech_pad
            } else {
                self.tail
            };
        } else {
            if let Some(last) = self.segments.last() {
                let last_end = last.1;
                start = if start > last_end + self.speech_pad {
                    start - self.speech_pad
                } else {
                    last_end
                };
            }
        }
        self.segments.push((start, self.current_end));
        self.current_start = 0;
        self.current_end = 0;
        self.padded = false;
    }

    fn flush(&mut self) -> candle_core::Result<&[(usize, usize)]> {
        if (self.current_end > 0 || self.current_start > 0)
            && self.head - self.current_start > self.min_speech
        {
            self.current_end = self.head;
            self.push_segment()
        }
        self.finish_padding(true);
        self.padded = true;
        self.triggered = false;

        self.current_start = 0;
        self.current_end = 0;
        self.prev_end = 0;
        self.next_start = 0;
        self.temp_end = 0;
        self.tail = self.head;

        self.state[1] = Tensor::zeros_like(&self.state[1])?;
        self.state[2] = Tensor::zeros_like(&self.state[2])?;

        Ok(&self.segments)
    }

    fn min_required_history(&self) -> usize {
        // 如果有待處理的區段，我們需要第一個區段的起始位置。
        if let Some(first) = self.segments.first() {
            return first.0;
        }
        // 如果已觸發，我們需要 current_start
        if self.triggered {
            return self.current_start;
        }
        // 否則我們需要 head（caller 會處理 padding）
        self.head
    }
}

struct AudioBuffer {
    queue: VecDeque<f32>,
    start: usize, // The absolute index of the first sample in the queue
}

struct SegmentData {
    pub audio: Vec<f32>,
}

impl AudioBuffer {
    fn new(_sample_rate: usize, capacity: usize) -> Self {
        AudioBuffer {
            queue: VecDeque::with_capacity(capacity),
            start: 0,
        }
    }

    fn input(&mut self, audio: &[f32]) {
        self.queue.extend(audio.iter());
    }

    // Prune data older than threshold
    fn prune(&mut self, threshold_start: usize) {
        if threshold_start > self.start {
            let to_remove = threshold_start - self.start;
            let to_remove = std::cmp::min(to_remove, self.queue.len());
            self.queue.drain(0..to_remove);
            self.start += to_remove;
        }
    }

    fn output(&mut self, from: usize, to: usize) -> Option<SegmentData> {
        if from < self.start {
            return None;
        }

        let relative_start = from - self.start;
        let len = to - from;

        if relative_start + len > self.queue.len() {
            return None;
        }

        let mut extracted = Vec::with_capacity(len);
        extracted.extend(self.queue.iter().skip(relative_start).take(len));

        Some(SegmentData { audio: extracted })
    }
}
