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
    pub log_on_error: bool,          // 是否在 VAD 處理錯誤時印出警告
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
            log_on_error: true,
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

        Ok(Self {
            inner,
            config,
            waiting_dropped_samples: 0,
            notified_silence: false,
            pending_segments: VecDeque::new(),
            audio_buffer: AudioBuffer::new(config.sample_rate as usize),
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

        // Feed to Audio Buffer (clone because we need ownership in buffer)
        self.audio_buffer.input(chunk_f32.clone());

        // Feed to InnerVad
        if let Err(e) = self.inner.feed_chunk(chunk_f32) {
            if self.config.log_on_error {
                eprintln!("SenseVoice VAD Warning: {}", e);
            }
            return None;
        }

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
            ("input".to_string(), chunk),
            ("sr".to_string(), self.state[0].clone()),
            ("state".to_string(), self.state[1].clone()),
        ])
    }

    fn update_state(&mut self, output: &Tensor, context: Tensor) {
        // 將輸出從計算圖中分離，防止記憶體堆積和釋放時的堆疊溢位
        // 如果 detach() 不可用，我們可以重新建立 tensor。
        // 檢查是否可 detach?
        // 假設需要斷開圖。
        // 這裡使用安全的方法：複製數據。
        // 雖然效率稍低，但安全。
        // 且對於 128 個浮點數來說，開銷可以忽略不計。
        let device = output.device();
        let dims = output.dims();
        // 展平為 vec 並重新建立
        // 這保證斷開了圖。
        if let Ok(data) = output.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
            if let Ok(new_tensor) = Tensor::from_vec(data, dims, device) {
                self.state[1] = new_tensor;
            } else {
                // 如果重新建立失敗（不太可能），保持原樣但警告？
                // 或者直接 clone（有崩潰風險）。
                self.state[1] = output.clone();
            }
        } else {
            self.state[1] = output.clone();
        }

        self.state[2] = context;
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

        // 1. 安全取得 Graph
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| Error::Msg("Invalid ONNX model: missing graph".into()))?;

        // 2. 檢查輸出數量是否足夠 (Silero VAD 需要至少 2 個輸出: probability, state)
        if graph.output.len() < 2 {
            return Err(Error::Msg(format!(
                "Invalid VAD model: expected at least 2 outputs, got {}",
                graph.output.len()
            )));
        }

        let out_names = &graph.output;

        // 3. 安全取得 Output Tensor
        let output_name = &out_names[0].name;
        let state_name = &out_names[1].name;

        let output_tensor = out
            .get(output_name)
            .ok_or_else(|| Error::Msg(format!("Model execution output missing: {}", output_name)))?
            .clone(); // 這裡 clone 是安全的因為 simple_eval output 沒有梯度

        let state_tensor = out
            .get(state_name)
            .ok_or_else(|| Error::Msg(format!("Model execution output missing: {}", state_name)))?;

        // 更新狀態
        self.update_state(state_tensor, next_context);

        let output_vec = output_tensor.flatten_all()?.to_vec1::<f32>()?;
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

        let device = Device::Cpu;
        self.state[1] = Tensor::zeros((2, 1, 128), DType::F32, &device)?;
        self.state[2] = Tensor::zeros((1, self.context_size), DType::F32, &device)?;

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
    queue: VecDeque<Vec<f32>>,
    length: usize,
    start: usize,
    offset: usize,
    // sample_rate: usize, // 移除未使用的欄位
}

struct SegmentData {
    pub audio: Vec<f32>,
}

impl AudioBuffer {
    fn new(_sample_rate: usize) -> Self {
        AudioBuffer {
            queue: VecDeque::new(),
            length: 0,
            start: 0,
            offset: 0,
            // sample_rate,
        }
    }

    fn input(&mut self, audio: Vec<f32>) {
        self.length += audio.len();
        self.queue.push_back(audio);
    }

    // Prune data older than threshold
    fn prune(&mut self, threshold_start: usize) {
        // 丟棄 chunk_end <= threshold_start 的區塊
        // self.start 追蹤 queue[0] 的起始位置。
        while !self.queue.is_empty() {
            let first_len = self.queue[0].len();
            if self.start + first_len <= threshold_start {
                self.start += first_len;
                self.queue.pop_front();
                self.length -= first_len;
                self.offset = 0;
            } else {
                break;
            }
        }
    }

    fn output(&mut self, from: usize, to: usize) -> Option<SegmentData> {
        if self.queue.is_empty() || from < self.start + self.offset || to > self.start + self.length
        {
            return None;
        }
        let chunk_size = to - from;
        let mut extracted = Vec::with_capacity(chunk_size);
        let mut needed = chunk_size;
        let mut temp_start = self.start;

        // 迭代查找並提取數據
        for chunk in &self.queue {
            let chunk_len = chunk.len();
            let chunk_end = temp_start + chunk_len;

            // 如果這個 chunk 包含我們需要的數據（完全或部分）
            if chunk_end > from {
                // 計算在該 chunk 內的起始和結束索引
                let start_in_chunk = if from > temp_start {
                    from - temp_start
                } else {
                    0
                };
                // 我們需要的長度是 needed，所以最多讀取到 start_in_chunk + needed
                // 但不能超過 chunk 的長度
                let len_to_read = std::cmp::min(needed, chunk_len - start_in_chunk);
                let end_in_chunk = start_in_chunk + len_to_read;

                if start_in_chunk < end_in_chunk {
                    extracted.extend_from_slice(&chunk[start_in_chunk..end_in_chunk]);
                    needed -= len_to_read;
                }
            }

            temp_start += chunk_len;
            if needed == 0 {
                break;
            }
        }

        if extracted.len() != chunk_size {
            return None;
        }

        Some(SegmentData { audio: extracted })
    }
}
