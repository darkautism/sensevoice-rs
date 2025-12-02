use std::collections::VecDeque;
use voice_activity_detector::{IteratorExt, VoiceActivityDetector};
pub const CHUNK_SIZE: usize = 512;

#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    sample_rate: u32,            // 採樣率，例如 16000 Hz
    speech_threshold: f32,       // 語音概率閾值，例如 0.5
    silence_duration_ms: u32,    // 靜音持續時間（毫秒），例如 500 ms
    max_speech_duration_ms: u32, // 最大語音段長（毫秒），例如 10000 ms
    rollback_duration_ms: u32,   // 剪斷後回退時間（毫秒），例如 200 ms
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            speech_threshold: 0.5,
            silence_duration_ms: 500,     // 500 ms 靜音算結束
            max_speech_duration_ms: 9000, // 9 秒最大語音段
            rollback_duration_ms: 200,    // 回退 200 ms
        }
    }
}

#[derive(Debug)]
pub struct VadProcessor {
    vad: VoiceActivityDetector,
    config: VadConfig,
    current_segment: Vec<i16>,      // 當前語音段的樣本
    pending_samples: VecDeque<i16>, // 未完成的樣本，等待下次處理
    silence_chunks: u32,            // 連續靜音塊數
    speech_chunks: u32,             // 當前語音段的塊數
}

impl VadProcessor {
    pub fn new(config: VadConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let vad = VoiceActivityDetector::builder()
            .sample_rate(config.sample_rate)
            .chunk_size(CHUNK_SIZE)
            .build()?;
        Ok(Self {
            vad,
            config,
            current_segment: Vec::new(),
            pending_samples: VecDeque::new(),
            silence_chunks: 0,
            speech_chunks: 0,
        })
    }

    pub fn process_chunk(&mut self, chunk: &[i16; CHUNK_SIZE]) -> Option<Vec<i16>> {
        let chunk_duration_ms = (CHUNK_SIZE as f32 / self.config.sample_rate as f32) * 1000.0;
        let probability = chunk
            .iter()
            .copied()
            .predict(&mut self.vad)
            .next()
            .unwrap()
            .1;

        // 將 chunk 添加到待處理樣本
        self.pending_samples.extend(chunk.iter().copied());

        if probability > self.config.speech_threshold {
            // 語音檢測到，重置靜音計數，增加語音塊數
            self.silence_chunks = 0;
            self.speech_chunks += 1;
            self.current_segment.extend(chunk);

            // 檢查是否超過最大語音長度
            let speech_duration_ms = self.speech_chunks as f32 * chunk_duration_ms;
            if speech_duration_ms >= self.config.max_speech_duration_ms as f32 {
                return self.finalize_segment();
            }
        } else {
            // 靜音檢測到，增加靜音計數
            self.silence_chunks += 1;
            let silence_duration_ms = self.silence_chunks as f32 * chunk_duration_ms;
            if !self.current_segment.is_empty()
                && silence_duration_ms >= self.config.silence_duration_ms as f32
            {
                return self.finalize_segment();
            }
            self.current_segment.extend(chunk); // 靜音部分也先保留，直到確認段結束
        }

        None // 未結束，返回 None
    }

    pub fn finalize_segment(&mut self) -> Option<Vec<i16>> {
        if self.current_segment.is_empty() {
            return None;
        }

        let chunk_duration_ms = (CHUNK_SIZE as f32 / self.config.sample_rate as f32) * 1000.0;
        let rollback_chunks =
            (self.config.rollback_duration_ms as f32 / chunk_duration_ms).ceil() as usize;
        let rollback_samples = rollback_chunks * CHUNK_SIZE;

        // 計算回退樣本數並分割
        let segment_len = self.current_segment.len();
        let rollback_start = segment_len.saturating_sub(rollback_samples);
        let segment = self.current_segment[..rollback_start].to_vec();
        let rollback = self.current_segment[rollback_start..].to_vec();

        // 重置當前段，將回退部分加入待處理
        self.current_segment.clear();
        self.pending_samples.clear();
        self.pending_samples.extend(rollback);
        self.silence_chunks = 0;
        self.speech_chunks = 0;

        Some(segment)
    }

    pub fn finish(&mut self) -> Option<Vec<i16>> {
        // 處理剩餘樣本作為最終段
        if !self.current_segment.is_empty() {
            let segment = self.current_segment.clone();
            self.current_segment.clear();
            self.pending_samples.clear();
            self.silence_chunks = 0;
            self.speech_chunks = 0;
            Some(segment)
        } else {
            None
        }
    }
}
