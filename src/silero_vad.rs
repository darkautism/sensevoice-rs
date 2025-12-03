use std::collections::VecDeque;
use voice_activity_detector::{IteratorExt, VoiceActivityDetector};
pub const CHUNK_SIZE: usize = 512;

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
    vad: VoiceActivityDetector,
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
        let vad = VoiceActivityDetector::builder()
            .sample_rate(config.sample_rate)
            .chunk_size(CHUNK_SIZE)
            .build()?;
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
        // 如果關閉了通知，重置狀態以防萬一
        if ms.is_none() {
            self.notified_silence = false;
        }
        // 如果開啟了且當前累積已超過，下一次 process_chunk 會觸發
    }

    pub fn process_chunk(&mut self, chunk: &[i16; CHUNK_SIZE]) -> Option<VadOutput> {
        let chunk_duration_ms = (CHUNK_SIZE as f32 / self.config.sample_rate as f32) * 1000.0;
        let probability = chunk
            .iter()
            .copied()
            .predict(&mut self.vad)
            .next()
            .unwrap()
            .1;

        match self.state {
            VadState::Waiting => {
                // 將塊加入歷史緩衝區
                self.history_buffer.extend(chunk.iter().copied());

                // 維護緩衝區大小 (rollback_duration)
                let rollback_samples = ((self.config.rollback_duration_ms as f32 / 1000.0)
                    * self.config.sample_rate as f32) as usize;
                while self.history_buffer.len() > rollback_samples {
                    self.history_buffer.pop_front();
                }

                if probability > self.config.speech_threshold {
                    // 檢測到語音，切換到 Recording 狀態
                    self.state = VadState::Recording;
                    // 將歷史緩衝區的內容移動到當前段（保留語音開頭的上下文）
                    self.current_segment.extend(self.history_buffer.iter());
                    self.history_buffer.clear(); // 清空緩衝區
                    self.silence_chunks = 0;
                    self.speech_chunks = 0;

                    // 重置 Waiting 相關計數
                    self.waiting_dropped_chunks = 0;
                    self.notified_silence = false;
                } else {
                    // 仍在等待，檢查是否需要發出靜音通知
                    if let Some(limit_ms) = self.config.notify_silence_after_ms {
                        self.waiting_dropped_chunks += 1;
                        let dropped_duration = self.waiting_dropped_chunks as f32 * chunk_duration_ms;
                        if dropped_duration >= limit_ms as f32 && !self.notified_silence {
                            self.notified_silence = true;
                            return Some(VadOutput::SilenceNotification);
                        }
                    }
                }
                None
            }
            VadState::Recording => {
                self.current_segment.extend(chunk);
                self.speech_chunks += 1;

                if probability > self.config.speech_threshold {
                    self.silence_chunks = 0;
                    // 檢查是否超過最大語音長度
                    let speech_duration_ms = self.speech_chunks as f32 * chunk_duration_ms;
                    if speech_duration_ms >= self.config.max_speech_duration_ms as f32 {
                        // 強制切斷
                        return self.finalize_segment(false);
                    }
                } else {
                    self.silence_chunks += 1;
                    let silence_duration_ms = self.silence_chunks as f32 * chunk_duration_ms;
                    if silence_duration_ms >= self.config.silence_duration_ms as f32 {
                        // 靜音時間過長，結束當前段
                        // 並修剪掉尾部的靜音
                        return self.finalize_segment(true);
                    }
                }
                None
            }
        }
    }

    // trim_tail: 是否修剪尾部的靜音
    fn finalize_segment(&mut self, trim_tail: bool) -> Option<VadOutput> {
        if self.current_segment.is_empty() {
            self.reset();
            return None;
        }

        let mut segment = if trim_tail {
            // 計算需要修剪的樣本數
            let chunk_len = CHUNK_SIZE;
            let silence_len = (self.silence_chunks as usize) * chunk_len;
            let valid_len = self.current_segment.len().saturating_sub(silence_len);
            if valid_len == 0 {
                Vec::new() // 全是靜音？
            } else {
                self.current_segment[..valid_len].to_vec()
            }
        } else {
            self.current_segment.clone()
        };

        // 最小長度檢查
        let duration_ms =
            (segment.len() as f32 / self.config.sample_rate as f32) * 1000.0;
        if duration_ms < self.config.min_speech_duration_ms as f32 {
            // 語音太短，視為噪音丟棄
            segment.clear(); // 清空以確保返回 None
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
        // 重置 waiting 狀態
        self.waiting_dropped_chunks = 0;
        self.notified_silence = false;
    }

    pub fn finish(&mut self) -> Option<VadOutput> {
        // 如果還在 Recording 狀態，返回剩餘內容
        if !self.current_segment.is_empty() {
             // 對於最後一段，我們也要做最小長度檢查
             let duration_ms = (self.current_segment.len() as f32 / self.config.sample_rate as f32) * 1000.0;
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
