use futures::stream::StreamExt;
use hf_hub::api::sync::Api;
use sensevoice_rs::{silero_vad::VadConfig, SenseVoiceSmall};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::thread;
use std::time::Duration; // For .next() in infer_stream if used, but here we pass Reader?
                         // infer_stream expects Stream of Vec<i16>, not a Reader directly unless adapted.
                         // Let's check lib.rs signature.
                         // infer_stream<S>(input_stream: S) where S: Stream<Item = Vec<i16>>
                         // The example code seems to be passing a Reader which is wrong unless there is an adapter.
                         // But I should fix the compilation errors first.

// 自定義 DelayedReader 模擬流式輸入，每次讀取延遲 0.5 秒
struct DelayedReader {
    file: File,
    chunk_size: usize,
    delay: Duration,
    buffer: Vec<u8>,
    pos: usize,
}

impl DelayedReader {
    fn new(mut file: File, chunk_size: usize, delay: Duration) -> Self {
        // 跳過 WAV 頭部（假設 44 字節）
        file.seek(SeekFrom::Start(44)).unwrap();
        DelayedReader {
            file,
            chunk_size,
            delay,
            buffer: Vec::new(),
            pos: 0,
        }
    }

    fn fill_buffer(&mut self) -> std::io::Result<()> {
        self.buffer.clear();
        let mut temp_buf = vec![0u8; self.chunk_size];
        let bytes_read = self.file.read(&mut temp_buf)?;
        if bytes_read > 0 {
            self.buffer.extend_from_slice(&temp_buf[..bytes_read]);
            self.pos = 0;
            // 模擬流式延遲
            thread::sleep(self.delay);
        }
        Ok(())
    }
}

impl Read for DelayedReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buffer.len() {
            self.fill_buffer()?;
            if self.buffer.is_empty() {
                return Ok(0); // 文件結束
            }
        }

        let remaining = self.buffer.len() - self.pos;
        let to_copy = std::cmp::min(remaining, buf.len());
        buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
        self.pos += to_copy;
        Ok(to_copy)
    }
}

// Need to adapt Reader to Stream for infer_stream
use async_stream::stream;
use futures::stream::Stream;

fn reader_to_stream(mut reader: DelayedReader) -> impl Stream<Item = Vec<i16>> {
    stream! {
        loop {
            let mut buf = vec![0u8; 1024]; // bytes
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    // Convert bytes to i16
                    let samples: Vec<i16> = buf[..n].chunks(2).map(|c| {
                        if c.len() == 2 {
                            i16::from_le_bytes([c[0], c[1]])
                        } else {
                            0
                        }
                    }).collect();
                    yield samples;
                }
                Err(_) => break,
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut svs = SenseVoiceSmall::init(VadConfig::default())?;

    let api = Api::new().unwrap();
    let repo = api.model("happyme531/SenseVoiceSmall-RKNN2".to_owned());
    let wav_path = repo.get("output.wav")?;
    let file_for_stream = File::open(&wav_path)?;
    let delayed_reader = DelayedReader::new(file_for_stream, 1024, Duration::from_millis(50)); // Faster delay

    let stream = Box::pin(reader_to_stream(delayed_reader));
    let mut stream_out = Box::pin(svs.infer_stream(stream));

    while let Some(res) = stream_out.next().await {
        match res {
            Ok(seg) => println!("{:?}", seg),
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    // Explicitly drop stream_out to release borrow on svs
    drop(stream_out);

    Ok(svs.destroy()?)
}
