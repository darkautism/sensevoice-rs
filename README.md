# SenseVoiceSmall [![dependency status](https://deps.rs/repo/github/darkautism/sensevoice-rs/status.svg)](https://deps.rs/repo/github/darkautism/sensevoice-rs)

A Pure Rust speech recognition library, using Candle for the non-RKNN runtime and RKNN for Rockchip NPU runtime.

## Rockchip Installation Only

You need to install `rknn.so` first:

```bash
sudo curl -L https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so -o /lib/librknnrt.so
```

Then, add the feature gate `rknpu` in your `Cargo.toml`.

## Runtime Notes

- Pure Rust ASR path: Candle + official `model.pt` (native PT loading).
- VAD path: Candle + official `funasr/fsmn-vad` `model.pt` (auto-downloaded by hf-hub).
- RKNN path: keep `rknpu` backend for Rockchip NPU.
- No external ONNX Runtime (`ort`) library is required.

## Usage & Example

This library provides two methods: it can process either an audio file or an audio stream.
Default VAD now follows the official FSMN-VAD path.

For official `model.pt`, use `SenseVoiceSmall::init_official_model_pt(...)` (or pass a `.pt` via `init_with_config`).
The Candle ASR runtime now expects `.pt` directly.

See the [examples](examples) directory for more details.

```Rust
use hf_hub::api::sync::Api;
use sensevoice_rs::{silero_vad::VadConfig, SenseVoiceSmall};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // init logic was changed to remove model_path argument
    let svs = SenseVoiceSmall::init(VadConfig::default())?;

    let api = Api::new().unwrap();
    // happyme531/SenseVoiceSmall-RKNN2 has output.wav.
    let repo = api.model("happyme531/SenseVoiceSmall-RKNN2".to_owned());
    // Use try-catch or ensure file exists.
    // For basic example, we assume we can download it.
    let wav_path = repo.get("output.wav")?;
    let allseg = svs.infer_file(wav_path)?;
    for seg in allseg {
        println!("{:?}", seg);
    }

    Ok(svs.destroy()?)
}

```

## Output Example

```Rust
VoiceText { language: NoSpeech, emotion: Unknown, event: Unknown, punctuation_normalization: Woitn, content: "" }
VoiceText { language: Zh, emotion: Happy, event: Bgm, punctuation_normalization: Woitn, content: "大家好喵今天给大家分享的是在线一线语音生成网站的合集能够更加方便大家选择自己想要生成的角色进入网站" }
VoiceText { language: Zh, emotion: Neutral, event: Bgm, punctuation_normalization: Woitn, content: "生成模型都在这里选择你想要深藏的角色点击进入就来到了" }
VoiceText { language: Zh, emotion: Happy, event: Bgm, punctuation_normalization: Woitn, content: "生成的页面在文本框内输入你想要生成的内容然后点击三层你的" }
VoiceText { language: Ja, emotion: Unknown, event: Bgm, punctuation_normalization: Woitn, content: "" }
VoiceText { language: NoSpeech, emotion: Unknown, event: Unknown, punctuation_normalization: Woitn, content: "" }
VoiceText { language: Zh, emotion: Neutral, event: Bgm, punctuation_normalization: Woitn, content: "另外呢因为每次的生成结果都会有一些不一样的地方如果您觉得第一次的生成效果不好的话可以尝试重新生成也可以稍微调节一下现面的注意" }
VoiceText { language: Zh, emotion: Neutral, event: Bgm, punctuation_normalization: Woitn, content: "在深造事实" }
VoiceText { language: Zh, emotion: Neutral, event: Bgm, punctuation_normalization: Woitn, content: "同时一定要遵守法律法规不可以损害刷人的形象哦" }
VoiceText { language: En, emotion: Unknown, event: Bgm, punctuation_normalization: Woitn, content: "" }

```
