use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
// 1. 改用同步 API，避免處理 Async Runtime
use hf_hub::api::sync::Api;
use sensevoice_rs::silero_vad::VadConfig;
use sensevoice_rs::SenseVoiceSmall;

fn criterion_benchmark(c: &mut Criterion) {
    // 2. Setup 階段：準備模型與檔案 (直接 unwrap，失敗就崩潰)
    let api = Api::new().expect("Failed to create API");
    let repo = api.model("happyme531/SenseVoiceSmall-RKNN2".to_owned());
    let wav_path = repo.get("output.wav").expect("Failed to download file");

    // 3. 模型初始化 (通常我們想測的是推論速度，所以把 Init 放在迴圈外)
    // 如果你確實想測「包含初始化的完整流程」，再把它搬回 iter 裡面
    let mut svs = SenseVoiceSmall::init(VadConfig::default()).expect("Failed to init model");

    c.bench_function("basic infer", |b| {
        b.iter(|| {
            // 4. 使用 black_box 包果輸入參數，防止編譯器優化
            // 這裡假設 infer_file 接受 &PathBuf 或類似型別
            let allseg = svs.infer_file(black_box(&wav_path)).unwrap();

            // 5. 不要 println!，改用 black_box 吃掉結果
            // 這告訴編譯器「這個結果有用，不要優化掉」，但不會產生 I/O 開銷
            black_box(allseg);
        })
    });

    // 如果 svs 需要手動 destroy，在 benchmark 結束後做一次即可
    // (但在 criterion 函式結束時通常記憶體會釋放，視你的 library 實作而定)
    let _ = svs.destroy();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
