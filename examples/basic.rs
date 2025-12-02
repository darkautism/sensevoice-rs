use hf_hub::api::sync::Api;
use sensevoice_rs::{silero_vad::VadConfig, SenseVoiceSmall};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // init logic was changed to remove model_path argument
    let svs = SenseVoiceSmall::init(VadConfig::default())?;

    let api = Api::new().unwrap();
    // Use the model repo that contains the sample file.
    // Since init() defaults to different models based on feature,
    // for non-rknpu (default), it uses haixuantao.
    // But haixuantao might not have output.wav.
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
