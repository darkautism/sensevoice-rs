use hf_hub::HFClientSync;
use sensevoice_rs::{silero_vad::VadConfig, SenseVoiceSmall};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // init logic was changed to remove model_path argument
    let svs = SenseVoiceSmall::init(VadConfig::default())?;

    let api = HFClientSync::new()?;
    // happyme531/SenseVoiceSmall-RKNN2 has output.wav.
    let repo = api.model("happyme531", "SenseVoiceSmall-RKNN2");
    // Use try-catch or ensure file exists.
    // For basic example, we assume we can download it.
    let wav_path = repo.download_file().filename("output.wav").send()?;
    let allseg = svs.infer_file(wav_path)?;
    for seg in allseg {
        println!("{:?}", seg);
    }

    Ok(svs.destroy()?)
}
