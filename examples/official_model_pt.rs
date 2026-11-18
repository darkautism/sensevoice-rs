use hf_hub::HFClientSync;
use sensevoice_rs::{silero_vad::VadConfig, SenseVoiceSmall};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let svs = SenseVoiceSmall::init_official_model_pt(VadConfig::default())?;

    let api = HFClientSync::new()?;
    let repo = api.model("happyme531", "SenseVoiceSmall-RKNN2");
    let wav_path = repo.download_file().filename("output.wav").send()?;
    let allseg = svs.infer_file(wav_path)?;
    for seg in allseg {
        println!("{:?}", seg);
    }

    Ok(svs.destroy()?)
}
