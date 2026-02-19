use std::path::{Path, PathBuf};
use std::process::Command;

use hf_hub::api::sync::Api;

const MIRROR_ONNX_REPO: &str = "jaman21/SenseVoiceSmall";

fn is_candle_compatible_sensevoice_onnx(path: &Path) -> bool {
    let model = match candle_onnx::read_file(path) {
        Ok(model) => model,
        Err(_) => return false,
    };
    let graph = match model.graph.as_ref() {
        Some(graph) => graph,
        None => return false,
    };

    let has_required_inputs = ["speech", "speech_lengths", "language", "textnorm"]
        .iter()
        .all(|name| graph.input.iter().any(|i| i.name == *name));
    if !has_required_inputs {
        return false;
    }

    let has_ctc_logits = graph.output.iter().any(|o| o.name == "ctc_logits");
    if !has_ctc_logits {
        return false;
    }

    // candle-onnx in this project does not support external tensor data files.
    if graph.initializer.iter().any(|t| t.data_location != 0) {
        return false;
    }

    // Keep compatibility with the currently validated ONNX opset.
    let max_main_opset = model
        .opset_import
        .iter()
        .filter(|x| x.domain.is_empty())
        .map(|x| x.version)
        .max()
        .unwrap_or(0);
    if max_main_opset > 14 {
        return false;
    }

    true
}

fn try_export_model_pt_to_onnx(model_dir: &Path, python: &str) -> bool {
    let script = r#"
import sys
from pathlib import Path
model_dir = sys.argv[1]
from funasr_onnx import SenseVoiceSmall
SenseVoiceSmall(model_dir, batch_size=1, quantize=False)
"#;

    let _ = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(model_dir)
        .output();

    model_dir.join("model.onnx").exists()
}

pub fn resolve_candle_model_path(model_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let is_pt = model_path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.eq_ignore_ascii_case("pt"))
        .unwrap_or(false);
    if !is_pt {
        return Ok(model_path.to_path_buf());
    }

    let model_dir = model_path.parent().ok_or_else(|| {
        std::io::Error::other(format!(
            "model.pt path has no parent directory: {}",
            model_path.display()
        ))
    })?;
    let local_onnx = model_dir.join("model.onnx");

    if local_onnx.exists() && is_candle_compatible_sensevoice_onnx(&local_onnx) {
        return Ok(local_onnx);
    }

    if let Ok(py) = std::env::var("SENSEVOICE_MODEL_PT_PYTHON") {
        if try_export_model_pt_to_onnx(model_dir, &py)
            && local_onnx.exists()
            && is_candle_compatible_sensevoice_onnx(&local_onnx)
        {
            return Ok(local_onnx);
        }
    } else {
        for py in ["python3", "python"] {
            if try_export_model_pt_to_onnx(model_dir, py)
                && local_onnx.exists()
                && is_candle_compatible_sensevoice_onnx(&local_onnx)
            {
                return Ok(local_onnx);
            }
        }
    }

    let api = Api::new()?;
    let repo = api.model(MIRROR_ONNX_REPO.to_owned());
    let mirror_onnx = repo.get("model.onnx")?;
    Ok(mirror_onnx)
}
