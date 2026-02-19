use std::collections::HashMap;
use std::path::Path;

use candle::{Device, Tensor};
use ndarray::Array2;

#[derive(Debug)]
pub struct CandleAsrSession {
    model: candle_onnx::onnx::ModelProto,
    output_name: String,
}

impl CandleAsrSession {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = candle_onnx::read_file(model_path)?;
        let graph = model
            .graph
            .as_mut()
            .ok_or_else(|| std::io::Error::other("ONNX model graph is missing"))?;

        // candle-onnx currently doesn't accept INT32 graph inputs in shape validation.
        // SenseVoice exports these control inputs as INT32, so normalize them to INT64.
        for input in graph.input.iter_mut() {
            if ["speech_lengths", "language", "textnorm"]
                .iter()
                .any(|n| *n == input.name)
            {
                if let Some(type_proto) = input.r#type.as_mut() {
                    if let Some(candle_onnx::onnx::type_proto::Value::TensorType(tensor_type)) =
                        type_proto.value.as_mut()
                    {
                        if tensor_type.elem_type == 6 {
                            tensor_type.elem_type = 7;
                        }
                    }
                }
            }
        }

        for required_input in ["speech", "speech_lengths", "language", "textnorm"] {
            if !graph.input.iter().any(|i| i.name == required_input) {
                return Err(std::io::Error::other(format!(
                    "Missing required input '{required_input}' in ASR ONNX graph"
                ))
                .into());
            }
        }

        let output_name = if graph.output.iter().any(|o| o.name == "ctc_logits") {
            "ctc_logits".to_string()
        } else {
            graph
                .output
                .first()
                .map(|o| o.name.clone())
                .ok_or_else(|| std::io::Error::other("ASR ONNX graph has no outputs"))?
        };

        Ok(Self { model, output_name })
    }

    pub fn run(
        &self,
        audio_feats: &Array2<f32>,
        speech_length: i64,
        language: i64,
        textnorm: i64,
    ) -> Result<(Vec<f32>, Vec<i64>), Box<dyn std::error::Error>> {
        let t = audio_feats.shape()[0];
        let d = audio_feats.shape()[1];
        let speech_data: Vec<f32> = audio_feats.iter().copied().collect();

        let speech = Tensor::from_vec(speech_data, (1, t, d), &Device::Cpu)?;
        let speech_lengths = Tensor::from_vec(vec![speech_length], (1,), &Device::Cpu)?;
        let language = Tensor::from_vec(vec![language], (1,), &Device::Cpu)?;
        let textnorm = Tensor::from_vec(vec![textnorm], (1,), &Device::Cpu)?;

        let mut inputs = HashMap::new();
        inputs.insert("speech".to_string(), speech);
        inputs.insert("speech_lengths".to_string(), speech_lengths);
        inputs.insert("language".to_string(), language);
        inputs.insert("textnorm".to_string(), textnorm);

        let mut outputs = candle_onnx::simple_eval(&self.model, inputs)?;
        let output = outputs
            .remove(&self.output_name)
            .or_else(|| outputs.remove("ctc_logits"))
            .ok_or_else(|| {
                std::io::Error::other(format!(
                    "ASR ONNX output '{}' not found",
                    self.output_name
                ))
            })?;

        let shape = output.dims().iter().map(|&x| x as i64).collect::<Vec<_>>();
        let data = output.flatten_all()?.to_vec1::<f32>()?;
        Ok((data, shape))
    }
}
