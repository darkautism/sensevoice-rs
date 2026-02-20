use std::path::Path;

use candle::{DType, Device, Tensor, D};
use candle_nn::{
    conv1d_no_bias, embedding, layer_norm, linear, ops::softmax_last_dim, Conv1d, Conv1dConfig,
    Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use ndarray::Array2;

// Native SenseVoice ASR path implemented from official model config and model.pt tensor layout.
const INPUT_SIZE: usize = 560;
const OUTPUT_SIZE: usize = 512;
const ATTENTION_HEADS: usize = 4;
const LINEAR_UNITS: usize = 2048;
const NUM_BLOCKS: usize = 50;
const TP_BLOCKS: usize = 20;
const KERNEL_SIZE: usize = 11;
const SANM_SHIFT: usize = 0;
const PROMPT_STATIC_IDS: [i64; 2] = [1, 2];
const PROMPT_LEN: i64 = 4;
const CTC_VOCAB_SIZE: usize = 25_055;
const LAYER_NORM_EPS: f64 = 1e-5;

#[derive(Debug)]
struct MultiHeadedAttentionSanm {
    head_dim: usize,
    n_head: usize,
    linear_out: Linear,
    linear_q_k_v: Linear,
    fsmn_block: Conv1d,
    left_padding: usize,
    right_padding: usize,
    scaling: f64,
}

impl MultiHeadedAttentionSanm {
    fn new(
        vb: VarBuilder,
        n_head: usize,
        in_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let head_dim = hidden_dim / n_head;
        let linear_out = linear(hidden_dim, hidden_dim, vb.pp("linear_out"))?;
        let linear_q_k_v = linear(in_dim, hidden_dim * 3, vb.pp("linear_q_k_v"))?;
        let fsmn_block = conv1d_no_bias(
            hidden_dim,
            hidden_dim,
            KERNEL_SIZE,
            Conv1dConfig {
                groups: hidden_dim,
                ..Default::default()
            },
            vb.pp("fsmn_block"),
        )?;
        let mut left_padding = (KERNEL_SIZE - 1) / 2;
        if SANM_SHIFT > 0 {
            left_padding += SANM_SHIFT;
        }
        let right_padding = KERNEL_SIZE - 1 - left_padding;
        let scaling = (head_dim as f64).powf(-0.5);
        Ok(Self {
            head_dim,
            n_head,
            linear_out,
            linear_q_k_v,
            fsmn_block,
            left_padding,
            right_padding,
            scaling,
        })
    }

    fn forward_simple(&self, xs: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let (b, t, _) = xs.dims3()?;
        let q_k_v = self.linear_q_k_v.forward(xs)?;
        let dim = self.head_dim * self.n_head;

        let q_h = q_k_v
            .narrow(D::Minus1, 0, dim)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k_h = q_k_v
            .narrow(D::Minus1, dim, dim)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = q_k_v.narrow(D::Minus1, dim * 2, dim)?;
        let v_h = v
            .reshape((b, t, self.n_head, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        let fsmn_memory = v.transpose(1, 2)?;
        let fsmn_memory =
            fsmn_memory.pad_with_zeros(D::Minus1, self.left_padding, self.right_padding)?;
        let fsmn_memory = self.fsmn_block.forward(&fsmn_memory)?;
        let fsmn_memory = fsmn_memory.transpose(1, 2)?;
        let fsmn_memory = fsmn_memory.add(&v)?;

        let q_h = q_h.affine(self.scaling, 0.0)?;
        let scores = q_h.matmul(&k_h.transpose(D::Minus2, D::Minus1)?)?;
        let attn = softmax_last_dim(&scores)?;
        let att_outs = attn.matmul(&v_h)?;
        let att_outs = att_outs
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, dim))?;
        let att_outs = self.linear_out.forward(&att_outs)?;
        Ok(att_outs.add(&fsmn_memory)?)
    }
}

#[derive(Debug)]
struct EncoderLayerSanm {
    self_attn: MultiHeadedAttentionSanm,
    feed_forward_w1: Linear,
    feed_forward_w2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    in_dim: usize,
    hidden_dim: usize,
}

impl EncoderLayerSanm {
    fn new(
        vb: VarBuilder,
        in_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let self_attn =
            MultiHeadedAttentionSanm::new(vb.pp("self_attn"), ATTENTION_HEADS, in_dim, hidden_dim)?;
        let feed_forward_w1 = linear(hidden_dim, LINEAR_UNITS, vb.pp("feed_forward").pp("w_1"))?;
        let feed_forward_w2 = linear(LINEAR_UNITS, hidden_dim, vb.pp("feed_forward").pp("w_2"))?;
        let norm1 = layer_norm(in_dim, LAYER_NORM_EPS, vb.pp("norm1"))?;
        let norm2 = layer_norm(hidden_dim, LAYER_NORM_EPS, vb.pp("norm2"))?;

        Ok(Self {
            self_attn,
            feed_forward_w1,
            feed_forward_w2,
            norm1,
            norm2,
            in_dim,
            hidden_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let residual = xs.clone();
        let mut xs = self.norm1.forward(xs)?;
        let attn = self.self_attn.forward_simple(&xs)?;
        if self.in_dim == self.hidden_dim {
            xs = residual.add(&attn)?;
        } else {
            xs = attn;
        }

        let residual = xs.clone();
        xs = self.norm2.forward(&xs)?;
        xs = self.feed_forward_w1.forward(&xs)?.relu()?;
        xs = self.feed_forward_w2.forward(&xs)?;
        Ok(residual.add(&xs)?)
    }
}

#[derive(Debug)]
struct SenseVoicePtEncoder {
    encoders0: EncoderLayerSanm,
    encoders: Vec<EncoderLayerSanm>,
    tp_encoders: Vec<EncoderLayerSanm>,
    after_norm: LayerNorm,
    tp_norm: LayerNorm,
    scaling: f64,
    device: Device,
}

impl SenseVoicePtEncoder {
    fn new(vb: VarBuilder, device: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        let encoders0 = EncoderLayerSanm::new(vb.pp("encoders0").pp(0), INPUT_SIZE, OUTPUT_SIZE)?;
        let mut encoders = Vec::with_capacity(NUM_BLOCKS.saturating_sub(1));
        for i in 0..(NUM_BLOCKS - 1) {
            encoders.push(EncoderLayerSanm::new(
                vb.pp("encoders").pp(i),
                OUTPUT_SIZE,
                OUTPUT_SIZE,
            )?);
        }
        let mut tp_encoders = Vec::with_capacity(TP_BLOCKS);
        for i in 0..TP_BLOCKS {
            tp_encoders.push(EncoderLayerSanm::new(
                vb.pp("tp_encoders").pp(i),
                OUTPUT_SIZE,
                OUTPUT_SIZE,
            )?);
        }
        let after_norm = layer_norm(OUTPUT_SIZE, LAYER_NORM_EPS, vb.pp("after_norm"))?;
        let tp_norm = layer_norm(OUTPUT_SIZE, LAYER_NORM_EPS, vb.pp("tp_norm"))?;
        let scaling = (OUTPUT_SIZE as f64).sqrt();
        Ok(Self {
            encoders0,
            encoders,
            tp_encoders,
            after_norm,
            tp_norm,
            scaling,
            device: device.clone(),
        })
    }

    fn positional_encoding(
        &self,
        seq_len: usize,
        dim: usize,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let half_dim = dim / 2;
        let positions =
            Tensor::arange(1f32, (seq_len + 1) as f32, &self.device)?.reshape((seq_len, 1))?;
        let dims = Tensor::arange(0f32, half_dim as f32, &self.device)?;
        let log_timescale = (10_000f64.ln() / ((half_dim as f64) - 1.0)) as f64;
        let inv_timescales = dims.affine(-log_timescale, 0.0)?.exp()?.unsqueeze(0)?;
        let angle = positions.broadcast_mul(&inv_timescales)?;
        let pe = Tensor::cat(&[&angle.sin()?, &angle.cos()?], 1)?;
        Ok(pe.unsqueeze(0)?)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let (_, seq_len, dim) = xs.dims3()?;
        let mut xs = xs.affine(self.scaling, 0.0)?;
        let pe = self.positional_encoding(seq_len, dim)?;
        xs = xs.broadcast_add(&pe)?;

        xs = self.encoders0.forward(&xs)?;
        for layer in self.encoders.iter() {
            xs = layer.forward(&xs)?;
        }
        xs = self.after_norm.forward(&xs)?;
        for layer in self.tp_encoders.iter() {
            xs = layer.forward(&xs)?;
        }
        Ok(self.tp_norm.forward(&xs)?)
    }
}

#[derive(Debug)]
pub struct CandlePtAsrSession {
    device: Device,
    prompt_embed: Embedding,
    encoder: SenseVoicePtEncoder,
    ctc: Linear,
}

impl CandlePtAsrSession {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_pth(model_path, DType::F32, &device)?;
        let prompt_embed = embedding(16, INPUT_SIZE, vb.pp("embed"))?;
        let encoder = SenseVoicePtEncoder::new(vb.pp("encoder"), &device)?;
        let ctc = linear(OUTPUT_SIZE, CTC_VOCAB_SIZE, vb.pp("ctc").pp("ctc_lo"))?;

        Ok(Self {
            device,
            prompt_embed,
            encoder,
            ctc,
        })
    }

    pub fn run(
        &self,
        audio_feats: &Array2<f32>,
        _speech_length: i64,
        language: i64,
        textnorm: i64,
    ) -> Result<(Vec<f32>, Vec<i64>), Box<dyn std::error::Error>> {
        let t = audio_feats.shape()[0];
        let d = audio_feats.shape()[1];
        if d != INPUT_SIZE {
            return Err(std::io::Error::other(format!(
                "Unexpected feature dim {d}, expected {INPUT_SIZE}"
            ))
            .into());
        }

        let speech_data: Vec<f32> = audio_feats.iter().copied().collect();
        let speech = Tensor::from_vec(speech_data, (1, t, d), &self.device)?;

        let lang_idx = Tensor::from_vec(vec![language], (1,), &self.device)?;
        let textnorm_idx = Tensor::from_vec(vec![textnorm], (1,), &self.device)?;
        let static_idx = Tensor::from_vec(
            PROMPT_STATIC_IDS.to_vec(),
            (PROMPT_STATIC_IDS.len(),),
            &self.device,
        )?;

        let lang_embed = self.prompt_embed.forward(&lang_idx)?.unsqueeze(1)?;
        let static_embed = self.prompt_embed.forward(&static_idx)?.unsqueeze(0)?;
        let textnorm_embed = self.prompt_embed.forward(&textnorm_idx)?.unsqueeze(1)?;
        let prompt = Tensor::cat(&[&lang_embed, &static_embed, &textnorm_embed], 1)?;
        let xs = Tensor::cat(&[&prompt, &speech], 1)?;

        let encoded = self.encoder.forward(&xs)?;
        let logits = self.ctc.forward(&encoded)?;

        let shape = logits.dims().iter().map(|&x| x as i64).collect::<Vec<_>>();
        let data = logits.flatten_all()?.to_vec1::<f32>()?;

        // Keep output length behavior aligned with ONNX path (speech + prompt tokens).
        let _encoder_out_lens = (t as i64) + PROMPT_LEN;

        Ok((data, shape))
    }
}
