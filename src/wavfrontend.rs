use std::{ffi::CStr, fs::File};

use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};
use ndarray::{s, Array1, Array2};

/// Represents the type of window function used in feature extraction.
///
/// This enum defines the supported window types for audio frame processing.
#[derive(Debug, Clone, Copy, Default)]
pub enum WindowType {
    /// Hanning window (default).
    #[default]
    Hanning,
    /// Sine window.
    Sine,
    /// Hamming window.
    Hamming,
    /// Povey window.
    Povey,
    /// Rectangular window.
    Rectangular,
    /// Blackman window.
    Blackman,
}

/// Configuration for the `WavFrontend` audio feature extraction system.
///
/// This structure defines parameters for processing waveforms into mel-frequency features.
#[derive(Debug, Clone)]
pub struct WavFrontendConfig {
    /// Sample rate of the audio in Hz (e.g., 16000).
    pub sample_rate: i32,
    /// Length of each frame in milliseconds.
    pub frame_length_ms: f32,
    /// Shift between consecutive frames in milliseconds.
    pub frame_shift_ms: f32,
    /// Number of mel filter banks.
    pub n_mels: usize,
    /// Number of frames to stack for low frame rate (LFR) processing.
    pub lfr_m: usize,
    /// Frame interval for LFR processing.
    pub lfr_n: usize,
    /// Optional path to the CMVN (cepstral mean and variance normalization) file.
    pub cmvn_file: Option<String>,
    /// Type of window function to apply to each frame.
    pub window_type: WindowType,
}

/// Implementation of the `Default` trait for `WavFrontendConfig`.
impl Default for WavFrontendConfig {
    /// Creates a `WavFrontendConfig` instance with default values.
    ///
    /// # Returns
    ///
    /// A `WavFrontendConfig` instance with commonly used default settings.
    fn default() -> Self {
        WavFrontendConfig {
            // Todo: 其實我不知道正確的config是甚麼，這裡是根據我專案寫的
            sample_rate: 16000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            n_mels: 80,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_file: None,
            window_type: WindowType::default(),
        }
    }
}

/// Audio feature extraction frontend for processing waveforms.
///
/// This structure handles the extraction of mel-frequency features from audio data, optionally applying LFR and CMVN.
#[derive(Debug, Clone)]
pub struct WavFrontend {
    /// Configuration settings for feature extraction.
    config: WavFrontendConfig,
    /// Optional array of mean values for CMVN.
    cmvn_means: Option<Array1<f32>>,
    /// Optional array of variance values for CMVN.
    cmvn_vars: Option<Array1<f32>>,
}

/// Implementation of methods for `WavFrontend`.
impl WavFrontend {
    /// Creates a new `WavFrontend` instance with the specified configuration.
    ///
    /// If a CMVN file is provided in the config, it loads the mean and variance values for normalization.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration settings for the frontend.
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `WavFrontend` instance or an error if CMVN loading fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the CMVN file cannot be opened or parsed.
    ///
    /// # Example
    ///
    /// ```
    /// use wavfrontend::{WavFrontend, WavFrontendConfig};
    ///
    /// let config = WavFrontendConfig::default();
    /// let frontend = WavFrontend::new(config).expect("Failed to initialize WavFrontend");
    /// ```
    pub fn new(config: WavFrontendConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let (cmvn_means, cmvn_vars) = if let Some(cmvn_path) = &config.cmvn_file {
            let (means, vars) = Self::load_cmvn(cmvn_path)?;
            (Some(means), Some(vars))
        } else {
            (None, None)
        };
        Ok(WavFrontend {
            config,
            cmvn_means,
            cmvn_vars,
        })
    }

    /// Loads CMVN statistics from a file.
    ///
    /// Parses a CMVN file to extract mean and variance vectors for normalization.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CMVN file.
    ///
    /// # Returns
    ///
    /// A `Result` containing a tuple of mean and variance arrays (`Array1<f32>`).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or if the CMVN data is malformed.
    fn load_cmvn(path: &str) -> Result<(Array1<f32>, Array1<f32>), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut lines = std::io::BufRead::lines(reader);

        let mut means = Vec::new();
        let mut vars = Vec::new();
        let mut is_means = false;
        let mut is_vars = false;

        while let Some(line) = lines.next() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts[0] == "<AddShift>" {
                is_means = true;
                continue;
            } else if parts[0] == "<Rescale>" {
                is_vars = true;
                continue;
            } else if parts[0] == "<LearnRateCoef>" && is_means {
                means = parts[3..parts.len() - 1]
                    .iter()
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect();
                is_means = false;
            } else if parts[0] == "<LearnRateCoef>" && is_vars {
                vars = parts[3..parts.len() - 1]
                    .iter()
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect();
                is_vars = false;
            }
        }

        Ok((Array1::from_vec(means), Array1::from_vec(vars)))
    }

    /// Computes mel-frequency filterbank (fbank) features from a waveform.
    ///
    /// Uses the `kaldi_fbank_rust` library to extract fbank features based on the configured parameters.
    ///
    /// # Arguments
    ///
    /// * `waveform` - Slice of audio samples as 32-bit floats.
    ///
    /// # Returns
    ///
    /// A `Result` containing a 2D array of fbank features with shape `(frames, n_mels)`.
    ///
    /// # Errors
    ///
    /// Returns an error if feature computation fails or if the output array cannot be constructed.
    fn compute_fbank_features(
        &self,
        waveform: &[f32],
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        // 創建 FbankOptions，與 Python 版本一致
        let opt = FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: self.config.sample_rate as f32, // 16000 Hz
                window_type: CStr::from_bytes_with_nul(b"hamming\0").unwrap().as_ptr(), // "hamming"
                dither: 0.0,                               // 無抖動
                frame_shift_ms: self.config.frame_shift_ms, // 10.0 ms
                frame_length_ms: self.config.frame_length_ms, // 25.0 ms
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: self.config.n_mels as i32, // 80
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };

        // 創建 OnlineFbank 實例
        let mut fbank = OnlineFbank::new(opt);

        // 傳入波形數據
        fbank.accept_waveform(self.config.sample_rate as f32, waveform);

        // 獲取特徵幀數
        let frames = fbank.num_ready_frames();

        // 收集所有特徵幀
        let mut fbank_feats = Vec::with_capacity(frames as usize);
        for i in 0..frames {
            let frame = fbank.get_frame(i).expect("Should have frame"); // 獲取第 i 幀特徵
            fbank_feats.push(frame.to_vec());
        }

        // 轉換為 Array2<f32>，形狀為 (frames, n_mels)
        let fbank_array = Array2::from_shape_vec(
            (frames as usize, self.config.n_mels),
            fbank_feats.into_iter().flatten().collect(),
        )?;

        Ok(fbank_array)
    }

    /// Applies low frame rate (LFR) processing to fbank features.
    ///
    /// Stacks and subsamples frames to reduce the frame rate, padding as necessary.
    ///
    /// # Arguments
    ///
    /// * `fbank` - 2D array of fbank features.
    /// * `lfr_m` - Number of frames to stack.
    /// * `lfr_n` - Frame interval for subsampling.
    ///
    /// # Returns
    ///
    /// A 2D array of LFR-processed features with shape `(t_lfr, n_mels * lfr_m)`.
    fn apply_lfr(&self, fbank: &Array2<f32>, lfr_m: usize, lfr_n: usize) -> Array2<f32> {
        let t = fbank.shape()[0];
        let t_lfr = ((t as f32) / lfr_n as f32).ceil() as usize;
        let left_padding_rows = (lfr_m - 1) / 2;
        let mut padded_fbank = Array2::zeros((t + left_padding_rows, fbank.shape()[1]));

        for i in 0..left_padding_rows {
            padded_fbank
                .slice_mut(s![i, ..])
                .assign(&fbank.slice(s![0, ..]));
        }
        for i in 0..t {
            padded_fbank
                .slice_mut(s![i + left_padding_rows, ..])
                .assign(&fbank.slice(s![i, ..]));
        }

        let feat_dim = self.config.n_mels * lfr_m;
        let mut lfr_array = Array2::zeros((t_lfr, feat_dim));
        for i in 0..t_lfr {
            let start = i * lfr_n;
            let end = if lfr_m <= t + left_padding_rows - start {
                start + lfr_m
            } else {
                t + left_padding_rows
            };
            let frame = padded_fbank.slice(s![start..end, ..]);
            let mut target = lfr_array.slice_mut(s![i, ..frame.len()]);
            target.assign(&frame.into_shape_with_order(frame.len()).unwrap());
            if end < start + lfr_m {
                let last_row = padded_fbank.slice(s![-1, ..]);
                for j in end - start..lfr_m {
                    lfr_array
                        .slice_mut(s![i, j * self.config.n_mels..(j + 1) * self.config.n_mels])
                        .assign(&last_row);
                }
            }
        }
        lfr_array
    }

    /// Applies cepstral mean and variance normalization (CMVN) to features.
    ///
    /// Normalizes the features using precomputed mean and variance values if available.
    ///
    /// # Arguments
    ///
    /// * `feats` - 2D array of features to normalize.
    ///
    /// # Returns
    ///
    /// A 2D array of normalized features, or the original features if no CMVN data is provided.
    fn apply_cmvn(&self, feats: &Array2<f32>) -> Array2<f32> {
        if let (Some(means), Some(vars)) = (&self.cmvn_means, &self.cmvn_vars) {
            let (frames, dim) = feats.dim();
            let means_expanded = means.broadcast((frames, dim)).unwrap();
            let vars_expanded = vars.broadcast((frames, dim)).unwrap();
            (feats + &means_expanded) * &vars_expanded
        } else {
            feats.to_owned()
        }
    }

    /// Extracts mel-frequency features from a waveform.
    ///
    /// Processes the input waveform to produce fbank features, applies LFR, and optionally applies CMVN.
    ///
    /// # Arguments
    ///
    /// * `waveform` - Slice of audio samples as 16-bit integers.
    ///
    /// # Returns
    ///
    /// A `Result` containing a 2D array of extracted features.
    ///
    /// # Errors
    ///
    /// Returns an error if fbank computation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use wavfrontend::{WavFrontend, WavFrontendConfig};
    ///
    /// let config = WavFrontendConfig::default();
    /// let frontend = WavFrontend::new(config).expect("Failed to initialize");
    /// let waveform = vec![0_i16; 16000]; // Example waveform
    /// let features = frontend.extract_features(&waveform)
    ///     .expect("Failed to extract features");
    /// println!("Feature shape: {:?}", features.shape());
    /// ```
    pub fn extract_features(
        &self,
        waveform: &[i16],
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let waveform: Vec<f32> = waveform.iter().map(|&s| s as f32).collect();
        let fbank = self.compute_fbank_features(&waveform)?;
        let lfr_feats = self.apply_lfr(&fbank, self.config.lfr_m, self.config.lfr_n);
        let feats = self.apply_cmvn(&lfr_feats);
        Ok(feats)
    }
}
