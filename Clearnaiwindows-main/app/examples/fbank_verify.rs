//! Verifies `tse_bsrnn::tse_speaker_encoder`'s Rust fbank output against a
//! real Python `torchaudio.compliance.kaldi.fbank` reference on the same
//! real audio (`poc/real_tse/data/reference.wav`) - this crate's own docs
//! flagged this as NEVER numerically cross-verified; this closes that gap.
//! Dither disabled on both sides for deterministic comparison (production
//! code keeps dither=1.0 - only this verification pass differs).
use kaldi_native_fbank::mel::MelOptions;
use kaldi_native_fbank::online::FeatureComputer;
use kaldi_native_fbank::{FbankComputer, FbankOptions, FrameOptions, OnlineFeature};

const FEAT_DIM: usize = 80;
const SAMPLE_SCALE: f32 = 32768.0;

fn main() {
    let mut reader = hound::WavReader::open("poc/real_tse/data/reference.wav").unwrap();
    let spec = reader.spec();
    let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    println!("Loaded {} samples @ {}Hz", samples.len(), spec.sample_rate);

    let scaled: Vec<f32> = samples.iter().map(|&s| s * SAMPLE_SCALE).collect();

    let opts = FbankOptions {
        frame_opts: FrameOptions {
            samp_freq: spec.sample_rate as f32,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            dither: 0.0, // matches the Python reference run for this verification
            preemph_coeff: 0.97,
            remove_dc_offset: true,
            window_type: "hamming".to_string(),
            round_to_power_of_two: true,
            blackman_coeff: 0.42,
            snip_edges: true,
        },
        mel_opts: MelOptions { num_bins: FEAT_DIM, low_freq: 20.0, high_freq: -400.0, ..MelOptions::default() },
        use_energy: false,
        ..FbankOptions::default()
    };

    let computer = FbankComputer::new(opts).unwrap();
    let mut online = OnlineFeature::new(FeatureComputer::Fbank(computer));
    online.accept_waveform(spec.sample_rate as f32, &scaled);
    online.input_finished();

    let num_frames = online.num_frames_ready();
    println!("Rust fbank shape: [{num_frames}, {FEAT_DIM}]");

    let mut flat = vec![0.0f32; num_frames * FEAT_DIM];
    for (i, out_row) in flat.chunks_mut(FEAT_DIM).enumerate() {
        let frame = online.get_frame(i).unwrap();
        out_row.copy_from_slice(frame);
    }

    println!("First row (first 10 vals): {:?}", &flat[..10]);
    let mean: f64 = flat.iter().map(|&v| v as f64).sum::<f64>() / flat.len() as f64;
    let var: f64 = flat.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / flat.len() as f64;
    println!("mean: {mean} std: {}", var.sqrt());

    // Compare against the Python reference.
    let python_bytes = std::fs::read("poc/real_tse/data/reference_fbank_python.bin").expect("run the Python script first");
    let python: Vec<f32> = python_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    println!("\nPython fbank total values: {}, Rust: {}", python.len(), flat.len());

    let n = python.len().min(flat.len());
    let max_abs_diff = python[..n].iter().zip(flat[..n].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mean_abs_diff: f64 = python[..n].iter().zip(flat[..n].iter()).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / n as f64;
    println!("max_abs_diff={max_abs_diff:.6} mean_abs_diff={mean_abs_diff:.6}");

    // Per-frame mean abs diff, to localize WHERE divergence starts.
    println!("\nPer-frame mean_abs_diff (every 20th frame):");
    for frame_idx in (0..num_frames).step_by(20) {
        let start = frame_idx * FEAT_DIM;
        let end = start + FEAT_DIM;
        if end > n {
            break;
        }
        let d: f64 = python[start..end].iter().zip(flat[start..end].iter()).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / FEAT_DIM as f64;
        println!("frame {frame_idx}: mean_abs_diff={d:.4} python_row0={:.2} rust_row0={:.2}", python[start], flat[start]);
    }
}
