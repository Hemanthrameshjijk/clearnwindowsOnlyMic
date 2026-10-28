//! Runs a raw 48 kHz mono f32le file through the mic pipeline's DeepFilterNet BVC backend, 480-sample frames
//! like the live pipeline: `deepfilter_file <in.f32> <out.f32>`.
const FRAME: usize = 480;
const TAIL_PAD: usize = 24_000; // 0.5 s of silence so the model's own delay doesn't cut off the last word

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut x: Vec<f32> = std::fs::read(&a[1]).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    x.resize(x.len() + TAIL_PAD, 0.0);
    x.resize(x.len().div_ceil(FRAME) * FRAME, 0.0);
    let mut st = noise_engine::load_bundled_deep_filter().expect("load DeepFilterNet");
    for f in x.chunks_exact_mut(FRAME) {
        dsp_core::Stage::process(&mut st, f);
    }
    std::fs::write(&a[2], x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
