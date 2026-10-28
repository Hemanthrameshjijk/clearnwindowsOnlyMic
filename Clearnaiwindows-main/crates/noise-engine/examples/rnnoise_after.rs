//! Runs a raw 48 kHz mono f32le file through the mic pipeline's RNNoise stage (the BVC-off path runs it after
//! Accent + AutoGain): `rnnoise_after <in.f32> <out.f32>`.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let x: Vec<f32> = std::fs::read(&a[1]).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut st = noise_engine::build(noise_engine::NoiseEngineKind::RnNoise, None).unwrap();
    let mut out = Vec::with_capacity(x.len());
    for f in x.chunks_exact(480) {
        let mut fr = f.to_vec();
        dsp_core::Stage::process(st.as_mut(), &mut fr);
        out.extend(fr);
    }
    std::fs::write(&a[2], out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
