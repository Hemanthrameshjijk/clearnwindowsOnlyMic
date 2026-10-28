//! Runs a raw 48 kHz mono f32le file through what the mic pipeline does after the Accent stage:
//! `post_chain <in.f32> <out_prefix>` writes `<prefix>_agc.f32` (AutoGain only, the BVC/Studio-off config) and
//! `<prefix>_studio{1,2,3}.f32` (AutoGain + Studio Natural / Balanced / Strong).
use dsp_core::{AutoGainStage, Stage};
use studio_dsp::{StudioPreset, StudioStage};

fn run(x: &[f32], mut stages: Vec<Box<dyn Stage>>) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for f in x.chunks_exact(480) {
        let mut fr = f.to_vec();
        for s in stages.iter_mut() {
            s.process(&mut fr);
        }
        out.extend(fr);
    }
    out
}
fn write(p: &str, x: &[f32]) {
    std::fs::write(p, x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let x: Vec<f32> = std::fs::read(&a[1]).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    write(&format!("{}_agc.f32", a[2]), &run(&x, vec![Box::new(AutoGainStage::new())]));
    for (i, p) in [StudioPreset::Natural, StudioPreset::Balanced, StudioPreset::Strong].into_iter().enumerate() {
        write(&format!("{}_studio{}.f32", a[2], i + 1), &run(&x, vec![Box::new(AutoGainStage::new()), Box::new(StudioStage::new(p))]));
    }
}
