//! Standalone verification tool - NOT part of the ClearNAI product.
//!
//! Opens the same named shared-memory section `clearnairt.exe`'s mic
//! pipeline writes into (`driver_transport::MIC_OUTPUT_SECTION_NAME`) from
//! a completely separate process, and dumps whatever it reads to a real
//! WAV file. This is the only way to actually prove the cross-process
//! transport works, independent of Voicemeeter/any virtual audio driver -
//! it talks to `driver-transport`'s shared section directly, nothing else.
//!
//! # What this proves
//!
//! "ClearNAI's processed mic audio really does reach a separate OS
//! process, correctly, in real time." That's it.
//!
//! # What this does NOT prove or provide
//!
//! - It is not a Windows audio device. Zoom/Meet/Discord cannot select it
//!   as a microphone. It never will be - that's what an actual kernel
//!   driver (see `docs/DRIVER_PLAN.md`) would be for.
//! - It does not replace Voicemeeter for real calls.
//! - It says nothing about a future driver's own correctness - a kernel-
//!   mode reader is a different, unverified piece of code entirely.
//!
//! # Usage
//!
//! Run `clearnairt.exe` first (it creates the section at startup and keeps
//! writing to it every 10ms regardless of whether anything reads it), then
//! in a separate terminal:
//!
//! ```text
//! cargo run -p driver-transport --features reader-tool --bin transport-reader -- 10 out.wav
//! ```
//!
//! captures 10 seconds starting now, writes to `out.wav` (mono, 48kHz,
//! 32-bit float - play it in anything that reads WAV). Args are optional;
//! defaults are 5 seconds and `transport_dump.wav` in the current directory.

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use driver_transport::{SharedRingReader, MIC_OUTPUT_SECTION_NAME};
    use std::time::{Duration, Instant};

    let mut args = std::env::args().skip(1);
    let seconds: f32 = args
        .next()
        .map(|s| s.parse().context("first argument must be a number of seconds"))
        .transpose()?
        .unwrap_or(5.0);
    let out_path = args.next().unwrap_or_else(|| "transport_dump.wav".to_string());

    if seconds <= 0.0 {
        anyhow::bail!("seconds must be positive, got {seconds}");
    }

    println!("Opening shared section '{MIC_OUTPUT_SECTION_NAME}'...");
    println!("(Make sure clearnairt.exe is already running - it creates this section at startup.)");
    let reader = SharedRingReader::open(MIC_OUTPUT_SECTION_NAME)
        .context("failed to open the shared section - is clearnairt.exe running?")?;
    println!("Opened successfully. Capturing {seconds:.1}s of audio into '{out_path}'...");
    println!("(Talk into your mic now, in the ClearNAI-running window, so there's something to hear.)");

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: dsp_core::SAMPLE_RATE_HZ,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&out_path, spec)
        .with_context(|| format!("creating WAV file at {out_path}"))?;

    let mut frame = vec![0.0f32; dsp_core::FRAME_SAMPLES];
    let deadline = Instant::now() + Duration::from_secs_f32(seconds);
    let mut frames_captured: u64 = 0;
    let mut empty_polls: u64 = 0;

    while Instant::now() < deadline {
        if reader.pop_frame(&mut frame) {
            for &s in &frame {
                writer.write_sample(s).context("writing sample to WAV file")?;
            }
            frames_captured += 1;
            empty_polls = 0;
        } else {
            // The section is empty right now (writer hasn't produced a new
            // frame yet, or clearnairt isn't actually running) - a short
            // sleep here is fine, this is a plain diagnostic-tool loop, not
            // the realtime audio thread.
            std::thread::sleep(Duration::from_millis(2));
            empty_polls += 1;
            if empty_polls == 1000 {
                println!(
                    "WARNING: nothing has arrived in 2s of polling - is clearnairt.exe actually \
                     running and is its mic pipeline actually processing (not just idle)?"
                );
            }
        }
    }

    writer.finalize().context("finalizing WAV file")?;
    println!(
        "Done. Captured {frames_captured} frames ({:.2}s of real audio) into '{out_path}'. \
         Writer-side dropped_frames so far: {}.",
        frames_captured as f32 * dsp_core::FRAME_SAMPLES as f32 / dsp_core::SAMPLE_RATE_HZ as f32,
        reader.dropped_frames()
    );
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("transport-reader only does anything real on Windows (the shared section this tool reads is Windows-only shared memory).");
}
