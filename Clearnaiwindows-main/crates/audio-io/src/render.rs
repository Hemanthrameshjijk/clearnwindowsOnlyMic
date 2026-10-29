//! Real-time hardware render via WASAPI shared-mode, event-driven playback,
//! pulling fixed `dsp_core::FRAME_SAMPLES` frames from a
//! `dsp_core::Consumer<f32>` and writing them to the device.
//!
//! Same real-time-priority gap as `capture.rs`: the `wasapi` crate exposes
//! no MMCSS/thread-priority helper, so this thread runs at normal OS
//! scheduling priority. Not faked here.

#![cfg(windows)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use dsp_core::{Consumer, FrameCounters, FRAME_SAMPLES};
use wasapi::{initialize_mta, Device, DeviceEnumerator, Direction, StreamMode};

use crate::bytes::push_f32_le;
use crate::devices::{find_render_device_by_name, open_render_device_by_id};
use crate::{engine_wave_format, warn_if_buffer_size_mismatched, RECONNECT_BACKOFF, REQUESTED_BUFFER_DURATION_HNS};

// NOTE: same as capture.rs - `wasapi::Device` is not `Send` (raw COM
// pointer), so only the device id `String` crosses into the render thread;
// the `Device`/`IAudioClient` are (re)activated from inside that thread.

/// Owns the hardware-render OS thread. Dropping (or calling `stop`) signals
/// the thread to stop and joins it.
pub struct HardwareRender {
    stop_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl HardwareRender {
    /// Opens the system default render device and starts pulling frames
    /// from `consumer` to play out. `counters` receives live underrun
    /// telemetry (see `render_loop`). The default is looked up again on every
    /// reconnect, so if the device it first resolved to goes away (Krisp
    /// closed, headset unplugged) playback moves to whatever Windows now
    /// considers the default instead of retrying the dead one forever.
    pub fn start_default(consumer: Consumer<f32>, counters: Arc<FrameCounters>) -> Result<Self> {
        // Checked once here so "there is no default render device at all" is
        // still reported to the caller as an error.
        default_render_device()?;
        Self::spawn(RenderTarget::Default, consumer, counters)
    }

    /// Opens the render device whose friendly name contains `name_substr`
    /// (case-insensitive) - e.g. "CABLE Input" to render into VB-Cable
    /// instead of the user's real speakers.
    pub fn start_by_name(name_substr: &str, consumer: Consumer<f32>, counters: Arc<FrameCounters>) -> Result<Self> {
        let device = find_render_device_by_name(name_substr)?
            .with_context(|| format!("no render device matching '{name_substr}'"))?;
        let device_id = device.get_id().context("reading render device id")?;
        Self::start_on_device_id(&device_id, consumer, counters)
    }

    /// Opens a specific render device by its enumerated device id (see
    /// `devices::list_render_devices` / `devices::find_vb_cable_endpoints`).
    /// This is the entry point the virtual-mic pipeline uses to render
    /// specifically into "CABLE Input" rather than whatever the OS default
    /// happens to be.
    pub fn start_on_device_id(device_id: &str, consumer: Consumer<f32>, counters: Arc<FrameCounters>) -> Result<Self> {
        Self::spawn(RenderTarget::Id(device_id.to_string()), consumer, counters)
    }

    fn spawn(target: RenderTarget, consumer: Consumer<f32>, counters: Arc<FrameCounters>) -> Result<Self> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let thread_stop = stop_flag.clone();

        let handle = thread::Builder::new()
            .name("audio-io-hw-render".into())
            .spawn(move || {
                let mut consumer = consumer;
                // Best-effort: see `mmcss.rs` - failure is silent and
                // harmless (thread just stays at normal priority). Held for
                // the whole thread's lifetime, across reconnects, not
                // re-acquired per `render_loop` call below.
                let _mmcss_guard = crate::mmcss::elevate_current_thread();
                // Real hardware finding: a WASAPI failure here (device
                // disabled/removed, sleep/wake, another app grabbing
                // exclusive mode) is very often transient, but this loop
                // used to just exit on the first `render_loop` error -
                // permanently killing playback until the whole app was
                // restarted, while the pipeline thread on the other end of
                // `consumer` kept running and spun forever logging "output
                // ring buffer full" once nothing was left to drain it. Retry
                // with a backoff instead, so the device is transparently
                // reopened once it comes back (or once shared/exclusive
                // contention clears), matching the "always warm, self
                // healing" behavior the rest of this crate already has for
                // toggles.
                let mut failed_attempts: u32 = 0;
                while !thread_stop.load(Ordering::Relaxed) {
                    let mut streaming = false;
                    let result = render_loop(&target, &mut consumer, &thread_stop, &counters, failed_attempts, &mut streaming);
                    if streaming {
                        failed_attempts = 0;
                    }
                    if let Err(e) = result {
                        failed_attempts += 1;
                        // A device that stays unavailable (unplugged headset,
                        // stale id) fails identically on every attempt: say so
                        // on the first failure and then about once a minute,
                        // not every backoff for hours.
                        if failed_attempts == 1 || failed_attempts % 12 == 0 {
                            crate::report_error(&format!(
                                "[audio-io] hardware render thread error (attempt {failed_attempts}), retrying every {}s: {e:#}",
                                RECONNECT_BACKOFF.as_secs()
                            ));
                        }
                    }
                    if thread_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    // Quick retries first, then every 5 s for a device that is simply gone (unplugged headset).
                    let backoff = if failed_attempts <= 3 { RECONNECT_BACKOFF } else { RECONNECT_BACKOFF * 5 / 2 };
                    wait_discarding(backoff, &mut consumer, &thread_stop);
                }
            })
            .context("spawning hardware render thread")?;

        Ok(Self {
            stop_flag,
            handle: Some(handle),
        })
    }

    /// Signals the render thread to stop and blocks until it exits.
    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for HardwareRender {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// What a render thread opens on every (re)connect attempt.
enum RenderTarget {
    Id(String),
    /// Whatever Windows' default render device is at the moment of the
    /// attempt (not fixed when the thread was started).
    Default,
}

fn default_render_device() -> Result<Device> {
    DeviceEnumerator::new()
        .context("creating WASAPI device enumerator")?
        .get_default_device(&Direction::Render)
        .context("opening default render device")
}

/// Sleeps for `duration` (like `sleep_respecting_stop`) while emptying
/// `consumer` every few milliseconds. Used while the device is unavailable:
/// nothing is playing the audio, so (a) the producer side must never find the
/// ring full - it would drop and log "output ring buffer full" on every frame
/// for as long as the device stays away - and (b) what piled up during the
/// outage must not be played late once the device returns. A ring that filled
/// during an outage stays full afterwards (producer and consumer both run at
/// real time), i.e. up to the ring's 2 s of extra latency until the app is
/// restarted.
fn wait_discarding(duration: Duration, consumer: &mut Consumer<f32>, stop_flag: &Arc<AtomicBool>) {
    let slice = Duration::from_millis(10);
    let mut remaining = duration;
    while remaining > Duration::ZERO {
        if stop_flag.load(Ordering::Relaxed) {
            return;
        }
        while consumer.pop().is_ok() {}
        let step = remaining.min(slice);
        thread::sleep(step);
        remaining -= step;
    }
}

fn render_loop(
    target: &RenderTarget,
    consumer: &mut Consumer<f32>,
    stop_flag: &Arc<AtomicBool>,
    counters: &FrameCounters,
    failed_attempts: u32,
    streaming: &mut bool,
) -> Result<()> {
    let _ = initialize_mta();

    let device = match target {
        RenderTarget::Id(id) => open_render_device_by_id(id)
            .with_context(|| format!("re-opening render device '{id}' on render thread"))?,
        RenderTarget::Default => default_render_device()?,
    };
    let name = device.get_friendlyname().unwrap_or_else(|_| "unnamed device".to_string());
    let mut audio_client = device
        .get_iaudioclient()
        .with_context(|| format!("activating IAudioClient on render device '{name}'"))?;

    let format = engine_wave_format();

    // Explicitly request a 10ms (480-sample) buffer rather than trusting
    // whatever `get_device_period()`'s default/min period happens to be -
    // see `REQUESTED_BUFFER_DURATION_HNS` docs in `lib.rs` for why (the
    // reference PipeWire port silently got a 256ms buffer until it did the
    // analogous thing explicitly).
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: REQUESTED_BUFFER_DURATION_HNS,
    };

    audio_client
        .initialize_client(&format, &Direction::Render, &mode)
        .context("initializing render AudioClient (shared, event-driven)")?;

    // Speculative defense, unverified without real hardware: confirm WASAPI
    // actually honored the request above before we start streaming.
    warn_if_buffer_size_mismatched(&audio_client, "hardware render");

    let event_handle = audio_client
        .set_get_eventhandle()
        .context("getting render event handle")?;
    let render_client = audio_client
        .get_audiorenderclient()
        .context("getting IAudioRenderClient")?;

    // Pre-fill one buffer's worth of silence before starting the stream, as
    // WASAPI shared-mode render expects the buffer primed before the clock
    // starts ticking.
    let buffer_frames = audio_client
        .get_buffer_size()
        .context("querying render buffer size")?;
    {
        let mut silence: VecDeque<u8> = VecDeque::with_capacity(buffer_frames as usize * 4);
        for _ in 0..buffer_frames {
            push_f32_le(&mut silence, 0.0);
        }
        render_client
            .write_to_device_from_deque(buffer_frames as usize, &mut silence, None)
            .context("priming render buffer with silence")?;
    }

    audio_client
        .start_stream()
        .context("starting render stream")?;
    *streaming = true;
    crate::report_error(&if failed_attempts > 0 {
        format!("[audio-io] INFO: hardware render is playing on '{name}' after {failed_attempts} failed attempt(s)")
    } else {
        format!("[audio-io] INFO: hardware render started on '{name}'")
    });

    let mut pending: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES);
    let mut out_bytes: VecDeque<u8> = VecDeque::with_capacity(FRAME_SAMPLES * 4 * 4);

    // Underruns are expected whenever nothing is feeding this device (idle, or a source that has not started);
    // log the first one and then a summary at most every 30 s instead of a line per second.
    let mut last_underrun_log: Option<std::time::Instant> = None;
    let mut underruns_since_log: u64 = 0;
    while !stop_flag.load(Ordering::Relaxed) {
        if event_handle.wait_for_event(1000).is_err() {
            bail!("render event wait failed or timed out; render device likely became unavailable");
        }

        let available = audio_client
            .get_available_space_in_frames()
            .context("querying available render buffer space")?;
        if available == 0 {
            continue;
        }

        // Pull whole FRAME_SAMPLES-sized chunks from the ring buffer until
        // we have enough samples to fill the available device buffer space,
        // padding the tail with silence on underrun so WASAPI never sees a
        // short write.
        while pending.len() < available as usize {
            let start = pending.len();
            pending.resize(start + FRAME_SAMPLES, 0.0);
            let got = pull_frame(consumer, &mut pending[start..]);
            if got < FRAME_SAMPLES {
                counters.output_underruns.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                underruns_since_log += 1;
                let due = last_underrun_log.map_or(true, |t| t.elapsed() >= std::time::Duration::from_secs(30));
                if due {
                    crate::report_error(&format!(
                        "[audio-io] render underrun: {underruns_since_log} frame(s) padded with silence since the last note"
                    ));
                    last_underrun_log = Some(std::time::Instant::now());
                    underruns_since_log = 0;
                }
            }
        }

        out_bytes.clear();
        for &sample in pending.iter().take(available as usize) {
            push_f32_le(&mut out_bytes, sample);
        }
        render_client
            .write_to_device_from_deque(available as usize, &mut out_bytes, None)
            .context("writing render audio to device")?;

        pending.drain(0..available as usize);

        if stop_flag.load(Ordering::Relaxed) {
            break;
        }
    }

    audio_client.stop_stream().ok();
    Ok(())
}

/// Pulls up to `out.len()` samples from `consumer` into `out`, returning how
/// many real samples were copied. Any remaining tail of `out` is left as
/// whatever it already was (callers pre-zero it), i.e. silence on underrun.
fn pull_frame(consumer: &mut Consumer<f32>, out: &mut [f32]) -> usize {
    let mut n = 0;
    for slot in out.iter_mut() {
        match consumer.pop() {
            Ok(sample) => {
                *slot = sample;
                n += 1;
            }
            Err(_) => break,
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::RingBuffer;

    #[test]
    fn backoff_wait_empties_a_full_ring() {
        let (mut producer, mut consumer) = RingBuffer::<f32>::new(FRAME_SAMPLES * 4);
        while producer.push(0.5).is_ok() {}
        assert!(producer.push(0.5).is_err(), "ring starts full, like the monitor ring of a missing device");

        let stop = Arc::new(AtomicBool::new(false));
        wait_discarding(Duration::from_millis(40), &mut consumer, &stop);

        assert!(consumer.is_empty(), "stale audio is discarded");
        assert!(producer.push(0.5).is_ok(), "the producer can write again");
    }

    #[test]
    fn backoff_wait_returns_early_when_stopped() {
        let (_producer, mut consumer) = RingBuffer::<f32>::new(FRAME_SAMPLES);
        let stop = Arc::new(AtomicBool::new(true));
        let t0 = std::time::Instant::now();
        wait_discarding(Duration::from_secs(5), &mut consumer, &stop);
        assert!(t0.elapsed() < Duration::from_secs(1));
    }
}
