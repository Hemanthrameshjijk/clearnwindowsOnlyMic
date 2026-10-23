//! Bounded-time isolation for a `Stage` that might block for real.
//!
//! Real hardware finding: a mic capture ring buffer can stay permanently
//! full (never draining) for a minute-plus straight, with the WASAPI
//! capture thread itself never erroring - meaning the pipeline thread that
//! drains it stopped making progress entirely, not just fell behind. The
//! only stage in the chain that calls into code this crate does not
//! control (a closed-source DLL, `bvc-hush`'s `weya_nc.dll`) is the prime
//! suspect: a plain Rust deadlock inside our own stages would trip a
//! `panic = "abort"` release build's guard rails eventually (a poisoned
//! `Mutex` panics on `.unwrap()`), but a foreign call that itself never
//! returns leaves the thread parked forever with nothing to catch.
//!
//! `WatchdogStage` runs the wrapped stage on its own persistent worker
//! thread and enforces a timeout on each frame via a channel round trip.
//! On a timeout the frame passes through unprocessed for that call (a
//! missed noise-reduction pass is a minor, transient audio quality hit;
//! a wedged pipeline is silence forever). After enough consecutive
//! timeouts the wrapped stage is assumed permanently wedged and is
//! bypassed for the rest of the process's life rather than paying a
//! timeout's worth of latency on every single frame going forward.
//!
//! Real hardware finding: this worker thread runs the single most
//! CPU-expensive operation in the whole pipeline (BVC's ONNX inference) but,
//! unlike every WASAPI worker thread and both DSP pipeline threads, was
//! never given MMCSS real-time scheduling priority (`audio_io::mmcss`) -
//! under system CPU contention (other apps competing for the CPU), this
//! thread is the one most likely to get descheduled, which stalls the
//! calling pipeline thread waiting on it and cascades into ring-buffer
//! drops elsewhere. `on_worker_start` lets a caller inject that elevation
//! (or any other one-time worker-thread setup) without this portable,
//! `dsp-core` crate needing to depend on the Windows-only `audio-io` crate -
//! the returned value is kept alive for the worker thread's entire
//! lifetime and dropped (reverting elevation) only on thread exit, matching
//! `audio_io::mmcss::MmcssGuard`'s RAII contract.
use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::Stage;

/// Consecutive per-frame timeouts after which the wrapped stage is
/// considered permanently wedged and bypassed for good.
const MAX_CONSECUTIVE_TIMEOUTS: u32 = 3;

/// A cheap, cloneable, thread-safe handle onto a `WatchdogStage`'s bypass
/// state - obtained via `WatchdogStage::status()` *before* the stage is
/// boxed as `Box<dyn Stage>` and moved into the realtime pipeline, so a
/// caller on another thread (in practice, the GUI) can honestly show
/// whether a backend is actually still running or has been silently
/// auto-bypassed after repeated stalls.
///
/// Real problem this fixes: before this type existed, a `WatchdogStage`
/// that tripped `MAX_CONSECUTIVE_TIMEOUTS` would bypass itself permanently
/// with no way for anything outside this module to find out - the GUI's
/// "BVC: on" toggle stayed showing "on" even once the backend had stopped
/// actually running, making a real-world test of that backend impossible
/// to trust (passing audio through unprocessed looks identical, from the
/// toggle's perspective, to the backend genuinely running and not
/// suppressing anything).
#[derive(Clone)]
pub struct WatchdogStatus(Arc<AtomicBool>);

impl WatchdogStatus {
    /// True once the wrapped stage has been permanently bypassed (either
    /// `MAX_CONSECUTIVE_TIMEOUTS` stalls in a row, or its worker thread
    /// died/disconnected) - from that point on, `WatchdogStage::process`
    /// always leaves the frame untouched and never calls the wrapped stage
    /// again for the rest of the process's life.
    pub fn is_bypassed(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

pub struct WatchdogStage {
    name: &'static str,
    to_worker: Sender<Vec<f32>>,
    from_worker: Receiver<Vec<f32>>,
    timeout: Duration,
    bypassed: Arc<AtomicBool>,
    consecutive_timeouts: u32,
    log: Box<dyn Fn(&str) + Send>,
}

impl WatchdogStage {
    /// Wraps `inner` so every `process` call is bounded to `timeout`.
    /// `log` is called (off the realtime thread's critical path only when
    /// something has actually gone wrong) with a human-readable message
    /// whenever a timeout or permanent bypass occurs.
    pub fn new(
        name: &'static str,
        mut inner: Box<dyn Stage>,
        timeout: Duration,
        log: impl Fn(&str) + Send + 'static,
        on_worker_start: impl FnOnce() -> Box<dyn Any + Send> + Send + 'static,
    ) -> Self {
        // `to_worker` must never block the caller no matter how long the
        // worker is stuck inside `inner.process()` - a bounded channel's
        // buffer fills with unconsumed frames the moment the worker fails
        // to loop back to its own `recv()` even once, which turns the very
        // next `send` here into exactly the permanent block this type
        // exists to avoid. `MAX_CONSECUTIVE_TIMEOUTS` already caps how many
        // frames a wedged worker can ever be sent before this stage
        // bypasses itself, so unbounded growth here is not a real concern.
        let (to_worker, worker_rx) = channel::<Vec<f32>>();
        let (worker_tx, from_worker) = sync_channel::<Vec<f32>>(1);

        thread::Builder::new()
            .name(format!("dsp-watchdog-{name}"))
            .spawn(move || {
                // Held for the whole worker loop below; dropped (reverting
                // any elevation it represents) only when this thread exits.
                let _keepalive = on_worker_start();
                while let Ok(mut frame) = worker_rx.recv() {
                    inner.process(&mut frame);
                    if worker_tx.send(frame).is_err() {
                        break;
                    }
                }
            })
            .expect("spawning watchdog worker thread");

        Self {
            name,
            to_worker,
            from_worker,
            timeout,
            bypassed: Arc::new(AtomicBool::new(false)),
            consecutive_timeouts: 0,
            log: Box::new(log),
        }
    }

    /// A cheap, cloneable handle onto this stage's bypass state - call this
    /// before boxing the stage as `Box<dyn Stage>` and moving it into the
    /// realtime pipeline (once boxed, this concrete type's methods are no
    /// longer reachable), and hold the returned `WatchdogStatus` wherever
    /// something outside the pipeline (in practice, `EngineHandles`, read
    /// by the GUI) needs to honestly show whether this backend is still
    /// actually running.
    pub fn status(&self) -> WatchdogStatus {
        WatchdogStatus(self.bypassed.clone())
    }
}

impl Stage for WatchdogStage {
    fn name(&self) -> &'static str {
        self.name
    }

    fn process(&mut self, frame: &mut [f32]) {
        if self.bypassed.load(Ordering::Relaxed) {
            return;
        }

        if self.to_worker.send(frame.to_vec()).is_err() {
            self.bypassed.store(true, Ordering::Relaxed);
            (self.log)(&format!(
                "[dsp-core] {} worker thread is gone; bypassing it for the rest of this session",
                self.name
            ));
            return;
        }

        match self.from_worker.recv_timeout(self.timeout) {
            Ok(processed) => {
                frame.copy_from_slice(&processed);
                self.consecutive_timeouts = 0;
            }
            Err(RecvTimeoutError::Timeout) => {
                self.consecutive_timeouts += 1;
                (self.log)(&format!(
                    "[dsp-core] {} did not respond within {:?} (consecutive stalls: {}); \
                     passing this frame through unprocessed",
                    self.name, self.timeout, self.consecutive_timeouts
                ));
                if self.consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS {
                    self.bypassed.store(true, Ordering::Relaxed);
                    (self.log)(&format!(
                        "[dsp-core] {} appears permanently stalled after {} consecutive \
                         timeouts; bypassing it for the rest of this session",
                        self.name, self.consecutive_timeouts
                    ));
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.bypassed.store(true, Ordering::Relaxed);
                (self.log)(&format!(
                    "[dsp-core] {} worker thread disconnected; bypassing it for the rest of this session",
                    self.name
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct Slow(Duration);
    impl Stage for Slow {
        fn name(&self) -> &'static str {
            "slow"
        }
        fn process(&mut self, frame: &mut [f32]) {
            thread::sleep(self.0);
            for s in frame.iter_mut() {
                *s += 1.0;
            }
        }
    }

    struct HangsForever;
    impl Stage for HangsForever {
        fn name(&self) -> &'static str {
            "hangs-forever"
        }
        fn process(&mut self, _frame: &mut [f32]) {
            loop {
                thread::sleep(Duration::from_secs(3600));
            }
        }
    }

    #[test]
    fn fast_stage_processes_normally() {
        let mut stage = WatchdogStage::new(
            "fast",
            Box::new(Slow(Duration::from_millis(1))),
            Duration::from_millis(200),
            |_| {},
            || Box::new(()),
        );
        let mut frame = [1.0f32, 2.0, 3.0];
        stage.process(&mut frame);
        assert_eq!(frame, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn hung_stage_passes_frame_through_unprocessed_on_timeout() {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let logs_clone = logs.clone();
        let mut stage = WatchdogStage::new(
            "hangs-forever",
            Box::new(HangsForever),
            Duration::from_millis(50),
            move |msg: &str| logs_clone.lock().unwrap().push(msg.to_string()),
            || Box::new(()),
        );
        let mut frame = [5.0f32, 6.0];
        stage.process(&mut frame);
        assert_eq!(frame, [5.0, 6.0], "unprocessed frame must pass through untouched");
        assert!(!logs.lock().unwrap().is_empty(), "a timeout must be logged");
    }

    #[test]
    fn permanently_wedged_stage_is_bypassed_after_max_consecutive_timeouts() {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let logs_clone = logs.clone();
        let mut stage = WatchdogStage::new(
            "hangs-forever",
            Box::new(HangsForever),
            Duration::from_millis(20),
            move |msg: &str| logs_clone.lock().unwrap().push(msg.to_string()),
            || Box::new(()),
        );
        for _ in 0..MAX_CONSECUTIVE_TIMEOUTS {
            let mut frame = [0.0f32; 4];
            stage.process(&mut frame);
        }
        assert!(stage.bypassed.load(Ordering::Relaxed), "stage must be marked permanently bypassed");
        let log_count_before = logs.lock().unwrap().len();

        // Further calls must be free (no channel round trip / timeout wait)
        // once bypassed - prove that by timing a bunch of them.
        let started = std::time::Instant::now();
        for _ in 0..1000 {
            let mut frame = [0.0f32; 4];
            stage.process(&mut frame);
        }
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "bypassed calls must not pay the timeout cost"
        );
        assert_eq!(
            logs.lock().unwrap().len(),
            log_count_before,
            "no further logging once already bypassed"
        );
    }

    /// Exercises the real usage pattern this type exists for: take a
    /// `WatchdogStatus` handle right after construction (before the stage
    /// is boxed and moved into the pipeline, exactly like `engine.rs` does
    /// for `EngineHandles`), then confirm it honestly reflects the stage's
    /// bypass transition as observed through a completely independent
    /// clone - not just the field the stage itself holds.
    #[test]
    fn watchdog_status_reflects_bypass_from_an_independent_handle() {
        let mut stage = WatchdogStage::new(
            "hangs-forever",
            Box::new(HangsForever),
            Duration::from_millis(20),
            |_| {},
            || Box::new(()),
        );
        let status = stage.status();
        assert!(!status.is_bypassed(), "must not report bypassed before anything has gone wrong");

        for _ in 0..MAX_CONSECUTIVE_TIMEOUTS {
            let mut frame = [0.0f32; 4];
            stage.process(&mut frame);
        }

        assert!(
            status.is_bypassed(),
            "a handle taken before the stage was ever processed must still observe the later bypass"
        );
        // A second, independently cloned handle must agree too.
        assert!(status.clone().is_bypassed());
    }
}
