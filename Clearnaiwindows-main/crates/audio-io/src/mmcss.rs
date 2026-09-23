//! MMCSS ("Multimedia Class Scheduler Service") thread elevation.
//!
//! Real hardware finding: with capture/render/pipeline threads all running
//! at plain `THREAD_PRIORITY_NORMAL`, real logs show sustained, chronic
//! `mic capture ring buffer full`/`render ring buffer underrun` storms - not
//! just an occasional blip, but every frame, for minutes at a time. That is
//! the textbook symptom of a realtime audio thread not being scheduled
//! promptly enough under system load, which on Windows is exactly what
//! MMCSS's "Pro Audio" task category exists to fix: it asks the scheduler to
//! treat this thread the way a DAW's or a voice-chat app's audio thread is
//! treated (short, guaranteed scheduling quanta, boosted above normal
//! background work) instead of competing as an ordinary thread. This module
//! was the "real-time-priority gap" already called out in `capture.rs`'s and
//! `render.rs`'s doc comments as a known, not-yet-implemented gap.
//!
//! Every WASAPI worker thread (mic capture, hardware render, loopback tap)
//! and both DSP pipeline threads (`app::engine::spawn_mic_pipeline`/
//! `spawn_speaker_pipeline`) should call `elevate_current_thread` once, right
//! after the thread starts, and hold onto the returned guard for the
//! thread's entire lifetime (dropping it reverts the elevation, which should
//! only happen at thread exit).

#![cfg(windows)]

use windows::core::PCWSTR;
use windows::Win32::System::Threading::{
    AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW, GetCurrentThread, SetThreadPriority,
    THREAD_PRIORITY_BELOW_NORMAL,
};

/// RAII guard for one thread's MMCSS elevation. Reverts automatically when
/// dropped (i.e. hold this for as long as the thread should stay elevated -
/// typically its entire lifetime).
pub struct MmcssGuard(windows::Win32::Foundation::HANDLE);

// SAFETY: `HANDLE` wraps a raw pointer (`*mut c_void`), which is not `Send`
// by default, but this specific handle is just an opaque identifier the
// Avrt API hands back for `AvRevertMmThreadCharacteristics` to consume - it
// is not documented as thread-affine (unlike the elevation effect itself,
// which does apply to whichever OS thread originally called
// `AvSetMmThreadCharacteristicsW`). Moving the guard to another thread and
// dropping it there still correctly reverts the *original* thread's
// elevation; it just means the `Drop` call itself runs on a different
// thread than the one that got elevated - fine, since `AvRevertMmThreadCharacteristics`
// takes the handle as its only real input, not "the calling thread".
unsafe impl Send for MmcssGuard {}

impl Drop for MmcssGuard {
    fn drop(&mut self) {
        // SAFETY: `self.0` was returned by a successful
        // `AvSetMmThreadCharacteristicsW` call and has not been reverted yet
        // (this is the only place that reverts it, and it runs at most once
        // per guard).
        unsafe {
            let _ = AvRevertMmThreadCharacteristics(self.0);
        }
    }
}

/// Elevates the calling thread to the MMCSS "Pro Audio" task category.
/// Best-effort: real hardware finding is that this project cannot assume
/// every target machine grants this (group policy, older Windows builds,
/// a locked-down environment), so a failure here must never be fatal - the
/// thread simply keeps running at normal priority, same as before this
/// module existed. Returns `None` on failure; callers that get `None` are
/// not expected to log it as an error (see call sites), only to proceed.
pub fn elevate_current_thread() -> Option<MmcssGuard> {
    let mut task_index: u32 = 0;
    // "Pro Audio" is the standard MMCSS task name for exclusive, low-latency
    // realtime audio processing (the same category real-world DAWs and
    // voice-chat apps register their audio threads under). Must be a
    // null-terminated UTF-16 string for the `PCWSTR` FFI boundary.
    let name: Vec<u16> = "Pro Audio\0".encode_utf16().collect();
    // SAFETY: `name` is a valid null-terminated UTF-16 buffer kept alive for
    // the duration of this call; `task_index` is a valid `&mut u32` for the
    // out-parameter.
    let handle = unsafe { AvSetMmThreadCharacteristicsW(PCWSTR(name.as_ptr()), &mut task_index) };
    match handle {
        Ok(h) if !h.is_invalid() => Some(MmcssGuard(h)),
        _ => None,
    }
}

/// Opposite of `elevate_current_thread`: drops the calling thread's OS
/// scheduling priority below normal, so it never competes with the
/// realtime-elevated audio threads under CPU contention. Real hardware
/// finding: a CPU-heavy background task (the GUI's offline "Test with TSE"
/// button, which runs the full TSE model across every core for up to a
/// minute with no coordination with the live pipeline) measurably degraded
/// live mic audio - added latency, frame drops - purely from competing for
/// CPU time with the already-elevated realtime threads. No RAII guard here
/// (unlike `elevate_current_thread`): callers of this function are one-shot
/// background worker threads that exit when their work is done, so there is
/// nothing to "revert" - the thread (and its lowered priority) simply goes
/// away. Best-effort like its counterpart: failure is silently ignored, the
/// thread just runs at normal priority instead, never fatal.
pub fn lower_current_thread_priority() {
    // SAFETY: `GetCurrentThread` returns a pseudo-handle valid for the
    // calling thread's entire lifetime, requiring no cleanup; `SetThreadPriority`
    // with it is a documented, safe-to-call-anytime Win32 operation.
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}
