//! `EnrollmentSession`: owns the non-realtime side of one-time target-
//! speaker enrollment - starting/stopping a recording, running VAD/quality
//! validation, calling the speaker encoder, and publishing the result to a
//! `speaker_profile::TargetSpeakerStore`. Everything in this file runs on a
//! plain background thread spawned per recording attempt (the same
//! "spawn a thread for this rare, user-initiated, potentially-slow action"
//! idiom `app::gui::spawn_device_switch` already uses for live device
//! switches) - never on the realtime mic pipeline thread.

use crate::error::EnrollmentError;
use crate::metrics::EnrollmentMetrics;
use crate::state::EnrollmentState;
use crate::validate::{validate, ValidationThresholds};
use dsp_core::{Consumer, Stage, StageToggle};
use speaker_profile::{SpeakerEncoder, TargetSpeakerStore};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Hard ceiling on one recording's length, in seconds - see
/// `tap::new_pair`'s docs: this is also the ring buffer's capacity, so it is
/// simultaneously "maximum enrollment duration" and "no unbounded memory
/// growth," in one number.
pub const MAX_RECORDING_SECS: f32 = 12.0;

/// How long a stretch of trailing near-silence, after some speech has
/// already been seen, ends the recording automatically - lets the user
/// finish the reference sentence and stop talking rather than requiring a
/// manual stop click every time, while `MAX_RECORDING_SECS` and the manual
/// "stop" button remain as the two explicit escape hatches the task asks
/// for ("if automatic stop is unreliable, provide a clear timeout/max
/// duration and allow the user to stop recording manually").
const AUTO_STOP_TRAILING_SILENCE_SECS: f32 = 1.5;

/// Per-sample energy threshold used only for the simple auto-stop heuristic
/// above - deliberately the same order of magnitude as
/// `ValidationThresholds::speech_energy_threshold`'s per-frame RMS
/// threshold, but this one is checked per-sample as recording proceeds
/// (streaming), not per-frame after the fact.
const AUTO_STOP_SILENCE_SAMPLE_THRESHOLD: f32 = 0.02;

/// Minimum samples that must be captured before the silence-based auto-stop
/// is allowed to fire at all - otherwise a slow start (silence while the
/// user takes a breath before speaking) could auto-stop the recording
/// before they've said anything.
fn min_samples_before_auto_stop(sample_rate_hz: u32) -> usize {
    (sample_rate_hz as f32 * 1.0) as usize
}

/// Owns the worker-thread half of the enrollment tap (`Consumer<f32>`), the
/// shared recording-active toggle, the current `EnrollmentState`, the
/// speaker encoder backend, and the target-speaker store to publish into.
/// One `EnrollmentSession` is constructed once at engine startup, alongside
/// every other stage (see `tap::new_pair`), and is cheap to clone (every
/// field is an `Arc`-backed handle) so the GUI can hold its own clone
/// exactly like it holds `EngineHandles`.
#[derive(Clone)]
pub struct EnrollmentSession {
    consumer: Arc<Mutex<Consumer<f32>>>,
    active: StageToggle,
    state: Arc<Mutex<EnrollmentState>>,
    encoder: Arc<dyn SpeakerEncoder>,
    store: TargetSpeakerStore,
    /// Extra (encoder, store) pairs, each run on the SAME captured
    /// recording as `encoder`/`store` above - lets additional, differently-
    /// shaped embeddings (e.g. a TSE model's own 192-dim conditioning
    /// embedding, or Vanta's own separate 192-dim ECAPA space - different
    /// checkpoints, not interchangeable, see `bvc_vanta::ecapa`'s docs) be
    /// produced from one enrollment attempt without recording multiple
    /// times. Empty for callers that only need one embedding (every
    /// existing call site, unchanged).
    secondary: Vec<(Arc<dyn SpeakerEncoder>, TargetSpeakerStore)>,
    /// The raw samples from the most recent recording attempt (kept
    /// regardless of whether validation/embedding succeeded, so a rejected
    /// or bad-sounding attempt can still be played back for the user to
    /// hear what was actually captured - e.g. to tell "too quiet" apart
    /// from "wrong device"). In memory only, never written to disk -
    /// consistent with this project's existing policy of never persisting
    /// raw reference audio (see `docs/TSE_ENROLLMENT.md`). Replaced on the
    /// next recording attempt and cleared by `clear_profile`.
    last_recording: Arc<Mutex<Option<(Vec<f32>, u32)>>>,
    metrics: Arc<EnrollmentMetrics>,
    max_recording_secs: f32,
}

impl EnrollmentSession {
    pub fn new(
        consumer: Consumer<f32>,
        active: StageToggle,
        encoder: Arc<dyn SpeakerEncoder>,
        store: TargetSpeakerStore,
    ) -> Self {
        Self {
            consumer: Arc::new(Mutex::new(consumer)),
            active,
            state: Arc::new(Mutex::new(EnrollmentState::Idle)),
            encoder,
            store,
            secondary: Vec::new(),
            last_recording: Arc::new(Mutex::new(None)),
            metrics: Arc::new(EnrollmentMetrics::default()),
            max_recording_secs: MAX_RECORDING_SECS,
        }
    }

    /// The most recent recording attempt's raw samples and sample rate, if
    /// any - see `last_recording`'s docs. Cloned out (not a reference) so
    /// the caller can hand it straight to a playback routine without
    /// holding this session's lock.
    pub fn last_recording(&self) -> Option<(Vec<f32>, u32)> {
        self.last_recording.lock().unwrap().clone()
    }

    /// Cheap presence check (no clone of the audio buffer) - for the GUI to
    /// decide whether to show a "Play Recording" button.
    pub fn has_last_recording(&self) -> bool {
        self.last_recording.lock().unwrap().is_some()
    }

    /// The most recent recording, run through the same RNNoise denoising
    /// step (`denoise_for_embedding`) real enrollment already applies
    /// before extracting a voice embedding - lets the user actually HEAR
    /// what the encoder sees, for direct A/B comparison against
    /// `last_recording`'s raw version, rather than trusting it blindly.
    /// Recomputed fresh each call (RNNoise's own state is freshly
    /// constructed per call already - see `denoise_for_embedding`'s docs -
    /// so this is deterministic, just not cached).
    pub fn last_recording_denoised(&self) -> Option<(Vec<f32>, u32)> {
        let (raw, sample_rate_hz) = self.last_recording()?;
        Some((denoise_for_embedding(&raw), sample_rate_hz))
    }

    /// Adds another (encoder, store) pair, run on the same recorded audio
    /// as the primary one - see `secondary`'s docs. Builder-style, and
    /// chainable (call multiple times for multiple extra encoders), so
    /// every existing single-call call site (and test) is unaffected.
    pub fn with_secondary_encoder(mut self, encoder: Arc<dyn SpeakerEncoder>, store: TargetSpeakerStore) -> Self {
        self.secondary.push((encoder, store));
        self
    }

    pub fn state(&self) -> EnrollmentState {
        self.state.lock().unwrap().clone()
    }

    pub fn metrics(&self) -> Arc<EnrollmentMetrics> {
        self.metrics.clone()
    }

    pub fn has_target_speaker(&self) -> bool {
        self.store.has_profile()
    }

    /// Starts a new recording attempt on a background thread. A no-op if a
    /// recording is already in progress (`EnrollmentState::is_in_progress`)
    /// - guards against a double-click spawning two worker threads racing
    /// over the same `Consumer`.
    pub fn start_recording(&self, sample_rate_hz: u32) {
        {
            let mut state = self.state.lock().unwrap();
            if state.is_in_progress() {
                return;
            }
            *state = EnrollmentState::Preparing;
        }

        let consumer = self.consumer.clone();
        let active = self.active.clone();
        let state = self.state.clone();
        let encoder = self.encoder.clone();
        let store = self.store.clone();
        let secondary = self.secondary.clone();
        let last_recording = self.last_recording.clone();
        let metrics = self.metrics.clone();
        let max_samples = (self.max_recording_secs * sample_rate_hz as f32) as usize;

        thread::spawn(move || {
            run_recording_attempt(
                consumer,
                active,
                state,
                encoder,
                store,
                secondary,
                last_recording,
                metrics,
                sample_rate_hz,
                max_samples,
            );
        });
    }

    /// Requests the in-progress recording stop early. The worker thread
    /// notices on its next poll and moves on to validation with whatever it
    /// has already captured - this is the "user-visible manual stop"
    /// escape hatch alongside the automatic silence-based stop and the hard
    /// `MAX_RECORDING_SECS` ceiling.
    pub fn stop_recording(&self) {
        self.active.set(false);
    }

    /// Clears any enrolled target speaker and resets to `Idle`. Does not
    /// touch an in-progress recording (if one is somehow running -
    /// `start_recording`'s guard makes this unlikely in practice).
    pub fn clear_profile(&self) {
        self.store.clear_profile();
        *self.last_recording.lock().unwrap() = None;
        *self.state.lock().unwrap() = EnrollmentState::Idle;
    }
}

#[allow(clippy::too_many_arguments)]
fn run_recording_attempt(
    consumer: Arc<Mutex<Consumer<f32>>>,
    active: StageToggle,
    state: Arc<Mutex<EnrollmentState>>,
    encoder: Arc<dyn SpeakerEncoder>,
    store: TargetSpeakerStore,
    secondary: Vec<(Arc<dyn SpeakerEncoder>, TargetSpeakerStore)>,
    last_recording: Arc<Mutex<Option<(Vec<f32>, u32)>>>,
    metrics: Arc<EnrollmentMetrics>,
    sample_rate_hz: u32,
    max_samples: usize,
) {
    metrics.attempts.fetch_add(1, Ordering::Relaxed);

    // Drain any stale samples left over from a previous attempt (e.g. a few
    // samples pushed between the previous recording's `active.set(false)`
    // and this thread ever locking the consumer) before arming - a fresh
    // recording must start from silence, not leftover audio.
    {
        let mut guard = consumer.lock().unwrap();
        while guard.pop().is_ok() {}
    }

    active.set(true);
    *state.lock().unwrap() = EnrollmentState::Recording;

    let recording_started = Instant::now();
    let mut buffer: Vec<f32> = Vec::with_capacity(max_samples.min(sample_rate_hz as usize * 16));
    let min_before_auto_stop = min_samples_before_auto_stop(sample_rate_hz);
    let silence_samples_needed = (sample_rate_hz as f32 * AUTO_STOP_TRAILING_SILENCE_SECS) as usize;
    let mut consecutive_silent_samples = 0usize;
    let mut has_seen_speech = false;

    {
        let mut guard = consumer.lock().unwrap();
        loop {
            if buffer.len() >= max_samples {
                break;
            }
            match guard.pop() {
                Ok(sample) => {
                    buffer.push(sample);
                    if sample.abs() > AUTO_STOP_SILENCE_SAMPLE_THRESHOLD {
                        has_seen_speech = true;
                        consecutive_silent_samples = 0;
                    } else {
                        consecutive_silent_samples += 1;
                    }
                    if has_seen_speech
                        && buffer.len() >= min_before_auto_stop
                        && consecutive_silent_samples >= silence_samples_needed
                    {
                        break;
                    }
                }
                Err(_) => {
                    if !active.is_on() {
                        // Manual stop was requested and the tap has nothing
                        // further buffered - end the recording now with
                        // whatever was captured.
                        break;
                    }
                    thread::sleep(Duration::from_millis(5));
                }
            }
        }
    }
    active.set(false);
    metrics
        .last_recording_duration_ms
        .store(recording_started.elapsed().as_millis() as u64, Ordering::Relaxed);

    // Store the raw capture for playback regardless of what validation/
    // encoding below decide - hearing a rejected or bad-sounding attempt is
    // exactly what helps tell "too quiet" apart from "wrong device" (see
    // `last_recording`'s docs). Only real captured samples are stored; an
    // empty buffer (see the check just below) leaves the previous
    // recording, if any, in place rather than clearing it to "nothing."
    if !buffer.is_empty() {
        *last_recording.lock().unwrap() = Some((buffer.clone(), sample_rate_hz));
    }

    *state.lock().unwrap() = EnrollmentState::Validating;
    if buffer.is_empty() {
        finish_with_error(&state, &metrics, EnrollmentError::RecordingError(
            "no audio was captured (the physical microphone tap never produced a sample)".to_string(),
        ));
        return;
    }
    if let Err(e) = validate(&buffer, sample_rate_hz, &ValidationThresholds::default()) {
        finish_with_error(&state, &metrics, e);
        return;
    }

    // Validation above ran on the RAW capture (so its checks, including
    // the new background-noise/SNR one, reflect what the microphone
    // actually heard, not an already-cleaned-up version - denoising here
    // would mask exactly what that check is meant to catch). Embedding
    // extraction, below, uses a denoised COPY instead - `buffer` itself
    // (and therefore `last_recording`, already stored above) stays the
    // real, unmodified capture for playback. See `denoise_for_embedding`'s
    // docs for what RNNoise does and does not remove (steady background
    // noise: yes; a second human voice: no).
    let embedding_input = denoise_for_embedding(&buffer);

    *state.lock().unwrap() = EnrollmentState::ExtractingEmbedding;
    let embed_started = Instant::now();
    let result = encoder.embed(&embedding_input, sample_rate_hz);
    metrics
        .last_embedding_extraction_ms
        .store(embed_started.elapsed().as_millis() as u64, Ordering::Relaxed);

    match result {
        Ok(profile) => {
            store.set_profile(profile);
            // Best-effort: run the same (denoised) buffer through each
            // extra encoder (e.g. a TSE model's own conditioning
            // embedding) that was configured - a failure here does not
            // fail the primary enrollment (which already succeeded and is
            // what the existing GUI/tests observe), it just leaves that
            // one secondary store unset, matching this project's
            // "degrade the optional piece honestly, don't crash the whole
            // feature" convention.
            for (secondary_encoder, secondary_store) in &secondary {
                match secondary_encoder.embed(&embedding_input, sample_rate_hz) {
                    Ok(secondary_profile) => secondary_store.set_profile(secondary_profile),
                    Err(_) => secondary_store.clear_profile(),
                }
            }
            metrics.successes.fetch_add(1, Ordering::Relaxed);
            *state.lock().unwrap() = EnrollmentState::Ready;
        }
        Err(e) => finish_with_error(&state, &metrics, e.into()),
    }
}

/// Runs `audio` (48kHz, matching `dsp_core::SAMPLE_RATE_HZ` - this is the
/// rate `start_recording` always captures at) through RNNoise
/// (`noise_engine::RnnoiseEngine`, the same real, already-tested engine
/// the live mic pipeline uses) once, for embedding-extraction purposes
/// only - the raw `buffer`/`last_recording` used for playback and
/// validation is never touched. Returns a same-length denoised copy.
///
/// Honest limitation: RNNoise suppresses steady background noise (hum,
/// fans, traffic, a droning TV) - it does NOT remove a second human
/// voice, since that still looks like real speech to it. This step
/// reduces noise contamination in the embedding; it does not solve "two
/// people were talking."
///
/// If `RnnoiseEngine` fails to construct (should not happen in practice -
/// see its own docs on the one condition that could cause it, a frame-size
/// mismatch that's actually a compile-time-fixed constant comparison) this
/// falls back to the original, undenoised audio rather than failing
/// enrollment over an optional cleanup step.
fn denoise_for_embedding(audio: &[f32]) -> Vec<f32> {
    let mut engine = match noise_engine::RnnoiseEngine::new() {
        Ok(e) => e,
        Err(_) => return audio.to_vec(),
    };

    let frame_len = dsp_core::FRAME_SAMPLES;
    let mut out = Vec::with_capacity(audio.len());
    let mut chunk_buf = vec![0.0f32; frame_len];
    for chunk in audio.chunks(frame_len) {
        let real_len = chunk.len();
        chunk_buf[..real_len].copy_from_slice(chunk);
        // Zero-pad a trailing partial frame - RnnoiseEngine requires
        // exactly FRAME_SAMPLES per call (`debug_assert_eq!` in its own
        // `process`). Only the first `real_len` samples of the result are
        // kept below, so the padding never leaks into the output.
        if real_len < frame_len {
            chunk_buf[real_len..].iter_mut().for_each(|v| *v = 0.0);
        }
        engine.process(&mut chunk_buf);
        out.extend_from_slice(&chunk_buf[..real_len]);
    }
    out
}

fn finish_with_error(state: &Arc<Mutex<EnrollmentState>>, metrics: &Arc<EnrollmentMetrics>, error: EnrollmentError) {
    metrics.failures.fetch_add(1, Ordering::Relaxed);
    *state.lock().unwrap() = EnrollmentState::Failed(error);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tap::new_pair;
    use speaker_profile::{SpeakerEncoderError, TargetSpeakerProfile};
    use std::time::Duration as StdDuration;

    #[test]
    fn denoise_for_embedding_preserves_length_for_arbitrary_sizes() {
        // Not a multiple of FRAME_SAMPLES (480) - exercises the zero-padded
        // trailing-partial-frame path.
        for len in [0, 1, 100, 480, 481, 999, 48_000 * 3 + 7] {
            let audio = vec![0.1f32; len];
            let out = denoise_for_embedding(&audio);
            assert_eq!(out.len(), len, "length mismatch for input len {len}");
        }
    }

    #[test]
    fn denoise_for_embedding_output_is_always_finite() {
        let mut audio = vec![0.0f32; 48_000 * 2];
        for (i, s) in audio.iter_mut().enumerate() {
            *s = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48_000.0).sin();
        }
        let out = denoise_for_embedding(&audio);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn denoise_for_embedding_actually_changes_a_noisy_signal() {
        // A real 220Hz tone plus a steady higher-frequency "noise" tone -
        // RNNoise should attenuate/reshape this relative to the raw input,
        // proving this step is real processing, not a silent passthrough.
        let n = 48_000 * 2;
        let mut audio = vec![0.0f32; n];
        for (i, s) in audio.iter_mut().enumerate() {
            let t = i as f32 / 48_000.0;
            *s = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 4000.0 * t).sin();
        }
        let out = denoise_for_embedding(&audio);
        assert_eq!(out.len(), audio.len());
        let max_abs_diff = out.iter().zip(&audio).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs_diff > 1e-4, "expected real denoising to change the signal, max_abs_diff={max_abs_diff}");
    }

    struct FixedEncoder {
        result: Result<Vec<f32>, SpeakerEncoderError>,
    }
    impl SpeakerEncoder for FixedEncoder {
        fn encoder_id(&self) -> &'static str {
            "fixed-test-encoder"
        }
        fn embedding_dim(&self) -> Option<usize> {
            Some(3)
        }
        fn expected_sample_rate_hz(&self) -> u32 {
            48_000
        }
        fn embed(&self, _audio: &[f32], _input_sample_rate_hz: u32) -> Result<TargetSpeakerProfile, SpeakerEncoderError> {
            self.result.clone().map(|v| {
                TargetSpeakerProfile::new(v, self.encoder_id(), self.expected_sample_rate_hz(), speaker_profile::NormalizationKind::None)
                    .unwrap()
            })
        }
    }

    fn wait_until_idle_or_terminal(session: &EnrollmentSession, timeout: StdDuration) -> EnrollmentState {
        let started = Instant::now();
        loop {
            let s = session.state();
            if !s.is_in_progress() {
                return s;
            }
            if started.elapsed() > timeout {
                return s;
            }
            thread::sleep(StdDuration::from_millis(10));
        }
    }

    #[test]
    fn successful_enrollment_reaches_ready_and_publishes_a_profile() {
        let (mut tap_producer_side, consumer, active, _dropped) = new_pair(2.0);
        let store = TargetSpeakerStore::new();
        let encoder: Arc<dyn SpeakerEncoder> = Arc::new(FixedEncoder { result: Ok(vec![1.0, 2.0, 3.0]) });
        let session = EnrollmentSession::new(consumer, active.clone(), encoder, store.clone());

        session.start_recording(48_000);
        // Give the worker thread a moment to arm (Preparing -> Recording)
        // before pushing samples through the tap.
        thread::sleep(StdDuration::from_millis(20));
        let frame = {
            let mut f = Vec::new();
            for i in 0..(48_000 * 3) {
                f.push(0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48_000.0).sin());
            }
            f
        };
        tap_producer_side.push_frame_if_active(&frame);
        // A continuous tone never satisfies the silence-based auto-stop (it
        // never holds below the per-sample threshold for
        // `AUTO_STOP_TRAILING_SILENCE_SECS` straight), so this test - unlike
        // real speech, which does have pauses - relies on a manual stop
        // here, exactly like a real "user clicks stop" flow.
        session.stop_recording();

        let final_state = wait_until_idle_or_terminal(&session, StdDuration::from_secs(5));
        assert_eq!(final_state, EnrollmentState::Ready, "expected Ready, got {final_state:?}");
        assert!(store.has_profile());
        assert_eq!(store.get_profile().unwrap().embedding(), &[1.0, 2.0, 3.0]);
        assert_eq!(session.metrics().successes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn silence_only_recording_fails_validation_not_the_encoder() {
        let (mut tap_producer_side, consumer, active, _dropped) = new_pair(2.0);
        let store = TargetSpeakerStore::new();
        let encoder: Arc<dyn SpeakerEncoder> = Arc::new(FixedEncoder { result: Ok(vec![1.0]) });
        let session = EnrollmentSession::new(consumer, active.clone(), encoder, store.clone());

        session.start_recording(48_000);
        thread::sleep(StdDuration::from_millis(20));
        let silence = vec![0.0f32; 48_000 * 3];
        tap_producer_side.push_frame_if_active(&silence);
        // Manual stop: silence never triggers the speech-seen auto-stop.
        session.stop_recording();

        let final_state = wait_until_idle_or_terminal(&session, StdDuration::from_secs(5));
        match final_state {
            EnrollmentState::Failed(EnrollmentError::TooQuiet) | EnrollmentState::Failed(EnrollmentError::NoSpeech) => {}
            other => panic!("expected a quiet/no-speech failure, got {other:?}"),
        }
        assert!(!store.has_profile(), "a rejected recording must never publish a profile");
    }

    #[test]
    fn model_unavailable_is_reported_not_faked() {
        let (mut tap_producer_side, consumer, active, _dropped) = new_pair(2.0);
        let store = TargetSpeakerStore::new();
        let encoder: Arc<dyn SpeakerEncoder> = Arc::new(FixedEncoder { result: Err(SpeakerEncoderError::ModelUnavailable) });
        let session = EnrollmentSession::new(consumer, active.clone(), encoder, store.clone());

        session.start_recording(48_000);
        thread::sleep(StdDuration::from_millis(20));
        let mut loud = Vec::new();
        for i in 0..(48_000 * 3) {
            loud.push(0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48_000.0).sin());
        }
        tap_producer_side.push_frame_if_active(&loud);
        session.stop_recording();

        let final_state = wait_until_idle_or_terminal(&session, StdDuration::from_secs(5));
        assert_eq!(final_state, EnrollmentState::Failed(EnrollmentError::ModelUnavailable));
        assert!(!store.has_profile(), "must never fabricate a profile when the encoder is unavailable");
    }

    #[test]
    fn double_start_is_ignored_while_one_attempt_is_in_progress() {
        let (_tap, consumer, active, _dropped) = new_pair(2.0);
        let store = TargetSpeakerStore::new();
        let encoder: Arc<dyn SpeakerEncoder> = Arc::new(FixedEncoder { result: Ok(vec![1.0]) });
        let session = EnrollmentSession::new(consumer, active, encoder, store);

        session.start_recording(48_000);
        thread::sleep(StdDuration::from_millis(20));
        assert!(session.state().is_in_progress());
        // Second call must not panic or spawn a competing thread over the
        // same Consumer (which would panic rtrb, since it's single-consumer).
        session.start_recording(48_000);
        session.stop_recording();
        let _ = wait_until_idle_or_terminal(&session, StdDuration::from_secs(5));
    }

    #[test]
    fn clear_profile_resets_to_idle_and_removes_the_profile() {
        let (_tap, consumer, active, _dropped) = new_pair(2.0);
        let store = TargetSpeakerStore::new();
        store.set_profile(TargetSpeakerProfile::new(vec![1.0], "e", 48_000, speaker_profile::NormalizationKind::None).unwrap());
        let encoder: Arc<dyn SpeakerEncoder> = Arc::new(FixedEncoder { result: Ok(vec![1.0]) });
        let session = EnrollmentSession::new(consumer, active, encoder, store.clone());

        assert!(session.has_target_speaker());
        session.clear_profile();
        assert!(!session.has_target_speaker());
        assert_eq!(session.state(), EnrollmentState::Idle);
    }
}
