use crate::error::EnrollmentError;

/// The enrollment state machine, per the task's required shape:
/// `Idle -> Preparing -> Recording -> Validating -> ExtractingEmbedding ->
/// Ready`, with `Failed` reachable from `Recording` onward carrying a
/// specific `EnrollmentError`. Deliberately has no payload beyond `Failed`'s
/// error - the enrolled profile itself lives in
/// `speaker_profile::TargetSpeakerStore`, not duplicated here, so this type
/// only ever describes "where in the process are we," never "what did we
/// get."
#[derive(Debug, Clone, PartialEq)]
pub enum EnrollmentState {
    /// No recording in progress; either never started or the previous
    /// result (success or failure) has been acknowledged/cleared.
    Idle,
    /// `start_recording` was just called; the realtime tap is being armed.
    /// Present as its own state (rather than jumping straight to
    /// `Recording`) so a GUI can show "Preparing..." for the brief window
    /// before the worker thread confirms it's actually pulling samples -
    /// see `EnrollmentSession::start_recording`.
    Preparing,
    /// Actively capturing reference audio.
    Recording,
    /// Recording finished (stopped manually, auto-stopped, or hit the
    /// maximum duration); running VAD/quality checks now.
    Validating,
    /// Validation passed; running the speaker encoder now.
    ExtractingEmbedding,
    /// A target speaker profile was successfully enrolled and published to
    /// `TargetSpeakerStore`.
    Ready,
    /// Something failed - see `EnrollmentError` for the specific reason.
    Failed(EnrollmentError),
}

impl EnrollmentState {
    /// `true` while a recording attempt is actively in flight (from
    /// `Preparing` through `ExtractingEmbedding`) - used to reject a second
    /// concurrent `start_recording` call rather than racing two worker
    /// threads over the same tap.
    pub fn is_in_progress(&self) -> bool {
        matches!(
            self,
            EnrollmentState::Preparing
                | EnrollmentState::Recording
                | EnrollmentState::Validating
                | EnrollmentState::ExtractingEmbedding
        )
    }
}
