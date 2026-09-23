use std::fmt;

/// Everything that can go wrong enrolling a target speaker, covering both
/// the recording/VAD/quality checks this crate owns directly and the
/// speaker-encoder failures it maps from `speaker_profile::SpeakerEncoderError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnrollmentError {
    /// No speech-level energy was detected anywhere in the recording (the
    /// user never spoke, or spoke far below the microphone's noise floor).
    NoSpeech,
    /// The recording was shorter than the minimum usable duration.
    TooShort,
    /// The recording's overall level (RMS) was too low to be usable, even
    /// though some speech-like energy was present.
    TooQuiet,
    /// Too large a fraction of samples were at or beyond the clipping
    /// threshold - the input gain was too hot for a clean reference.
    Clipped,
    /// The gap between the recording's quietest ("background") and loudest
    /// ("speech") frames was too small - a rough, explainable stand-in for
    /// "the room was too noisy" (steady background noise, a TV, another
    /// conversation) rather than a real SNR measurement. See
    /// `validate::estimate_snr_db`'s docs for exactly what this does and
    /// does not detect (it does NOT reliably tell a second human voice
    /// apart from the target speaker's own).
    TooNoisy,
    /// The speaker encoder ran but produced something unusable (not a
    /// recording/VAD problem - the embedding itself failed a sanity check,
    /// e.g. non-finite values). Carries the encoder's own reason.
    InvalidEmbedding(String),
    /// No real speaker-encoder backend is available. See
    /// `speaker_profile::SpeakerEncoderError::ModelUnavailable` and this
    /// project's honesty rule: this is reported, never silently
    /// papered over with a fake embedding.
    ModelUnavailable,
    /// Something went wrong capturing audio itself (not a quality problem -
    /// e.g. the realtime tap never received a single sample before the
    /// session ended). Carries a human-readable reason.
    RecordingError(String),
}

impl fmt::Display for EnrollmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnrollmentError::NoSpeech => write!(f, "no speech was detected in the recording"),
            EnrollmentError::TooShort => write!(f, "recording was too short"),
            EnrollmentError::TooQuiet => write!(f, "recording was too quiet"),
            EnrollmentError::Clipped => write!(f, "recording was clipped (input level too hot)"),
            EnrollmentError::TooNoisy => write!(
                f,
                "background noise was too loud relative to your voice - try re-recording somewhere quieter"
            ),
            EnrollmentError::InvalidEmbedding(reason) => write!(f, "invalid speaker embedding: {reason}"),
            EnrollmentError::ModelUnavailable => write!(f, "speaker encoder backend unavailable"),
            EnrollmentError::RecordingError(reason) => write!(f, "recording error: {reason}"),
        }
    }
}

impl std::error::Error for EnrollmentError {}

impl From<speaker_profile::SpeakerEncoderError> for EnrollmentError {
    fn from(e: speaker_profile::SpeakerEncoderError) -> Self {
        match e {
            speaker_profile::SpeakerEncoderError::ModelUnavailable => EnrollmentError::ModelUnavailable,
            speaker_profile::SpeakerEncoderError::InvalidAudio(reason) => EnrollmentError::RecordingError(reason),
            speaker_profile::SpeakerEncoderError::InferenceFailed(reason) => EnrollmentError::InvalidEmbedding(reason),
        }
    }
}
