/// Default reference sentence shown to the user during "Record Voice
/// Reference" - long enough to give a real encoder several seconds of
/// varied phonetic content, short enough to read comfortably in one breath.
/// Configurable (see `Settings::reference_sentence` in the app crate) - this
/// is only the fallback when nothing has been customized.
pub const DEFAULT_REFERENCE_SENTENCE: &str =
    "Hello, this is my ClearNAI voice reference. I am ready to start my call.";
