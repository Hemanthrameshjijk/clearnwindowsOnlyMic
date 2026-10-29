//! Pure, platform-independent logic behind the tabbed window: tab list, header status chip, the three profiles,
//! transcript search/format helpers. Nothing here touches iced, the engine or the OS, so it is all unit-tested.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Home,
    Microphone,
    CallAudio,
    Transcript,
    Devices,
    Advanced,
}

impl Tab {
    pub const ALL: [Tab; 6] = [Tab::Home, Tab::Microphone, Tab::CallAudio, Tab::Transcript, Tab::Devices, Tab::Advanced];

    pub const fn label(self) -> &'static str {
        match self {
            Tab::Home => "Home",
            Tab::Microphone => "Microphone",
            Tab::CallAudio => "Call audio",
            Tab::Transcript => "Transcript",
            Tab::Devices => "Devices",
            Tab::Advanced => "Advanced",
        }
    }
}

/// The one-word state shown next to the logo (and in the compact bar).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusChip {
    Muted,
    Paused,
    CallAudioNotSetUp,
    Protected,
}

impl StatusChip {
    pub const fn label(self) -> &'static str {
        match self {
            StatusChip::Muted => "Mic muted",
            StatusChip::Paused => "Paused",
            StatusChip::CallAudioNotSetUp => "Call audio not set up",
            StatusChip::Protected => "Protected",
        }
    }
}

/// Priority order: a muted mic beats everything, then "nothing is processing", then the call-audio warning.
/// `any_mic_stage_on`: at least one of noise / background voices / quiet-voice gate / studio / accent is on.
pub fn status_chip(mic_enabled: bool, any_mic_stage_on: bool, speaker_ready: bool) -> StatusChip {
    if !mic_enabled {
        StatusChip::Muted
    } else if !any_mic_stage_on {
        StatusChip::Paused
    } else if !speaker_ready {
        StatusChip::CallAudioNotSetUp
    } else {
        StatusChip::Protected
    }
}

/// A fixed combination of EXISTING mic-path switches (no new processing). Applied through the same message handlers
/// the individual controls use, so it persists and takes effect live. Accent and transcript are left as they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    pub name: &'static str,
    pub noise: bool,
    pub voice_gate: bool,
    pub studio: bool,
    pub studio_preset: &'static str,
    pub bvc_backend: &'static str,
}

/// - Office: quiet room, light touch - noise removal + Hush + Natural studio.
/// - Busy floor: many people talking nearby - adds the quiet-voice gate, studio off to save CPU.
/// - Presenting: you are the only voice that matters - noise removal + Hush + a fuller Balanced studio tone.
pub const PROFILES: [Profile; 3] = [
    Profile { name: "Office", noise: true, voice_gate: false, studio: true, studio_preset: "Natural", bvc_backend: "Hush" },
    Profile { name: "Busy floor", noise: true, voice_gate: true, studio: false, studio_preset: "Natural", bvc_backend: "Hush" },
    Profile { name: "Presenting", noise: true, voice_gate: false, studio: true, studio_preset: "Balanced", bvc_backend: "Hush" },
];

pub const PROFILE_NAMES: [&str; 3] = [PROFILES[0].name, PROFILES[1].name, PROFILES[2].name];

pub fn profile_by_name(name: &str) -> Option<&'static Profile> {
    PROFILES.iter().find(|p| p.name == name)
}

/// What the mic path is doing right now, for deciding whether a saved profile still describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicSnapshot {
    pub noise: bool,
    pub voice_gate: bool,
    pub studio: bool,
    pub studio_preset: &'static str,
    pub bvc_on: bool,
    pub bvc_backend: &'static str,
}

impl Profile {
    pub fn matches(&self, cur: &MicSnapshot) -> bool {
        self.noise == cur.noise
            && self.voice_gate == cur.voice_gate
            && self.studio == cur.studio
            && self.studio_preset == cur.studio_preset
            && cur.bvc_on
            && self.bvc_backend == cur.bvc_backend
    }
}

/// The profile to show as selected: the saved one, but only while the switches still match it (change a switch by
/// hand and the picker goes back to its placeholder instead of claiming a profile that is no longer true).
pub fn active_profile(saved: &str, cur: &MicSnapshot) -> Option<&'static str> {
    profile_by_name(saved).filter(|p| p.matches(cur)).map(|p| p.name)
}

/// Case-insensitive substring filter; an empty (or blank) query keeps every line.
pub fn filter_lines<'a>(lines: &'a [(u32, String)], query: &str) -> Vec<&'a (u32, String)> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return lines.iter().collect();
    }
    lines.iter().filter(|(_, t)| t.to_lowercase().contains(&q)).collect()
}

/// `[mm:ss] text` per line, the same shape the live transcript file uses.
pub fn format_transcript(lines: &[(u32, String)]) -> String {
    let mut out = String::new();
    for (secs, text) in lines {
        out.push_str(&format!("[{}] {}\n", asr_onnx::pipeline::fmt_time(*secs), text));
    }
    out
}

/// A problem the user should see right now (banner on every tab and in the collapsed bar) and that is also written
/// to the log once when it appears. `key` identifies the problem so it is logged once, not every tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alert {
    pub key: &'static str,
    pub text: String,
}

/// What the alert rules look at. Each field is only set when the matching feature is switched on.
#[derive(Debug, Default, Clone)]
pub struct AlertInput {
    /// Accent is on but fell back to the cleaned voice because the CPU could not keep up (worker real-time factor).
    pub accent_paused_rtf: Option<f32>,
    /// Accent is on but could not start (reason from the engine).
    pub accent_unavailable: Option<String>,
    /// Background voice removal is on but its model did not load.
    pub bvc_unavailable: bool,
    /// Background voice removal is on but stopped responding and was switched off for this session.
    pub bvc_stopped: bool,
    /// Processing has been over its time budget for several seconds.
    pub cpu_struggling: bool,
    /// The window lost its connection to the background engine (it reconnects by itself).
    pub engine_disconnected: bool,
}

pub fn alerts(i: &AlertInput) -> Vec<Alert> {
    let mut out = Vec::new();
    if i.engine_disconnected {
        out.push(Alert { key: "engine-disconnected", text: "Lost contact with the ClearNAI background engine. Reconnecting; audio keeps running if the engine is still alive.".into() });
    }
    if i.bvc_unavailable {
        out.push(Alert { key: "bvc-unavailable", text: "Background voice removal could not start, so other voices are not being removed.".into() });
    }
    if i.bvc_stopped {
        out.push(Alert { key: "bvc-stopped", text: "Background voice removal stopped responding and is off for this session. Restart ClearNAI to fix it.".into() });
    }
    if let Some(rtf) = i.accent_paused_rtf {
        out.push(Alert {
            key: "accent-paused",
            text: format!("Accent is paused: this PC cannot keep up (load {rtf:.2}). You are heard in your own voice. Switch off Transcript or close other apps."),
        });
    }
    if let Some(reason) = &i.accent_unavailable {
        out.push(Alert { key: "accent-unavailable", text: format!("Accent could not start: {reason}") });
    }
    if i.cpu_struggling {
        out.push(Alert { key: "cpu", text: "This PC is struggling; audio may crackle. Try the Low CPU option in Advanced.".into() });
    }
    out
}

/// An engine warning reads "Title - details": the title is shown in bold and the details below it, starting with a
/// capital. A warning without " - " is all details.
pub fn split_warning(w: &str) -> (Option<&str>, String) {
    let (title, rest) = match w.split_once(" - ") {
        Some((a, b)) => (Some(a), b),
        None => (None, w),
    };
    let mut chars = rest.chars();
    let rest = match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    };
    (title, rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_priority_is_muted_then_paused_then_call_audio_then_protected() {
        assert_eq!(status_chip(false, true, true), StatusChip::Muted);
        assert_eq!(status_chip(false, false, false), StatusChip::Muted);
        assert_eq!(status_chip(true, false, true), StatusChip::Paused);
        assert_eq!(status_chip(true, false, false), StatusChip::Paused);
        assert_eq!(status_chip(true, true, false), StatusChip::CallAudioNotSetUp);
        assert_eq!(status_chip(true, true, true), StatusChip::Protected);
    }

    #[test]
    fn chip_labels_match_the_prototype() {
        assert_eq!(StatusChip::Protected.label(), "Protected");
        assert_eq!(StatusChip::Muted.label(), "Mic muted");
        assert_eq!(StatusChip::Paused.label(), "Paused");
        assert_eq!(StatusChip::CallAudioNotSetUp.label(), "Call audio not set up");
    }

    #[test]
    fn every_profile_uses_names_the_real_handlers_accept() {
        for p in PROFILES {
            assert!(crate::shared::studio_preset_from_name(p.studio_preset).is_some(), "{}", p.name);
            assert!(tse_engine::BvcBackend::ALL.iter().any(|b| b.name() == p.bvc_backend), "{}", p.name);
            assert!(profile_by_name(p.name).is_some());
        }
        assert_eq!(PROFILE_NAMES, ["Office", "Busy floor", "Presenting"]);
    }

    #[test]
    fn profiles_are_distinct() {
        assert_ne!(PROFILES[0], PROFILES[1]);
        assert_ne!(PROFILES[0], PROFILES[2]);
        assert_ne!(PROFILES[1], PROFILES[2]);
    }

    fn snapshot_of(p: &Profile) -> MicSnapshot {
        MicSnapshot {
            noise: p.noise,
            voice_gate: p.voice_gate,
            studio: p.studio,
            studio_preset: p.studio_preset,
            bvc_on: true,
            bvc_backend: p.bvc_backend,
        }
    }

    #[test]
    fn saved_profile_is_shown_only_while_it_still_matches() {
        let office = profile_by_name("Office").unwrap();
        let mut cur = snapshot_of(office);
        assert_eq!(active_profile("Office", &cur), Some("Office"));
        cur.voice_gate = true;
        assert_eq!(active_profile("Office", &cur), None);
        assert_eq!(active_profile("", &snapshot_of(office)), None);
        assert_eq!(active_profile("Nope", &snapshot_of(office)), None);
        let mut bvc_off = snapshot_of(office);
        bvc_off.bvc_on = false;
        assert_eq!(active_profile("Office", &bvc_off), None);
    }

    #[test]
    fn transcript_filter_is_case_insensitive_and_blank_keeps_all() {
        let lines = vec![(1, "Hello World".to_string()), (2, "second line".to_string()), (3, "HELLO again".to_string())];
        assert_eq!(filter_lines(&lines, "").len(), 3);
        assert_eq!(filter_lines(&lines, "   ").len(), 3);
        let hits = filter_lines(&lines, "hello");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].0, 3);
        assert!(filter_lines(&lines, "zzz").is_empty());
    }

    #[test]
    fn transcript_format_is_one_timestamped_line_each() {
        let lines = vec![(5, "a".to_string()), (65, "b".to_string())];
        let out = format_transcript(&lines);
        assert_eq!(out.lines().count(), 2);
        assert!(out.lines().next().unwrap().ends_with("] a"));
        assert!(out.ends_with("b\n"));
        assert_eq!(format_transcript(&[]), "");
    }

    #[test]
    fn alerts_are_empty_when_all_is_well_and_one_per_problem_otherwise() {
        assert!(alerts(&AlertInput::default()).is_empty());
        let a = alerts(&AlertInput { accent_paused_rtf: Some(0.93), bvc_stopped: true, ..Default::default() });
        let keys: Vec<_> = a.iter().map(|x| x.key).collect();
        assert_eq!(keys, vec!["bvc-stopped", "accent-paused"]);
        assert!(a[1].text.contains("0.93"));
    }

    #[test]
    fn warnings_split_into_a_bold_title_and_capitalised_details() {
        let (t, r) = split_warning("No speaker-source device selected - incoming call audio is off.");
        assert_eq!(t, Some("No speaker-source device selected"));
        assert_eq!(r, "Incoming call audio is off.");
        let (t, r) = split_warning("plain warning");
        assert_eq!((t, r.as_str()), (None, "Plain warning"));
        assert_eq!(split_warning("").1, "");
    }
}
