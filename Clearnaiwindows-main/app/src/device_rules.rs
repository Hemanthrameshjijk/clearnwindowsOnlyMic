//! Startup sanity check of the saved device choices against what is actually installed, so a stale or wrong
//! saved id never silently routes audio the wrong way (e.g. the physical mic chosen as the "call audio source",
//! which plays your own voice out of the speakers).

use crate::settings::Settings;
use crate::shared::{DeviceList, DeviceOption};

/// True for virtual-cable style endpoints (VB-CABLE and similar).
fn is_virtual(label: &str) -> bool {
    let l = label.to_ascii_lowercase();
    l.contains("cable") || l.contains("vb-audio") || l.contains("virtual")
}

/// "CABLE Input (VB-Audio Virtual Cable)" -> "cable output (vb-audio virtual cable)": the other end of the same
/// cable. Whatever we render into one end comes out of the other, so that end must never feed the speaker path.
fn paired_capture_label(render_label: &str) -> String {
    render_label.to_ascii_lowercase().replacen("input", "output", 1)
}

fn find<'a>(list: &'a [DeviceOption], id: &str) -> Option<&'a DeviceOption> {
    list.iter().find(|d| d.id == id)
}

/// Picker choices: devices that are (`virtual_wanted`) or are not virtual cables. A real mic must not be a cable
/// and the speakers must not be a cable; the mic target and call source must be.
pub fn only_virtual(options: &[DeviceOption], virtual_wanted: bool) -> Vec<DeviceOption> {
    options.iter().filter(|d| is_virtual(&d.label) == virtual_wanted).cloned().collect()
}

/// Call-audio source choices: virtual capture ends, minus the end paired with the mic target's cable.
pub fn call_source_choices(devices: &DeviceList, mic_target_id: Option<&str>) -> Vec<DeviceOption> {
    let paired = mic_target_id
        .and_then(|id| find(&devices.render, id))
        .map(|d| paired_capture_label(&d.label));
    only_virtual(&devices.capture, true)
        .into_iter()
        .filter(|d| paired.as_deref() != Some(d.label.to_ascii_lowercase().as_str()))
        .collect()
}

/// What to actually use this run, and which saved choices were never valid (and so are cleared in settings).
pub struct Resolved {
    pub settings: Settings,
    pub mic_target: Option<String>,
    pub speaker_source: Option<String>,
    pub real_output: Option<String>,
    /// Plain-language notes for the log.
    pub notes: Vec<String>,
    /// Settings were changed in a way that should be saved (a choice that could never work was cleared).
    pub changed: bool,
}

pub fn resolve(saved: &Settings, devices: &DeviceList, auto_mic_target: Option<String>) -> Resolved {
    let mut s = saved.clone();
    let mut notes = Vec::new();
    let mut changed = false;
    let no_enumeration = devices.render.is_empty() && devices.capture.is_empty();

    // Virtual mic target: a present virtual render endpoint, else the single auto-detected CABLE Input.
    let mic_target = match saved.virtual_mic_target_id.as_deref() {
        Some(id) if no_enumeration || find(&devices.render, id).is_some_and(|d| is_virtual(&d.label)) => Some(id.to_string()),
        Some(_) => {
            notes.push("saved virtual microphone target is missing or not a virtual cable; auto-detecting".into());
            auto_mic_target
        }
        None => auto_mic_target,
    };
    let mic_target_label = mic_target.as_deref().and_then(|id| find(&devices.render, id)).map(|d| d.label.clone());

    // Call audio source: must be a virtual cable's capture end, and not the end paired with the mic target.
    let mut clear_source_for_good = false;
    let speaker_source = match saved.virtual_speaker_source_id.as_deref() {
        Some(id) if no_enumeration => Some(id.to_string()),
        Some(id) => match find(&devices.capture, id) {
            Some(d) if is_virtual(&d.label) => {
                let loops_back = mic_target_label
                    .as_deref()
                    .is_some_and(|m| paired_capture_label(m) == d.label.to_ascii_lowercase());
                if loops_back {
                    notes.push(format!("call audio source '{}' is the other end of the mic cable (it would echo your voice); cleared", d.label));
                    clear_source_for_good = true;
                    None
                } else {
                    Some(id.to_string())
                }
            }
            Some(d) => {
                notes.push(format!("call audio source '{}' is a real microphone, not a virtual cable; cleared", d.label));
                clear_source_for_good = true;
                None
            }
            None => {
                notes.push("saved call audio source is not connected; call audio cleanup is off".into());
                None
            }
        },
        None => None,
    };
    if clear_source_for_good {
        s.virtual_speaker_source_id = None;
        changed = true;
    }

    // Real microphone must not be a virtual cable (that would feed the cable back into itself).
    if let Some(d) = saved.physical_mic_device_id.as_deref().and_then(|id| find(&devices.capture, id)) {
        if is_virtual(&d.label) {
            notes.push(format!("saved microphone '{}' is a virtual cable; using the default microphone", d.label));
            s.physical_mic_device_id = None;
            changed = true;
        }
    }

    // Real speakers: a missing device means "system default" for this run (kept in settings for when it returns);
    // a virtual cable would feed call audio back into the mic path.
    let real_output = match saved.real_speaker_output_device_id.as_deref() {
        Some(id) if no_enumeration => Some(id.to_string()),
        Some(id) => match find(&devices.render, id) {
            Some(d) if is_virtual(&d.label) => {
                notes.push(format!("speaker output '{}' is a virtual cable; using the default speakers", d.label));
                s.real_speaker_output_device_id = None;
                changed = true;
                None
            }
            Some(_) => Some(id.to_string()),
            None => {
                notes.push("saved speaker output is not connected; using the default speakers".into());
                None
            }
        },
        None => None,
    };

    Resolved { settings: s, mic_target, speaker_source, real_output, notes, changed }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(id: &str, label: &str) -> DeviceOption {
        DeviceOption { id: id.into(), label: label.into() }
    }

    fn list() -> DeviceList {
        DeviceList {
            render: vec![
                dev("realtek-out", "Speakers (Realtek(R) Audio)"),
                dev("cable-in", "CABLE Input (VB-Audio Virtual Cable)"),
                dev("epos-out", "Speakers (EPOS PC 8 USB)"),
            ],
            capture: vec![
                dev("realtek-mic", "Microphone Array (Realtek(R) Audio)"),
                dev("cable-out", "CABLE Output (VB-Audio Virtual Cable)"),
                dev("epos-mic", "Microphone (EPOS PC 8 USB)"),
            ],
        }
    }

    #[test]
    fn physical_mic_as_call_source_is_cleared() {
        let mut s = Settings::default();
        s.virtual_speaker_source_id = Some("realtek-mic".into());
        let r = resolve(&s, &list(), None);
        assert!(r.speaker_source.is_none());
        assert!(r.changed && r.settings.virtual_speaker_source_id.is_none());
    }

    #[test]
    fn other_end_of_the_mic_cable_is_not_a_call_source() {
        let mut s = Settings::default();
        s.virtual_mic_target_id = Some("cable-in".into());
        s.virtual_speaker_source_id = Some("cable-out".into());
        assert!(resolve(&s, &list(), None).speaker_source.is_none());
    }

    #[test]
    fn a_separate_cable_is_a_valid_call_source() {
        let mut devices = list();
        devices.render.push(dev("b-in", "CABLE-B Input (VB-Audio Cable B)"));
        devices.capture.push(dev("b-out", "CABLE-B Output (VB-Audio Cable B)"));
        let mut s = Settings::default();
        s.virtual_mic_target_id = Some("cable-in".into());
        s.virtual_speaker_source_id = Some("b-out".into());
        assert_eq!(resolve(&s, &devices, None).speaker_source.as_deref(), Some("b-out"));
    }

    #[test]
    fn missing_devices_fall_back_without_forgetting_the_saved_speaker() {
        let mut s = Settings::default();
        s.real_speaker_output_device_id = Some("gone".into());
        s.virtual_mic_target_id = Some("gone-too".into());
        let r = resolve(&s, &list(), Some("cable-in".into()));
        assert!(r.real_output.is_none());
        assert_eq!(r.mic_target.as_deref(), Some("cable-in"));
        assert_eq!(r.settings.real_speaker_output_device_id.as_deref(), Some("gone"));
    }

    #[test]
    fn cable_as_real_speakers_or_mic_is_cleared() {
        let mut s = Settings::default();
        s.real_speaker_output_device_id = Some("cable-in".into());
        s.physical_mic_device_id = Some("cable-out".into());
        let r = resolve(&s, &list(), None);
        assert!(r.real_output.is_none() && r.settings.physical_mic_device_id.is_none() && r.changed);
    }

    #[test]
    fn empty_enumeration_keeps_saved_choices() {
        let mut s = Settings::default();
        s.virtual_speaker_source_id = Some("x".into());
        assert_eq!(resolve(&s, &DeviceList::empty(), None).speaker_source.as_deref(), Some("x"));
    }
}
