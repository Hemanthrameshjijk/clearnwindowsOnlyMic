//! Line icons for the collapsed bar, as tiny inline SVGs (24x24, drawn in black with round caps and tinted when shown).

macro_rules! icon {
    ($body:expr) => {
        concat!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" "#,
            r#"stroke-linecap="round" stroke-linejoin="round">"#,
            $body,
            "</svg>"
        )
        .as_bytes()
    };
}

pub const MIC: &[u8] = icon!(r#"<rect x="9" y="2.5" width="6" height="12" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>"#);

pub const SPEAKER: &[u8] =
    icon!(r#"<path d="M11 5 6.5 9H3v6h3.5L11 19z"/><path d="M15 9a4 4 0 0 1 0 6M18 6a8 8 0 0 1 0 12"/>"#);

/// Noise removal: a sound wave.
pub const WAVE: &[u8] = icon!(r#"<path d="M3 12h0M7 8v8M11 4v16M15 7v10M19 10v4"/>"#);

/// Background voices: two people.
pub const PEOPLE: &[u8] = icon!(
    r#"<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><circle cx="17.5" cy="9" r="2.5"/><path d="M17.5 14.2a5 5 0 0 1 4.5 5"/>"#
);

/// Accent: translate.
pub const TRANSLATE: &[u8] =
    icon!(r#"<path d="M3 6h9M7.5 4v2M5 6c0 4 3 7 7 8M11 6c0 3-2 6-6 8"/><path d="M12.5 20l4-9 4 9M14 17h5"/>"#);

/// Transcript: closed captions.
pub const CAPTIONS: &[u8] = icon!(r#"<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M7 11h4M13 11h4M7 15h2M11 15h6"/>"#);

pub const GEAR: &[u8] = icon!(
    r#"<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1"/>"#
);

/// Home: a window with a title bar (as on the product page).
pub const HOME: &[u8] = icon!(r#"<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/>"#);

/// Devices: a laptop.
pub const DEVICES: &[u8] = icon!(
    r#"<path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16"/>"#
);

/// Advanced / processing: two sliders.
pub const SLIDERS: &[u8] = icon!(r#"<path d="M20 7h-9M14 17H5"/><circle cx="17" cy="17" r="3"/><circle cx="7" cy="7" r="3"/>"#);

/// Call audio: a phone.
pub const PHONE: &[u8] = icon!(
    r#"<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>"#
);

/// Transcript page: a document with lines.
pub const DOC: &[u8] = icon!(r#"<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/><path d="M14 2v4a2 2 0 0 0 2 2h4M10 9H8M16 13H8M16 17H8"/>"#);

pub const SHIELD: &[u8] = icon!(
    r#"<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>"#
);

pub const CLIPBOARD: &[u8] = icon!(r#"<rect width="8" height="4" x="8" y="2" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>"#);

pub const SEARCH: &[u8] = icon!(r#"<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>"#);

pub const REFRESH: &[u8] =
    icon!(r#"<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8M21 3v5h-5M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16M8 16H3v5"/>"#);

pub const COPY: &[u8] = icon!(r#"<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>"#);

pub const DOWNLOAD: &[u8] = icon!(r#"<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>"#);

pub const FOLDER: &[u8] = icon!(
    r#"<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>"#
);

pub const TRASH: &[u8] = icon!(r#"<path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M10 11v6M14 11v6"/>"#);

pub const PULSE: &[u8] = icon!(r#"<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>"#);

pub const USER: &[u8] = icon!(r#"<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>"#);

pub const HELP: &[u8] = icon!(r#"<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/>"#);

pub const MOON: &[u8] = icon!(r#"<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>"#);

pub const SUN: &[u8] =
    icon!(r#"<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>"#);

pub const CHEVRON_RIGHT: &[u8] = icon!(r#"<path d="m9 18 6-6-6-6"/>"#);
pub const CHEVRON_DOWN: &[u8] = icon!(r#"<path d="m6 9 6 6 6-6"/>"#);
pub const CHECK: &[u8] = icon!(r#"<path d="M20 6 9 17l-5-5"/>"#);
pub const CHECK_CIRCLE: &[u8] = icon!(r#"<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>"#);

pub const HEADPHONES: &[u8] = icon!(
    r#"<path d="M3 14v-3a9 9 0 0 1 18 0v3"/><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"/>"#
);

pub const SPARKLES: &[u8] = icon!(
    r#"<path d="M9.9 14.1 8 20l-1.9-5.9L0 12l6.1-2.1L8 4l1.9 5.9L16 12zM20 3v4M22 5h-4M20 17v4M22 19h-4"/>"#
);

/// Noise: three wavy lines.
pub const WAVES: &[u8] = icon!(
    r#"<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>"#
);

/// A speaker box (capture source).
pub const SPEAKER_BOX: &[u8] = icon!(r#"<rect x="5" y="2" width="14" height="20" rx="2"/><circle cx="12" cy="14" r="3.5"/><path d="M12 6h.01"/>"#);

/// Close.
pub const CLOSE: &[u8] = icon!(r#"<path d="M6 6l12 12M18 6 6 18"/>"#);

/// Expand to the full window.
pub const EXPAND: &[u8] = icon!(r#"<path d="M14.5 3H21v6.5M9.5 21H3v-6.5M21 3l-7 7M3 21l7-7"/>"#);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_icon_is_a_complete_svg_document() {
        for (name, bytes) in [
            ("mic", MIC),
            ("speaker", SPEAKER),
            ("wave", WAVE),
            ("people", PEOPLE),
            ("translate", TRANSLATE),
            ("captions", CAPTIONS),
            ("gear", GEAR),
            ("expand", EXPAND),
            ("close", CLOSE),
            ("home", HOME),
            ("devices", DEVICES),
            ("sliders", SLIDERS),
            ("phone", PHONE),
            ("doc", DOC),
            ("shield", SHIELD),
            ("clipboard", CLIPBOARD),
            ("search", SEARCH),
            ("refresh", REFRESH),
            ("copy", COPY),
            ("download", DOWNLOAD),
            ("folder", FOLDER),
            ("trash", TRASH),
            ("pulse", PULSE),
            ("user", USER),
            ("help", HELP),
            ("moon", MOON),
            ("sun", SUN),
            ("chevron_right", CHEVRON_RIGHT),
            ("chevron_down", CHEVRON_DOWN),
            ("check", CHECK),
            ("check_circle", CHECK_CIRCLE),
            ("headphones", HEADPHONES),
            ("sparkles", SPARKLES),
            ("waves", WAVES),
            ("speaker_box", SPEAKER_BOX),
        ] {
            let s = std::str::from_utf8(bytes).unwrap();
            assert!(s.starts_with("<svg") && s.ends_with("</svg>"), "{name} is not a whole svg");
        }
    }
}
