//! The tabbed full window (Home / Microphone / Call audio / Transcript / Devices / Advanced) and the compact bar.
//! Every control here already existed in the old single-page `view_main`; this only reorganises them and adds the
//! header chip, Mute button, profile picker, transcript tools and the Advanced diagnostics. All messages go through
//! the same `update_main` handlers as before.

use super::{device_picker, device_picker_of, speaker_status_row, MainState, Message, BVC_BACKEND_NAMES, SPEAKER_BVC_BACKEND_NAMES};
use crate::shared::{studio_preset_from_index, studio_preset_name, STUDIO_PRESET_NAMES};
use crate::status_ui::{status_color, status_label};
use crate::ui_logic::{self, MicSnapshot, StatusChip, Tab};
use iced::widget::{button, column, container, pick_list, progress_bar, row, scrollable, svg, text, text_input, toggler};
use iced::{Color, Element, Length};
use tse_engine::BvcBackend;

fn grey() -> Color {
    Color::from_rgb(0.55, 0.55, 0.55)
}

fn red() -> Color {
    Color::from_rgb(0.85, 0.2, 0.2)
}

const LOGO_MARK_LIGHT: &[u8] = include_bytes!("../../assets/logo_mark_light.svg");
const LOGO_MARK_DARK: &[u8] = include_bytes!("../../assets/logo_mark_dark.svg");

/// The ClearNAI mark (arc + sound bars) in the light or dark variant.
pub fn logo_mark<'a>(dark: bool, size: f32) -> Element<'a, Message> {
    let bytes = if dark { LOGO_MARK_DARK } else { LOGO_MARK_LIGHT };
    svg(svg::Handle::from_memory(bytes)).width(size).height(size).into()
}

/// The mark plus the "ClearNAI" wordmark (Clear in navy and NAI in blue on light; all white on dark).
pub fn logo_with_wordmark<'a>(dark: bool, size: f32) -> Element<'a, Message> {
    let hex = |v: u32| Color::from_rgb8((v >> 16) as u8, (v >> 8) as u8, v as u8);
    let (c1, c2) = if dark { (Color::WHITE, Color::WHITE) } else { (hex(0x0B1736), hex(0x0A6CF0)) };
    let bold = iced::Font { weight: iced::font::Weight::Bold, ..iced::Font::DEFAULT };
    row![
        logo_mark(dark, size),
        row![text("Clear").size(size * 0.75).font(bold).color(c1), text("NAI").size(size * 0.75).font(bold).color(c2)],
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center)
    .into()
}

/// (background, text) for a chip, using the prototype's light and dark tokens.
fn chip_colors(kind: ChipKind, dark: bool) -> (Color, Color) {
    let hex = |v: u32| Color::from_rgb8((v >> 16) as u8, (v >> 8) as u8, v as u8);
    match (kind, dark) {
        (ChipKind::Ok, false) => (hex(0xE5F6EE), hex(0x1A6A44)),
        (ChipKind::Ok, true) => (hex(0x0E3324), hex(0x6FDCA5)),
        (ChipKind::Warn, false) => (hex(0xFFF4DC), hex(0x6E4300)),
        (ChipKind::Warn, true) => (hex(0x33260A), hex(0xF5C76A)),
        (ChipKind::Off, false) => (hex(0xECEFF4), hex(0x4A5368)),
        (ChipKind::Off, true) => (hex(0x1E2936), hex(0xB3BFCD)),
        (ChipKind::Bad, false) => (hex(0xFDECEA), hex(0xA1231B)),
        (ChipKind::Bad, true) => (hex(0x3D1517), hex(0xFF8F85)),
    }
}

#[derive(Clone, Copy)]
enum ChipKind {
    Ok,
    Warn,
    Off,
    Bad,
}

fn chip_kind(chip: StatusChip) -> ChipKind {
    match chip {
        StatusChip::Protected => ChipKind::Ok,
        StatusChip::CallAudioNotSetUp => ChipKind::Warn,
        StatusChip::Paused => ChipKind::Off,
        StatusChip::Muted => ChipKind::Bad,
    }
}

fn chip<'a>(label: &'a str, kind: ChipKind, dark: bool) -> Element<'a, Message> {
    let (bg, fg) = chip_colors(kind, dark);
    container(text(label).size(12))
        .padding([3, 10])
        .style(move |_| container::Style {
            background: Some(bg.into()),
            text_color: Some(fg),
            border: iced::Border::default().rounded(12),
            ..container::Style::default()
        })
        .into()
}

fn any_mic_stage_on(state: &MainState) -> bool {
    let h = &state.handles;
    h.mic_noise.is_on() || h.mic_bvc.is_on() || h.mic_voice_gate.is_on() || h.mic_studio.is_on() || state.settings.accent_on
}

fn speaker_ready(state: &MainState) -> bool {
    state.settings.virtual_speaker_source_id.is_some()
}

pub fn current_chip(state: &MainState) -> StatusChip {
    ui_logic::status_chip(state.handles.mic_master_enabled.is_on(), any_mic_stage_on(state), speaker_ready(state))
}

fn mic_snapshot(state: &MainState) -> MicSnapshot {
    let h = &state.handles;
    MicSnapshot {
        noise: h.mic_noise.is_on(),
        voice_gate: h.mic_voice_gate.is_on(),
        studio: h.mic_studio.is_on(),
        studio_preset: studio_preset_name(studio_preset_from_index(h.mic_studio_preset.load())),
        bvc_on: h.mic_bvc.is_on(),
        bvc_backend: BvcBackend::from_index(h.bvc_backend.load()).name(),
    }
}

/// A rounded on/off button (the prototype's quick "pills"): filled when on.
fn pill<'a>(label: &'a str, on: bool, msg: Message) -> Element<'a, Message> {
    button(text(label).size(12))
        .padding([7, 12])
        .on_press(msg)
        .style(move |t, s| if on { button::primary(t, s) } else { button::secondary(t, s) })
        .into()
}

fn mute_button<'a>(state: &MainState) -> Element<'a, Message> {
    let muted = !state.handles.mic_master_enabled.is_on();
    button(text(if muted { "Unmute mic" } else { "Mute mic" }).size(14))
        .padding(12)
        .width(Length::Fill)
        .on_press(Message::ToggleMute)
        .style(move |t, s| if muted { button::danger(t, s) } else { button::primary(t, s) })
        .into()
}

fn card<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    container(content).padding(16).width(Length::Fill).style(container::bordered_box).into()
}

fn card_title<'a>(title: &'a str, subtitle: &'a str) -> Element<'a, Message> {
    column![text(title).size(15), text(subtitle).size(12).style(text::secondary)].spacing(2).into()
}

pub fn view_full(state: &MainState) -> Element<'_, Message> {
    let dark = state.settings.theme_dark;
    let chip_now = current_chip(state);
    let selected_profile = ui_logic::active_profile(&state.settings.profile, &mic_snapshot(state));
    let header = row![
        logo_with_wordmark(dark, 30.0),
        chip(chip_now.label(), chip_kind(chip_now), dark),
        container(text("")).width(Length::Fill),
        text("Profile").size(12),
        pick_list(&ui_logic::PROFILE_NAMES[..], selected_profile, Message::SelectProfile).placeholder("Custom"),
        button(text("Collapse").size(12)).on_press(Message::ToggleCompact),
    ]
    .spacing(10)
    .align_y(iced::Alignment::Center)
    .padding([14, 18]);

    let mut tab_bar = row![].spacing(2);
    for t in Tab::ALL {
        let current = t == state.tab;
        tab_bar = tab_bar.push(
            button(text(t.label()).size(13))
                .padding([8, 10])
                .on_press(Message::SelectTab(t))
                .style(move |th, s| if current { button::primary(th, s) } else { button::text(th, s) }),
        );
    }

    let body = match state.tab {
        Tab::Home => home_tab(state),
        Tab::Microphone => microphone_tab(state),
        Tab::CallAudio => call_audio_tab(state),
        Tab::Transcript => transcript_tab(state),
        Tab::Devices => devices_tab(state),
        Tab::Advanced => advanced_tab(state),
    };

    let mut warnings_col = column![].spacing(4);
    for w in &state.warnings {
        warnings_col = warnings_col.push(text(format!("\u{26A0} {w}")).size(12).style(text::warning));
    }

    column![
        header,
        alert_banner(state),
        container(tab_bar).padding([0, 12]),
        iced::widget::rule::horizontal(1),
        scrollable(column![body, warnings_col].spacing(16).padding(18)).height(Length::Fill),
    ]
    .into()
}

/// Red banner listing the current problems (nothing when there are none). Shown on every tab.
fn alert_banner(state: &MainState) -> Element<'_, Message> {
    if state.alerts.is_empty() {
        return column![].into();
    }
    let mut col = column![].spacing(4);
    for a in &state.alerts {
        col = col.push(text(format!("\u{26A0} {}", a.text)).size(13));
    }
    container(col).padding(10).width(Length::Fill).style(container::danger).into()
}

/// The collapsed always-on-top bar: status chip, Mute, the four quick switches, and a hint while call audio is not
/// set up.
pub fn view_compact(state: &MainState) -> Element<'_, Message> {
    let dark = state.settings.theme_dark;
    let chip_now = current_chip(state);
    let h = &state.handles;
    let noise = h.mic_noise.is_on();
    let voices = h.mic_bvc.is_on();
    let accent = state.settings.accent_on;
    let transcript = state.settings.transcript_on;

    let header = row![
        logo_mark(dark, 22.0),
        chip(chip_now.label(), chip_kind(chip_now), dark),
        container(text("")).width(Length::Fill),
        button(text("Expand").size(12)).on_press(Message::ToggleCompact),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center);

    let mut col = column![
        header,
        progress_bar(0.0..=1.0, state.level.clamp(0.0, 1.0)).girth(Length::Fixed(8.0)),
        mute_button(state),
        row![
            pill(if noise { "Noise \u{00B7} On" } else { "Noise \u{00B7} Off" }, noise, Message::ToggleMicNoise(!noise)),
            pill(if voices { "Voices \u{00B7} On" } else { "Voices \u{00B7} Off" }, voices, Message::ToggleMicBvc(!voices)),
            pill(if accent { "Accent \u{00B7} On" } else { "Accent \u{00B7} Off" }, accent, Message::ToggleAccent(!accent)),
        ]
        .spacing(6),
        row![pill(
            if transcript { "Transcript \u{00B7} On" } else { "Transcript \u{00B7} Off" },
            transcript,
            Message::ToggleTranscript(!transcript)
        )],
    ]
    .spacing(8);
    if let Some(a) = state.alerts.first() {
        col = col.push(container(text(format!("\u{26A0} {}", a.text)).size(11)).padding(6).width(Length::Fill).style(container::danger));
    }
    if !speaker_ready(state) {
        col = col.push(text("Call audio isn't set up yet. Open the full window to finish.").size(11).style(text::secondary));
    }
    container(col).padding(12).into()
}

fn home_tab(state: &MainState) -> Element<'_, Message> {
    let dark = state.settings.theme_dark;
    let h = &state.handles;
    let mic_enabled = h.mic_master_enabled.is_on();
    // The mic card's own chip: Active / Paused / Muted (the header chip also folds in call-audio readiness).
    let mic_chip = match ui_logic::status_chip(mic_enabled, any_mic_stage_on(state), true) {
        StatusChip::Protected => ("Active", ChipKind::Ok),
        StatusChip::Paused => ("Paused", ChipKind::Off),
        _ => ("Muted", ChipKind::Bad),
    };
    let noise = h.mic_noise.is_on();
    let voices = h.mic_bvc.is_on();
    let accent = state.settings.accent_on;
    let studio = h.mic_studio.is_on();

    let mut mic_col = column![
        row![
            container(card_title("Your microphone", "What others hear on the call")).width(Length::Fill),
            chip(mic_chip.0, mic_chip.1, dark),
        ]
        .align_y(iced::Alignment::Center),
        // Before = raw mic peak (`EngineHandles::input_level`), After = peak of the processed frame
        // (`EngineHandles::level`, taken after the whole chain).
        row![
            text("Before").size(12).width(Length::Fixed(44.0)),
            progress_bar(0.0..=1.0, state.input_level.clamp(0.0, 1.0)).girth(Length::Fixed(8.0)),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center),
        row![
            text("After").size(12).width(Length::Fixed(44.0)),
            progress_bar(0.0..=1.0, state.level.clamp(0.0, 1.0)).girth(Length::Fixed(8.0)),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center),
        row![
            pill("Noise", noise, Message::ToggleMicNoise(!noise)),
            pill("Background voices", voices, Message::ToggleMicBvc(!voices)),
            pill("Accent", accent, Message::ToggleAccent(!accent)),
            pill("Studio", studio, Message::ToggleMicStudio(!studio)),
        ]
        .spacing(8),
        mute_button(state),
    ]
    .spacing(12);
    if accent {
        mic_col = mic_col.push(text("Accent conversion adds about 0.5 s of delay.").size(12).style(text::warning));
    }

    let ready = speaker_ready(state);
    let spk_enabled = h.speaker_master_enabled.is_on();
    let spk_chip = if !ready {
        ("Not set up", ChipKind::Warn)
    } else if spk_enabled {
        ("Active", ChipKind::Ok)
    } else {
        ("Off", ChipKind::Off)
    };
    let mut spk_col = column![row![
        container(card_title("Call audio", "What you hear from the caller")).width(Length::Fill),
        chip(spk_chip.0, spk_chip.1, dark),
    ]
    .align_y(iced::Alignment::Center)]
    .spacing(12);
    if !ready {
        spk_col = spk_col.push(
            row![
                container(text("Incoming audio isn't being cleaned yet. It needs a second virtual cable.").size(12).style(text::secondary))
                    .width(Length::Fill),
                button(text("Set up").size(13)).on_press(Message::SelectTab(Tab::Devices)),
            ]
            .spacing(12)
            .align_y(iced::Alignment::Center),
        );
    } else {
        spk_col = spk_col.push(
            toggler(spk_enabled).label("Clean the caller's side (noise and background voices)").on_toggle(Message::ToggleSpeakerMasterEnabled),
        );
    }

    column![card(mic_col), card(spk_col)].spacing(14).into()
}

/// `technical` (Advanced tab only) adds the underlying reason; the normal view stays plain.
fn bvc_note(state: &MainState, technical: bool) -> Option<Element<'_, Message>> {
    let selected = BvcBackend::from_index(state.handles.bvc_backend.load());
    match selected {
        BvcBackend::Hush if !state.handles.mic_bvc_available => Some(
            text(if technical {
                format!(
                    "Hush is unavailable: {}",
                    state.handles.mic_bvc_unavailable_reason.as_deref().unwrap_or("weya_nc.dll not found")
                )
            } else {
                "Background voice removal isn't available right now. Try another model in Advanced.".to_string()
            })
            .size(12)
            .style(text::danger)
            .into(),
        ),
        // The backend's own watchdog gave up on it after repeated stalls (see `EngineHandles::mic_hush_status`).
        BvcBackend::Hush if state.handles.mic_hush_status.as_ref().is_some_and(|s| s.is_bypassed()) => Some(
            text("Background voice removal stopped responding and was switched off for this session. Restart ClearNAI to try again.")
            .size(12)
            .style(text::danger)
            .into(),
        ),
        BvcBackend::DeepFilter if !state.handles.mic_deep_filter_available => Some(
            text(if technical {
                format!(
                    "DeepFilterNet is unavailable: {}",
                    state.handles.mic_deep_filter_unavailable_reason.as_deref().unwrap_or("model failed to load")
                )
            } else {
                "Background voice removal isn't available right now. Try another model in Advanced.".to_string()
            })
            .size(12)
            .style(text::danger)
            .into(),
        ),
        BvcBackend::DeepFilter if technical => Some(
            text("A different noise and voice model (DeepFilterNet). Try it if the default doesn't suit your room.")
                .size(11)
                .style(text::secondary)
                .into(),
        ),
        BvcBackend::HushBpoV1 if !state.handles.mic_bpo_v1_available => Some(
            text(if technical {
                format!(
                    "bpo_v1 is unavailable: {}",
                    state.handles.mic_bpo_v1_unavailable_reason.as_deref().unwrap_or("model failed to load")
                )
            } else {
                "Background voice removal isn't available right now. Try another model in Advanced.".to_string()
            })
            .size(12)
            .style(text::danger)
            .into(),
        ),
        BvcBackend::HushBpoV1 if state.handles.mic_bpo_v1_status.as_ref().is_some_and(|s| s.is_bypassed()) => Some(
            text("Background voice removal stopped responding and was switched off for this session. Restart ClearNAI to try again.")
            .size(12)
            .style(text::danger)
            .into(),
        ),
        BvcBackend::HushBpoV1 if technical => Some(
            text("An alternative version of the default model (Hush bpo_v1).")
                .size(11)
                .style(text::secondary)
                .into(),
        ),
        _ => None,
    }
}

fn accent_section(state: &MainState) -> Element<'_, Message> {
    let status = state.handles.accent.status();
    let label = accent_onnx::VOICE_CHOICES[(state.settings.accent_voice_index() - 1) as usize];
    // The switch only records what you clicked; this line says what is REALLY happening (green = converting,
    // amber = starting, red = switched on but NOT converting, grey = off).
    let (dot, color) = match status.state {
        accent_onnx::pipeline::AccentState::Active => ("\u{25CF}", Color::from_rgb(0.2, 0.65, 0.3)),
        accent_onnx::pipeline::AccentState::Loading | accent_onnx::pipeline::AccentState::Starting => ("\u{25CF}", Color::from_rgb(0.9, 0.65, 0.1)),
        accent_onnx::pipeline::AccentState::Paused | accent_onnx::pipeline::AccentState::Unavailable => ("\u{25CF}", red()),
        accent_onnx::pipeline::AccentState::Off => ("\u{25CB}", grey()),
    };
    column![
        toggler(state.settings.accent_on).label("Accent conversion").on_toggle(Message::ToggleAccent),
        text("Indian -> American English \u{00B7} adds about 0.5 s delay").size(12).style(text::secondary),
        row![
            text("Accent voice").size(12),
            pick_list(&accent_onnx::VOICE_CHOICES[..], Some(label), Message::SelectAccentVoice),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center),
        text(format!("{dot} {}", status.line())).size(12).color(color),
        text("Cleaner input converts better.").size(11).style(text::secondary),
    ]
    .spacing(2)
    .into()
}

fn microphone_tab(state: &MainState) -> Element<'_, Message> {
    let h = &state.handles;
    let mut voices = column![
        toggler(h.mic_bvc.is_on()).label("Background voice removal").on_toggle(Message::ToggleMicBvc),
        text("Removes other people talking near you. Choose the model in Advanced.").size(12).style(text::secondary),
    ]
    .spacing(2);
    if let Some(note) = bvc_note(state, false) {
        voices = voices.push(note);
    }
    let studio = column![
        toggler(h.mic_studio.is_on()).label("Studio voice").on_toggle(Message::ToggleMicStudio),
        text("Fuller, clearer tone").size(12).style(text::secondary),
        row![
            text("Studio preset").size(12),
            pick_list(
                &STUDIO_PRESET_NAMES[..],
                Some(studio_preset_name(studio_preset_from_index(h.mic_studio_preset.load()))),
                Message::SelectMicStudioPreset,
            ),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center),
    ]
    .spacing(2);
    card(
        column![
            column![
                toggler(h.mic_noise.is_on()).label("Noise removal").on_toggle(Message::ToggleMicNoise),
                text("Fans, typing, traffic").size(12).style(text::secondary),
            ]
            .spacing(2),
            voices,
            column![
                toggler(h.mic_voice_gate.is_on()).label("Quiet nearby voices").on_toggle(Message::ToggleMicVoiceGate),
                text("Fast. Lowers voices quieter than yours.").size(12).style(text::secondary),
            ]
            .spacing(2),
            column![
                toggler(h.echo_cancel.is_on()).label("Echo removal").on_toggle(Message::ToggleEchoCancel),
                text("Removes call audio from your speakers out of your mic. Not needed with headphones.").size(12).style(text::secondary),
            ]
            .spacing(2),
            accent_section(state),
            studio,
        ]
        .spacing(16),
    )
}

fn call_audio_tab(state: &MainState) -> Element<'_, Message> {
    let h = &state.handles;
    let selected = tse_engine::SpeakerBvcBackend::from_index(h.speaker_bvc_backend.load());
    let note: Option<Element<'_, Message>> = match selected {
        tse_engine::SpeakerBvcBackend::Hush if !h.speaker_bvc_available => Some(
            text("Background voice removal isn't available right now. Try another model.")
                .size(12)
                .style(text::danger)
                .into(),
        ),
        tse_engine::SpeakerBvcBackend::DeepFilter if !h.speaker_deep_filter_available => Some(
            text("Background voice removal isn't available right now. Try another model.")
            .size(12)
            .style(text::danger)
            .into(),
        ),
        _ => None,
    };
    let mut col = column![
        text("Call audio isn't set up yet. Choose your devices in the Devices tab.")
            .size(12)
            .style(text::warning),
    ]
    .spacing(12);
    if speaker_ready(state) {
        col = column![].spacing(12);
    }
    col = col
        .push(speaker_status_row(state))
        .push(toggler(h.speaker_master_enabled.is_on()).label("Call audio path enabled").on_toggle(Message::ToggleSpeakerMasterEnabled))
        .push(toggler(h.speaker_noise.is_on()).label("Noise removal").on_toggle(Message::ToggleSpeakerNoise))
        .push(
            row![
                text("Background voice model").size(12),
                pick_list(&SPEAKER_BVC_BACKEND_NAMES[..], Some(selected.name()), Message::SelectSpeakerBvcBackend),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center),
        );
    if let Some(n) = note {
        col = col.push(n);
    }
    col = col
        .push(toggler(h.speaker_studio.is_on()).label("Studio voice").on_toggle(Message::ToggleSpeakerStudio))
        .push(
            row![
                text("Studio preset").size(12),
                pick_list(
                    &STUDIO_PRESET_NAMES[..],
                    Some(studio_preset_name(studio_preset_from_index(h.speaker_studio_preset.load()))),
                    Message::SelectSpeakerStudioPreset,
                ),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center),
        );
    card(col)
}

fn transcript_tab(state: &MainState) -> Element<'_, Message> {
    let status = state.handles.transcript.status();
    let (dot, color) = match status.state {
        asr_onnx::pipeline::TranscriptState::Listening | asr_onnx::pipeline::TranscriptState::Transcribing => {
            ("\u{25CF}", Color::from_rgb(0.2, 0.65, 0.3))
        }
        asr_onnx::pipeline::TranscriptState::Loading => ("\u{25CF}", Color::from_rgb(0.9, 0.65, 0.1)),
        asr_onnx::pipeline::TranscriptState::Unavailable => ("\u{25CF}", red()),
        asr_onnx::pipeline::TranscriptState::Off => ("\u{25CB}", grey()),
    };
    let mut col = column![
        toggler(state.settings.transcript_on).label("Live transcript").on_toggle(Message::ToggleTranscript),
        text(format!("{dot} {}", status.line())).size(12).color(color),
    ]
    .spacing(6);
    if state.settings.transcript_on {
        let (lines, partial) = state.handles.transcript.snapshot(500);
        let all: Vec<(u32, String)> = lines.iter().map(|l| (l.at_secs, l.text.clone())).collect();
        let shown = ui_logic::filter_lines(&all, &state.transcript_query);
        let mut body = column![].spacing(2);
        if all.is_empty() && partial.is_empty() {
            body = body.push(text("Say something - your words appear here.").size(12).style(text::secondary));
        } else if shown.is_empty() && partial.is_empty() {
            body = body.push(text("No lines match your search.").size(12).style(text::secondary));
        }
        for (secs, t) in shown {
            body = body.push(text(format!("[{}] {}", asr_onnx::pipeline::fmt_time(*secs), t)).size(13));
        }
        if !partial.is_empty() && state.transcript_query.trim().is_empty() {
            body = body.push(text(format!("... {partial}")).size(13).style(text::secondary));
        }
        col = col
            .push(text_input("Search transcript", &state.transcript_query).on_input(Message::TranscriptSearch))
            .push(container(scrollable(body).height(Length::Fixed(260.0)).anchor_bottom()).padding(6).width(Length::Fill))
            .push(
                row![
                    button(text("Copy").size(12)).on_press(Message::CopyTranscript),
                    button(text("Export").size(12)).on_press(Message::ExportTranscript),
                    button(text("Open folder").size(12)).on_press(Message::OpenTranscriptFolder),
                    button(text("Clear").size(12)).on_press(Message::ClearTranscript),
                ]
                .spacing(8),
            )
            .push(
                text("Saved automatically for this call")
                .size(11)
                .style(text::secondary),
            );
    }
    if let Some(note) = &state.transcript_note {
        col = col.push(text(note.as_str()).size(12).style(text::secondary));
    }
    card(col)
}

fn devices_tab(state: &MainState) -> Element<'_, Message> {
    let mic = column![
        row![
            container(card_title("Microphone path", "Your real microphone in, cleaned audio out")).width(Length::Fill),
            button(text("Refresh devices").size(12)).on_press(Message::RefreshDevices),
        ]
        .align_y(iced::Alignment::Center),
        text("Plugged in a headset or started VB-Cable? Refresh to find it.").size(12).style(text::secondary),
        device_picker_of(
            "Your real microphone",
            crate::device_rules::only_virtual(&state.devices.capture, false),
            &state.settings.physical_mic_device_id,
            Message::SelectPhysicalMic,
        ),
        device_picker_of(
            "Send cleaned audio to",
            crate::device_rules::only_virtual(&state.devices.render, true),
            &state.settings.virtual_mic_target_id,
            Message::SelectMicTarget,
        ),
        text("In Zoom, Teams or your dialer, set the microphone to CABLE Output.").size(12).style(text::secondary),
    ]
    .spacing(8);

    let call = column![
        card_title("Call audio path", "Cleans what the caller says"),
        row![
            container(text("This needs a second virtual cable (VB-CABLE A+B), separate from the microphone's.").size(12).style(text::secondary))
                .width(Length::Fill),
            button(text("Install guide").size(12)).on_press(Message::OpenInstallGuide),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
        device_picker_of(
            "Capture call audio from",
            crate::device_rules::call_source_choices(&state.devices, state.settings.virtual_mic_target_id.as_deref()),
            &state.settings.virtual_speaker_source_id,
            Message::SelectSpeakerSource,
        ),
        device_picker_of(
            "Play cleaned audio on",
            crate::device_rules::only_virtual(&state.devices.render, false),
            &state.settings.real_speaker_output_device_id,
            Message::SelectRealSpeakerOutput,
        ),
        text("In your call app, set the speaker to CABLE-A Input.").size(12).style(text::secondary),
    ]
    .spacing(8);

    let monitor = column![
        card_title("Monitor my processed mic", "For testing only. Plays your cleaned voice back to you"),
        toggler(state.handles.monitor.is_on()).label("Hear my processed mic").on_toggle(Message::ToggleMonitor),
        device_picker(
            "Play it on (for example your headphones)",
            &state.devices.render,
            &state.settings.monitor_device_id,
            Message::SelectMonitorDevice,
        ),
    ]
    .spacing(8);

    column![card(mic), card(call), card(monitor)].spacing(14).into()
}

fn advanced_tab(state: &MainState) -> Element<'_, Message> {
    let h = &state.handles;
    let selected = BvcBackend::from_index(h.bvc_backend.load());

    let mut low_cpu = column![
        row![
            button(text("Low CPU mode").size(13)).on_press(Message::EnableLowCpuMode),
            text("Pauses background voice removal and Studio for the mic and call audio. Noise removal stays on.").size(12).style(text::secondary),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
    ]
    .spacing(6);
    if state.show_low_cpu_suggestion {
        low_cpu = low_cpu.push(
            text("\u{26A0} Your PC is struggling to keep up. Try Low CPU mode.")
                .size(12)
                .style(text::warning),
        );
    }

    let mut model = column![
        text("Voice removal model").size(14),
        pick_list(&BVC_BACKEND_NAMES[..], Some(selected.name()), Message::SelectBvcBackend),
    ]
    .spacing(6);
    if let Some(n) = bvc_note(state, true) {
        model = model.push(n);
    }
    // The call-audio model's technical reason belongs here too (its plain message is on the Call audio tab).
    let spk_reason = match tse_engine::SpeakerBvcBackend::from_index(h.speaker_bvc_backend.load()) {
        tse_engine::SpeakerBvcBackend::Hush if !h.speaker_bvc_available => h.speaker_bvc_unavailable_reason.clone(),
        tse_engine::SpeakerBvcBackend::DeepFilter if !h.speaker_deep_filter_available => {
            h.speaker_deep_filter_unavailable_reason.clone()
        }
        _ => None,
    };
    if let Some(reason) = spk_reason {
        model = model.push(text(format!("Call audio model unavailable: {reason}")).size(12).style(text::danger));
    }

    let system = column![
        toggler(state.settings.start_with_windows).label("Start with Windows").on_toggle(Message::ToggleStartWithWindows),
        toggler(state.settings.minimize_to_tray).label("Minimize to tray").on_toggle(Message::ToggleMinimizeToTray),
        text("Closing the window keeps ClearNAI running in the tray.").size(12).style(text::secondary),
        toggler(state.settings.theme_dark).label("Dark theme").on_toggle(Message::ToggleDarkTheme),
        text("Mute hotkey: Ctrl+Shift+M").size(12).style(text::secondary),
    ]
    .spacing(8);

    let io = state.io_counters_snapshot;
    let sio = state.speaker_io_counters_snapshot;
    let accent_line = if state.settings.accent_on {
        let s = h.accent.status();
        format!("{:.2} / {:.0} ms", s.rtf, s.latency_ms)
    } else {
        "off".to_string()
    };
    let cpu_line = state.cpu_percent.map_or("measuring...".to_string(), |p| format!("{p:.0} %"));
    let status_dot_color = status_color(state.status);
    let diag_row = |label: &'static str, value: String| -> Element<'_, Message> {
        row![container(text(label).size(12).style(text::secondary)).width(Length::Fill), text(value).size(12)].into()
    };
    let diagnostics = column![
        text("Diagnostics").size(14),
        row![
            text("\u{25CF}").size(14).color(Color::from_rgb(status_dot_color.r, status_dot_color.g, status_dot_color.b)),
            text(format!("Engine: {}", status_label(state.status))).size(12),
        ]
        .spacing(6),
        diag_row("Mic dropped frames", io.dropped_frames.to_string()),
        diag_row("Mic input overruns", io.input_overruns.to_string()),
        diag_row("Mic output underruns", io.output_underruns.to_string()),
        diag_row("Accent speed / added delay", accent_line),
        diag_row("Call audio underruns", sio.output_underruns.to_string()),
        diag_row("CPU (this app, all cores)", cpu_line),
        row![
            button(text("Export logs").size(12)).on_press(Message::ExportLogs),
            button(text(if state.reset_armed { "Click again to reset everything" } else { "Reset to defaults" }).size(12))
                .on_press(Message::ResetDefaults)
                .style(move |t, s| if state.reset_armed { button::danger(t, s) } else { button::secondary(t, s) }),
        ]
        .spacing(8),
    ]
    .spacing(6);

    let mut col = column![low_cpu, model, system, diagnostics].spacing(18);
    if let Some(note) = &state.diag_note {
        col = col.push(text(note.as_str()).size(12).style(text::secondary));
    }
    card(col)
}
