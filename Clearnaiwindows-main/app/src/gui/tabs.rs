//! The tabbed full window (Home / Microphone / Call audio / Transcript / Devices / Advanced) and the compact bar.
//! Every control here already existed in the old single-page `view_main`; this only reorganises them and adds the
//! header chip, Mute button, profile picker, transcript tools and the Advanced diagnostics. All messages go through
//! the same `update_main` handlers as before.

use super::{MainState, Message, BVC_BACKEND_NAMES, SPEAKER_BVC_BACKEND_NAMES};
use crate::shared::{studio_preset_from_index, studio_preset_name, STUDIO_PRESET_NAMES};
use crate::status_ui::{status_color, status_label};
use crate::ui_logic::{self, MicSnapshot, StatusChip, Tab};
use super::style::{self, button as btn};
use iced::widget::{button, column, container, image, pick_list, progress_bar, row, scrollable, svg, text, text_input, toggler};
use crate::shared::DeviceOption;
use dsp_core::RealtimeStatus;
use iced::{Color, Element, Length};
use tse_engine::BvcBackend;

fn grey() -> Color {
    Color::from_rgb(0.55, 0.55, 0.55)
}

fn red() -> Color {
    Color::from_rgb(0.85, 0.2, 0.2)
}

const LOGO_MARK_PNG: &[u8] = include_bytes!("../../assets/logo_mark.png");
const LOGO_WORD_LIGHT_PNG: &[u8] = include_bytes!("../../assets/logo_word_light.png");
const LOGO_WORD_DARK_PNG: &[u8] = include_bytes!("../../assets/logo_word_dark.png");
/// Width / height of the wordmark pictures (700 x 190).
const WORD_ASPECT: f32 = 700.0 / 190.0;

// `image::Handle::from_bytes` makes a new texture id on every call, so each picture is built once and cloned.
static MARK_HANDLE: std::sync::OnceLock<image::Handle> = std::sync::OnceLock::new();
static WORD_LIGHT_HANDLE: std::sync::OnceLock<image::Handle> = std::sync::OnceLock::new();
static WORD_DARK_HANDLE: std::sync::OnceLock<image::Handle> = std::sync::OnceLock::new();

/// The ClearNAI mark (the C with the sound bars).
pub fn logo_mark<'a>(_dark: bool, size: f32) -> Element<'a, Message> {
    image(MARK_HANDLE.get_or_init(|| image::Handle::from_bytes(LOGO_MARK_PNG)).clone()).width(size).height(size).into()
}

/// The mark plus the "ClearNAI" wordmark: navy and blue on light, white and blue on dark.
pub fn logo_with_wordmark<'a>(dark: bool, size: f32) -> Element<'a, Message> {
    let handle = if dark {
        WORD_DARK_HANDLE.get_or_init(|| image::Handle::from_bytes(LOGO_WORD_DARK_PNG))
    } else {
        WORD_LIGHT_HANDLE.get_or_init(|| image::Handle::from_bytes(LOGO_WORD_LIGHT_PNG))
    };
    image(handle.clone()).width(size * WORD_ASPECT).height(size).into()
}

/// A line icon tinted with `color`.
fn glyph_icon<'a>(bytes: &'static [u8], size: f32, color: Color) -> Element<'a, Message> {
    svg(svg::Handle::from_memory(bytes)).width(size).height(size).style(move |_, _| svg::Style { color: Some(color) }).into()
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
    container(
        row![text("\u{25CF}").size(8).color(fg), text(label).size(11).font(iced::Font { weight: iced::font::Weight::Bold, ..iced::Font::DEFAULT })]
            .spacing(6)
            .align_y(iced::Alignment::Center),
    )
    .padding([4, 11])
    .style(move |_| container::Style {
        background: Some(bg.into()),
        text_color: Some(fg),
        border: iced::Border::default().rounded(14),
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

fn amber_text(dark: bool) -> Color {
    if dark { Color::from_rgb8(0xF6, 0xC4, 0x57) } else { Color::from_rgb8(0xA3, 0x6B, 0x12) }
}

fn bold() -> iced::Font {
    iced::Font { weight: iced::font::Weight::Bold, ..iced::Font::DEFAULT }
}

fn hrule<'a>() -> Element<'a, Message> {
    container(text("")).width(Length::Fill).height(1).style(style::rule_line).into()
}

fn vrule<'a>(height: f32) -> Element<'a, Message> {
    container(text("")).width(1).height(height).style(style::rule_line).into()
}

/// A white card with the page's padding.
fn card<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    container(content).padding(24).width(Length::Fill).style(style::card).into()
}

/// The soft tinted square with an icon at the left of a card heading.
fn badge<'a>(icon: &'static [u8], accent: style::Accent, dark: bool) -> Element<'a, Message> {
    let (bg, fg) = style::accent_colors(dark, accent);
    container(glyph_icon(icon, 19.0, fg))
        .width(38)
        .height(38)
        .center_x(Length::Fixed(38.0))
        .center_y(Length::Fixed(38.0))
        .style(move |_| container::Style {
            background: Some(bg.into()),
            border: iced::Border { radius: 11.0.into(), ..Default::default() },
            ..Default::default()
        })
        .into()
}

/// Icon badge, title (with an optional chip), subtitle, and an optional control at the right.
fn card_head<'a>(
    state: &MainState,
    icon: &'static [u8],
    accent: style::Accent,
    title: &'a str,
    chip_el: Option<Element<'a, Message>>,
    subtitle: &'a str,
    trailing: Option<Element<'a, Message>>,
) -> Element<'a, Message> {
    let t = style::tokens_for(state.settings.theme_dark);
    let mut title_row = row![text(title).size(18)].spacing(10).align_y(iced::Alignment::Center);
    if let Some(c) = chip_el {
        title_row = title_row.push(c);
    }
    let mut head = row![
        badge(icon, accent, state.settings.theme_dark),
        column![title_row, text(subtitle).size(13).color(t.muted)].spacing(4).width(Length::Fill),
    ]
    .spacing(14)
    .align_y(iced::Alignment::Center);
    if let Some(tr) = trailing {
        head = head.push(tr);
    }
    head.into()
}

/// One setting: title and description at the left, a pill switch at the right.
fn setting_row<'a>(
    state: &MainState,
    title: &'a str,
    desc: Option<&'a str>,
    on: bool,
    msg: impl Fn(bool) -> Message + 'a,
) -> Element<'a, Message> {
    let t = style::tokens_for(state.settings.theme_dark);
    let mut left = column![text(title).size(15)].spacing(4).width(Length::Fill);
    if let Some(d) = desc {
        left = left.push(text(d).size(13).color(t.muted));
    }
    row![left, toggler(on).on_toggle(msg).size(24).style(style::toggle)].spacing(16).align_y(iced::Alignment::Center).into()
}

/// A setting the engine does not have yet: same look, dimmed, no handler.
fn disabled_setting_row<'a>(state: &MainState, title: &'a str, desc: &'a str, on: bool) -> Element<'a, Message> {
    let t = style::tokens_for(state.settings.theme_dark);
    let row_el = row![
        column![text(title).size(15).color(t.muted), text(desc).size(13).color(t.muted)].spacing(4).width(Length::Fill),
        toggler(on).size(24).style(style::toggle),
    ]
    .spacing(16)
    .align_y(iced::Alignment::Center);
    tip(row_el.into(), "Not available yet", state.settings.theme_dark)
}

/// "Label  [select]" on one line.
fn select_row<'a>(state: &MainState, label: &'a str, control: Element<'a, Message>) -> Element<'a, Message> {
    let t = style::tokens_for(state.settings.theme_dark);
    row![text(label).size(14).color(t.muted_strong), control].spacing(14).align_y(iced::Alignment::Center).into()
}

fn disabled_select<'a>(state: &MainState, label: &'a str, value: &'a str) -> Element<'a, Message> {
    let t = style::tokens_for(state.settings.theme_dark);
    let field = container(
        row![text(value).size(13).color(t.muted), glyph_icon(super::icons::CHEVRON_DOWN, 14.0, t.muted)]
            .spacing(24)
            .align_y(iced::Alignment::Center),
    )
    .padding([9, 12])
    .style(style::field);
    tip(select_row(state, label, field.into()), "Not adjustable yet", state.settings.theme_dark)
}

/// A device dropdown; `selected_id == None` shows `placeholder` (the system default).
fn device_select<'a>(
    options: Vec<DeviceOption>,
    selected_id: &Option<String>,
    placeholder: &'a str,
    on: fn(DeviceOption) -> Message,
) -> Element<'a, Message> {
    let selected = selected_id.as_ref().and_then(|id| options.iter().find(|d| &d.id == id).cloned());
    pick_list(options, selected, on)
        .placeholder(placeholder)
        .width(Length::Fill)
        .padding([9, 12])
        .text_size(13)
        .style(style::select)
        .into()
}

fn icon_button<'a>(icon: &'static [u8], label: &'a str, msg: Message, st: fn(&iced::Theme, button::Status) -> button::Style, tint: Color) -> Element<'a, Message> {
    btn(row![glyph_icon(icon, 16.0, tint), text(label).size(13).font(bold())].spacing(8).align_y(iced::Alignment::Center))
        .padding([10, 16])
        .on_press(msg)
        .style(st)
        .into()
}

fn grid2<'a>(a: Element<'a, Message>, b: Element<'a, Message>, wide: bool) -> Element<'a, Message> {
    if wide {
        row![container(a).width(Length::FillPortion(1)), container(b).width(Length::FillPortion(1))].spacing(18).into()
    } else {
        column![a, b].spacing(18).into()
    }
}

/// Below this width the sidebar would squeeze the content, so the tabs move to the top.
const WIDE_MIN: f32 = 760.0;

/// Below this window width side-by-side cards and two-column rows stack, so nothing is squeezed or clipped.
const GRID_MIN: f32 = 960.0;

/// The full window. Wide: sidebar, top bar, page, status bar (like the product page). Narrow: the same pages under a
/// row of tabs, so it stays usable at any window size.
pub fn view_full(state: &MainState) -> Element<'_, Message> {
    iced::widget::responsive(move |size| {
        if size.width >= WIDE_MIN {
            wide_layout(state, size.width >= GRID_MIN, size.width)
        } else {
            narrow_layout(state, size.width)
        }
    })
    .into()
}

fn tab_body(state: &MainState, wide: bool, win_w: f32) -> Element<'_, Message> {
    match state.tab {
        Tab::Home => home_tab(state, wide, win_w),
        Tab::Microphone => microphone_tab(state, wide),
        Tab::CallAudio => call_audio_tab(state),
        Tab::Transcript => transcript_tab(state),
        Tab::Devices => devices_tab(state, wide),
        Tab::Advanced => advanced_tab(state, wide),
    }
}

/// (eyebrow, headline, intro) of each page - the same words as the product page.
fn page_text(tab: Tab) -> (&'static str, &'static str, &'static str) {
    match tab {
        Tab::Home => ("AUDIO CONTROL CENTER", "Good audio starts here.", "Tune your voice and call experience before you join the conversation."),
        Tab::Microphone => ("MICROPHONE", "Shape what others hear.", "Clean up your voice before it reaches your meeting or call."),
        Tab::CallAudio => ("CALL AUDIO", "Make the caller easier to hear.", "Control incoming audio cleanup separately from your microphone."),
        Tab::Transcript => ("TRANSCRIPT", "Keep the conversation in view.", "Search and save a live transcript from your calls."),
        Tab::Devices => ("DEVICES", "Route your audio clearly.", "Choose which devices ClearNAI should listen to and play through."),
        Tab::Advanced => ("ADVANCED", "Fine-tune your workspace.", "Choose the models, shortcuts, and system behavior that fit your setup."),
    }
}

/// The small second line under each sidebar entry.
fn nav_hint(tab: Tab) -> &'static str {
    match tab {
        Tab::Home => "Audio overview",
        Tab::Microphone => "Voice processing",
        Tab::CallAudio => "Incoming sound",
        Tab::Transcript => "Live captions",
        Tab::Devices => "Audio routing",
        Tab::Advanced => "Fine-tune ClearNAI",
    }
}

fn nav_icon(tab: Tab) -> &'static [u8] {
    use super::icons;
    match tab {
        Tab::Home => icons::HOME,
        Tab::Microphone => icons::MIC,
        Tab::CallAudio => icons::PHONE,
        Tab::Transcript => icons::DOC,
        Tab::Devices => icons::DEVICES,
        Tab::Advanced => icons::SLIDERS,
    }
}

/// The status chip (or button) at the right end of a page heading.
fn page_action(state: &MainState) -> Element<'_, Message> {
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    match state.tab {
        Tab::Home => status_dot_label("System ready", t.success, t.muted_strong),
        Tab::Microphone => {
            let (label, kind) = match current_chip(state) {
                StatusChip::Muted => ("Microphone muted", ChipKind::Bad),
                StatusChip::Paused => ("Microphone paused", ChipKind::Off),
                _ => ("Microphone active", ChipKind::Ok),
            };
            chip(label, kind, dark)
        }
        Tab::CallAudio => {
            if speaker_ready(state) {
                chip("Running smoothly", ChipKind::Ok, dark)
            } else {
                chip("Not set up", ChipKind::Warn, dark)
            }
        }
        Tab::Transcript => {
            if state.settings.transcript_on {
                chip("Listening", ChipKind::Ok, dark)
            } else {
                chip("Off", ChipKind::Off, dark)
            }
        }
        Tab::Devices => icon_button(super::icons::REFRESH, "Refresh devices", Message::RefreshDevices, style::outline, t.muted_strong),
        Tab::Advanced => chip(if state.settings.profile.is_empty() { "Custom profile" } else { state.settings.profile.as_str() }, ChipKind::Off, dark),
    }
}

/// A green (or amber) dot followed by a label.
fn status_dot_label<'a>(label: &'a str, dot: Color, ink: Color) -> Element<'a, Message> {
    row![text("\u{25CF}").size(11).color(dot), text(label).size(13).color(ink)].spacing(7).align_y(iced::Alignment::Center).into()
}

/// The bar above every page: call-audio state at the left; profile and "collapse to bar" at the right.
fn top_bar(state: &MainState) -> Element<'_, Message> {
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    let left = if speaker_ready(state) { chip("Call audio ready", ChipKind::Ok, dark) } else { chip("Call audio not set up", ChipKind::Warn, dark) };
    let selected_profile = ui_logic::active_profile(&state.settings.profile, &mic_snapshot(state));
    let bar = row![
        left,
        container(text("")).width(Length::Fill),
        glyph_icon(super::icons::USER, 16.0, t.muted),
        text(selected_profile.unwrap_or("Custom")).size(13).color(t.muted_strong),
        pick_list(&ui_logic::PROFILE_NAMES[..], selected_profile, Message::SelectProfile)
            .placeholder("Custom")
            .width(Length::Fixed(112.0))
            .padding([9, 12])
            .text_size(13)
            .style(style::select),
        btn(container(glyph_icon(super::icons::EXPAND, 16.0, t.muted_strong)).center_x(Length::Fill).center_y(Length::Fill))
            .width(36)
            .height(36)
            .padding(0)
            .on_press(Message::ToggleCompact)
            .style(style::outline),
    ]
    .spacing(10)
    .align_y(iced::Alignment::Center);
    column![
        container(bar).padding([0, 34]).width(Length::Fill).height(70).align_y(iced::alignment::Vertical::Center).style(style::surface),
        hrule(),
    ]
    .into()
}

/// The strip under every page: CPU, audio delay, health and the mute hotkey.
fn status_bar(state: &MainState) -> Element<'_, Message> {
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    let cpu = state.cpu_percent.map_or("--".to_string(), |p| format!("{p:.0}%"));
    let delay = if state.settings.accent_on {
        let ms = state.handles.accent.status().latency_ms;
        if ms > 0.0 { format!("~{:.1} s", ms / 1000.0) } else { "~0.5 s".to_string() }
    } else {
        "<20 ms".to_string()
    };
    let (health, health_color) = if !state.alerts.is_empty() {
        ("Needs attention", t.warning)
    } else {
        match state.status {
            RealtimeStatus::Overloaded => ("Overloaded", t.danger),
            RealtimeStatus::Degraded => ("Degraded", t.warning),
            _ => ("System healthy", t.success),
        }
    };
    let small = |s: String| text(s).size(11).color(t.muted);
    let strong = |s: String| text(s).size(11).color(t.muted_strong).font(bold());
    let bar = row![
        small("CPU".into()),
        strong(cpu),
        vrule(14.0),
        small("Audio delay".into()),
        strong(delay),
        container(text("")).width(Length::Fill),
        glyph_icon(super::icons::PULSE, 14.0, health_color),
        small(health.into()),
        vrule(14.0),
        strong("Ctrl + Shift + M".into()),
    ]
    .spacing(10)
    .align_y(iced::Alignment::Center);
    column![hrule(), container(bar).padding([0, 34]).width(Length::Fill).height(48).align_y(iced::alignment::Vertical::Center).style(style::surface)].into()
}

fn sidebar_row<'a>(icon: &'static [u8], label: &'a str, right: Option<String>, msg: Message, t: style::Tokens) -> Element<'a, Message> {
    let mut r = row![glyph_icon(icon, 17.0, t.muted_strong), text(label).size(13), container(text("")).width(Length::Fill)]
        .spacing(12)
        .align_y(iced::Alignment::Center);
    if let Some(s) = right {
        r = r.push(text(s).size(11).color(t.muted));
    }
    btn(r).width(Length::Fill).padding([9, 10]).on_press(msg).style(style::nav_item(false)).into()
}

fn wide_layout(state: &MainState, grid: bool, win_w: f32) -> Element<'_, Message> {
    use super::icons;
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    let selected_profile = ui_logic::active_profile(&state.settings.profile, &mic_snapshot(state));
    let profile_name = selected_profile.unwrap_or("Custom");

    // Workspace card = the profile picker.
    let initial = profile_name.chars().next().unwrap_or('C').to_uppercase().to_string();
    let avatar = container(text(initial).size(13).color(Color::WHITE).font(bold()))
        .width(28)
        .height(28)
        .center_x(Length::Fixed(28.0))
        .center_y(Length::Fixed(28.0))
        .style(move |_| container::Style {
            background: Some(Color::from_rgb8(0x15, 0x2E, 0x64).into()),
            border: iced::Border { radius: 8.0.into(), ..Default::default() },
            ..Default::default()
        });
    let workspace = container(
        row![
            avatar,
            column![
                pick_list(&ui_logic::PROFILE_NAMES[..], selected_profile, Message::SelectProfile)
                    .placeholder("Custom")
                    .width(Length::Fill)
                    .padding(0)
                    .text_size(13)
                    .font(bold())
                    .style(style::select_bare),
                text("Workspace").size(11).color(t.muted),
            ]
            .spacing(1)
            .width(Length::Fill),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center),
    )
    .padding([10, 12])
    .width(Length::Fill)
    .style(style::card);

    let mut nav = column![].spacing(4);
    for tab in Tab::ALL {
        let current = tab == state.tab;
        let ink = if current { t.primary } else { t.muted_strong };
        let mut entry = row![
            glyph_icon(nav_icon(tab), 19.0, ink),
            column![text(tab.label()).size(14).font(bold()), text(nav_hint(tab)).size(11).color(if current { t.primary } else { t.muted })].spacing(2),
            container(text("")).width(Length::Fill),
        ]
        .spacing(13)
        .align_y(iced::Alignment::Center);
        if current {
            entry = entry.push(glyph_icon(icons::CHEVRON_RIGHT, 15.0, t.primary));
        }
        nav = nav.push(btn(entry).width(Length::Fill).padding([10, 12]).on_press(Message::SelectTab(tab)).style(style::nav_item(current)));
    }

    let theme_row = sidebar_row(
        if dark { icons::SUN } else { icons::MOON },
        if dark { "Light mode" } else { "Dark mode" },
        Some(if dark { "On" } else { "Off" }.to_string()),
        Message::ToggleDarkTheme(!dark),
        t,
    );
    let profile_row = btn(
        row![
            container(glyph_icon(icons::USER, 17.0, t.primary))
                .width(36)
                .height(36)
                .center_x(Length::Fixed(36.0))
                .center_y(Length::Fixed(36.0))
                .style(move |_| container::Style {
                    background: Some(t.primary_soft.into()),
                    border: iced::Border { radius: 18.0.into(), ..Default::default() },
                    ..Default::default()
                }),
            column![text("Profile").size(16).font(bold()), text("Personal settings").size(11).color(t.muted)].spacing(1),
            container(text("")).width(Length::Fill),
            glyph_icon(icons::CHEVRON_RIGHT, 15.0, t.muted),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center),
    )
    .width(Length::Fill)
    .padding([8, 10])
    .on_press(Message::SelectTab(Tab::Advanced))
    .style(style::nav_item(false));

    let top = column![
        container(logo_with_wordmark(dark, 34.0)).padding([6, 8]),
        workspace,
        container(text("WORKSPACE").size(11).color(t.muted).font(bold())).padding([10, 12]),
        nav,
    ]
    .spacing(8);
    let bottom = column![
        hrule(),
        sidebar_row(icons::HELP, "Help center", None, Message::OpenInstallGuide, t),
        theme_row,
        hrule(),
        profile_row,
    ]
    .spacing(6);
    // Short windows: the navigation scrolls, Help center / Dark mode / Profile never leave the screen.
    let sidebar = container(column![scrollable(top).height(Length::Fill), bottom].spacing(8))
        .padding([16, 14])
        .width(Length::Fixed(254.0))
        .height(Length::Fill)
        .style(style::sidebar);

    let (eyebrow, title, intro) = page_text(state.tab);
    let header = row![
        column![text(eyebrow).size(11).color(t.muted_strong).font(bold()), text(title).size(36), text(intro).size(14).color(t.muted)]
            .spacing(8)
            .width(Length::Fill),
        container(page_action(state)).align_y(iced::alignment::Vertical::Bottom).height(Length::Fixed(88.0)),
    ]
    .spacing(12)
    .align_y(iced::Alignment::End);

    let page = column![header, alert_banner(state), tab_body(state, grid, win_w), warning_banner(state)].spacing(22).padding([30, 30]);
    row![
        sidebar,
        column![top_bar(state), scrollable(page).width(Length::Fill).height(Length::Fill), status_bar(state)].width(Length::Fill).height(Length::Fill),
    ]
    .into()
}

/// The engine's start-up warnings as the page's amber banner: bold title, details, and (for call-audio setup) a chevron to
/// the Devices page. Warnings read "Title - details".
fn warning_banner(state: &MainState) -> Element<'_, Message> {
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    let ink = amber_text(dark);
    let mut col = column![].spacing(10);
    for w in &state.warnings {
        let (title, rest) = ui_logic::split_warning(w);
        let mark = container(text("!").size(12).color(Color::WHITE).font(bold()))
            .width(20)
            .height(20)
            .center_x(Length::Fixed(20.0))
            .center_y(Length::Fixed(20.0))
            .style(move |_| container::Style {
                background: Some(t.warning.into()),
                border: iced::Border { radius: 10.0.into(), ..Default::default() },
                ..Default::default()
            });
        let mut body = column![].spacing(3).width(Length::Fill);
        if let Some(title) = title {
            body = body.push(text(title.to_string()).size(14).color(ink).font(bold()));
        }
        body = body.push(text(rest).size(14).color(ink));
        let mut line = row![mark, body].spacing(12).align_y(iced::Alignment::Start);
        if w.contains("speaker-source") || w.contains("virtual") {
            line = line.push(
                btn(glyph_icon(super::icons::CHEVRON_RIGHT, 16.0, t.muted))
                    .padding([2, 8])
                    .on_press(Message::SelectTab(Tab::Devices))
                    .style(style::text_button),
            );
        }
        col = col.push(container(line).padding([4, 4]).width(Length::Fill));
    }
    col.into()
}

fn narrow_layout(state: &MainState, win_w: f32) -> Element<'_, Message> {
    let dark = state.settings.theme_dark;
    let chip_now = current_chip(state);
    let selected_profile = ui_logic::active_profile(&state.settings.profile, &mic_snapshot(state));
    let header = row![
        logo_with_wordmark(dark, 30.0),
        chip(chip_now.label(), chip_kind(chip_now), dark),
        container(text("")).width(Length::Fill),
        text("Profile").size(12),
        pick_list(&ui_logic::PROFILE_NAMES[..], selected_profile, Message::SelectProfile).placeholder("Custom"),
        btn(container(glyph_icon(super::icons::EXPAND, 16.0, style::tokens_for(dark).muted_strong)).center_x(Length::Fill).center_y(Length::Fill))
            .width(34)
            .height(34)
            .padding(0)
            .on_press(Message::ToggleCompact)
            .style(style::outline),
    ]
    .spacing(10)
    .align_y(iced::Alignment::Center)
    .padding([14, 18]);

    let mut tab_bar = row![].spacing(2);
    for t in Tab::ALL {
        let current = t == state.tab;
        tab_bar = tab_bar.push(
            btn(text(t.label()).size(13))
                .padding([8, 10])
                .on_press(Message::SelectTab(t))
                .style(move |th, s| if current { style::primary(th, s) } else { style::text_button(th, s) }),
        );
    }

    let body = tab_body(state, false, win_w);

    column![
        header,
        alert_banner(state),
        container(tab_bar).padding([0, 12]),
        iced::widget::rule::horizontal(1),
        scrollable(column![body, warning_banner(state)].spacing(18).padding(18)).height(Length::Fill),
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

/// Height of the see-through strip above the collapsed bar, where hover names are drawn (a window can only draw
/// inside itself). `gui::COMPACT_WINDOW_SIZE` includes it.
pub const TIP_STRIP: f32 = 28.0;

/// A hover name for a control in the collapsed bar.
fn tip<'a>(content: Element<'a, Message>, label: impl Into<String>, dark: bool) -> Element<'a, Message> {
    let bg = if dark { Color::from_rgb8(0x0B, 0x0E, 0x14) } else { Color::from_rgb8(0x1F, 0x25, 0x33) };
    iced::widget::tooltip(
        content,
        container(text(label.into()).size(12).color(Color::WHITE)).padding([4, 8]).style(move |_| container::Style {
            background: Some(bg.into()),
            border: iced::Border { radius: 6.0.into(), ..Default::default() },
            ..Default::default()
        }),
        iced::widget::tooltip::Position::Top,
    )
    .gap(4)
    .into()
}

/// The collapsed bar (a Krisp-style horizontal pill): logo, microphone and call-audio status with the level meter, four
/// quick switches (Noise, Voices, Accent, Transcript), Mute, Settings and Expand. Undecorated and draggable; Expand opens
/// the full window in the middle of the screen.
pub fn view_compact(state: &MainState) -> Element<'_, Message> {
    use super::icons;
    let dark = state.settings.theme_dark;
    let h = &state.handles;
    let tk = style::tokens_for(dark);
    let (ink, blue, blue_wash, green, orange) = (tk.text, tk.primary, tk.primary_soft, tk.success, tk.warning);
    let (panel_bg, line, button_bg) = (tk.surface, tk.line, tk.soft);

    fn glyph<'a>(bytes: &'static [u8], size: f32, color: Color) -> Element<'a, Message> {
        svg(svg::Handle::from_memory(bytes))
            .width(size)
            .height(size)
            .style(move |_, _| svg::Style { color: Some(color) })
            .into()
    }

    // An icon with a small status dot at its lower right.
    let with_dot = |bytes: &'static [u8], dot: Color| -> Element<'_, Message> {
        let badge = container(text("")).width(8).height(8).style(move |_| container::Style {
            background: Some(dot.into()),
            border: iced::Border { color: panel_bg, width: 1.5, radius: 6.0.into() },
            ..Default::default()
        });
        iced::widget::stack![
            container(glyph(bytes, 17.0, ink)).padding([0, 2]),
            container(badge).width(Length::Fill).height(Length::Fill).align_x(iced::alignment::Horizontal::Right).align_y(iced::alignment::Vertical::Bottom),
        ]
        .width(22)
        .height(22)
        .into()
    };

    // A square icon switch: highlighted with a blue outline while on.
    let switch = |bytes: &'static [u8], on: bool, msg: Message| -> Element<'_, Message> {
        btn(container(glyph(bytes, 16.0, if on { blue } else { ink })).center_x(Length::Fill).center_y(Length::Fill))
            .width(32)
            .height(32)
            .padding(0)
            .on_press(msg)
            .style(move |_, status| {
                let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                button::Style {
                    background: if on {
                        Some(blue_wash.into())
                    } else if hovered {
                        Some(button_bg.into())
                    } else {
                        None
                    },
                    border: iced::Border { color: if on { blue } else { Color::TRANSPARENT }, width: 1.5, radius: 9.0.into() },
                    ..Default::default()
                }
            })
            .into()
    };

    let plain = |bytes: &'static [u8], msg: Message| -> Element<'_, Message> {
        btn(container(glyph(bytes, 16.0, ink)).center_x(Length::Fill).center_y(Length::Fill))
            .width(28)
            .height(28)
            .padding(0)
            .on_press(msg)
            .style(move |_, status| button::Style {
                background: matches!(status, button::Status::Hovered | button::Status::Pressed).then(|| button_bg.into()),
                border: iced::Border { radius: 10.0.into(), ..Default::default() },
                ..Default::default()
            })
            .into()
    };

    let muted = !h.mic_master_enabled.is_on();
    let mic_dot = match current_chip(state) {
        StatusChip::Muted => red(),
        StatusChip::Paused => grey(),
        _ => green,
    };
    let speaker_dot = if speaker_ready(state) { green } else { orange };
    let divider = || container(text("")).width(1).height(22).style(move |_| container::Style { background: Some(line.into()), ..Default::default() });

    let mute = btn(
        row![glyph(icons::MIC, 14.0, if muted { red() } else { ink }), text(if muted { "Unmute" } else { "Mute" }).size(12).color(if muted { red() } else { ink })]
            .spacing(5)
            .align_y(iced::Alignment::Center),
    )
    .padding([6, 11])
    .on_press(Message::ToggleMute)
    .style(move |_, status| button::Style {
        background: Some(if matches!(status, button::Status::Hovered | button::Status::Pressed) { line } else { button_bg }.into()),
        border: iced::Border { radius: 10.0.into(), ..Default::default() },
        ..Default::default()
    });

    let bar = row![
        tip(logo_mark(dark, 24.0), "ClearNAI", dark),
        divider(),
        tip(with_dot(icons::MIC, mic_dot), format!("Microphone: {}", current_chip(state).label()), dark),
        tip(
            progress_bar(0.0..=1.0, state.level.clamp(0.0, 1.0)).length(Length::Fixed(48.0)).girth(Length::Fixed(5.0)).into(),
            "Microphone level",
            dark,
        ),
        tip(
            with_dot(icons::SPEAKER, speaker_dot),
            if speaker_ready(state) { "Call audio: ready" } else { "Call audio: needs a second virtual cable" },
            dark,
        ),
        divider(),
        tip(switch(icons::WAVE, h.mic_noise.is_on(), Message::ToggleMicNoise(!h.mic_noise.is_on())), "Noise removal", dark),
        tip(switch(icons::PEOPLE, h.mic_bvc.is_on(), Message::ToggleMicBvc(!h.mic_bvc.is_on())), "Background voices", dark),
        tip(switch(icons::TRANSLATE, state.settings.accent_on, Message::ToggleAccent(!state.settings.accent_on)), "Accent conversion", dark),
        tip(switch(icons::CAPTIONS, state.settings.transcript_on, Message::ToggleTranscript(!state.settings.transcript_on)), "Live transcript", dark),
        tip(mute.into(), "Mute or unmute the microphone (Ctrl+Shift+M)", dark),
        tip(plain(icons::GEAR, Message::OpenFullAt(Tab::Advanced)), "Settings", dark),
        tip(plain(icons::EXPAND, Message::ToggleCompact), "Open the full window", dark),
        tip(plain(icons::CLOSE, Message::WindowClosed), "Close (ClearNAI keeps running in the tray)", dark),
    ]
    .spacing(4)
    .align_y(iced::Alignment::Center);

    // An alert (accent paused, engine lost...) shows as a thin line under the controls.
    // The bar is one row high, so a problem (accent paused, engine lost...) shows as a red "!" at its right end; the
    // full window has the details.
    let inner = if state.alerts.is_empty() {
        bar
    } else {
        bar.push(
            container(text("!").size(13).color(Color::WHITE))
                .width(20)
                .height(20)
                .center_x(Length::Fixed(20.0))
                .center_y(Length::Fixed(20.0))
                .style(move |_| container::Style {
                    background: Some(red().into()),
                    border: iced::Border { radius: 10.0.into(), ..Default::default() },
                    ..Default::default()
                }),
        )
    };

    let pill = container(inner).padding([6, 11]).style(move |_| container::Style {
        background: Some(panel_bg.into()),
        border: iced::Border { color: line, width: 1.0, radius: 15.0.into() },
        shadow: iced::Shadow { color: Color::from_rgba(0.0, 0.0, 0.0, 0.18), offset: iced::Vector::new(0.0, 3.0), blur_radius: 10.0 },
        ..Default::default()
    });
    // Press anywhere that is not a control to drag the (undecorated) window.
    iced::widget::mouse_area(column![container(text("")).height(Length::Fixed(TIP_STRIP)), container(pill).padding(4)])
        .on_press(Message::DragWindow)
        .into()
}

fn home_tab(state: &MainState, wide: bool, win_w: f32) -> Element<'_, Message> {
    use super::icons;
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
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

    let percent = |v: f32| format!("{}%", (v.clamp(0.0, 1.0) * 100.0).round() as u32);
    let meter_row = |label: &'static str, v: f32| -> Element<'_, Message> {
        row![
            text(label).size(14).color(t.muted_strong).width(Length::Fixed(58.0)),
            progress_bar(0.0..=1.0, v.clamp(0.0, 1.0)).girth(Length::Fixed(9.0)).style(style::meter),
            text(percent(v)).size(13).color(t.muted).width(Length::Fixed(44.0)).align_x(iced::alignment::Horizontal::Right),
        ]
        .spacing(12)
        .align_y(iced::Alignment::Center)
        .into()
    };
    let signal = column![
        row![
            text("LIVE AUDIO SIGNAL").size(11).color(t.muted_strong).font(bold()),
            container(text("")).width(Length::Fill),
            status_dot_label(if mic_enabled { "Monitoring" } else { "Muted" }, if mic_enabled { t.success } else { t.danger }, t.muted_strong),
        ]
        .align_y(iced::Alignment::Center),
        // Before = raw mic peak (`EngineHandles::input_level`), After = peak of the processed frame
        // (`EngineHandles::level`, taken after the whole chain).
        meter_row("Before", state.input_level),
        meter_row("After", state.level),
    ]
    .spacing(14);

    let tile = |icon: &'static [u8], label: &'static str, on: bool, msg: Message| -> Element<'_, Message> {
        let ink = if on { t.primary } else { t.muted_strong };
        let mut r = row![glyph_icon(icon, 16.0, ink), text(label).size(13).font(bold()).width(Length::Fill)].spacing(8).width(Length::Fill).align_y(iced::Alignment::Center);
        if on {
            r = r.push(glyph_icon(icons::CHECK, 15.0, t.primary));
        }
        btn(container(r).width(Length::Fill).center_y(Length::Fill)).width(Length::FillPortion(1)).height(46).padding([0, 12]).on_press(msg).style(style::tile(on)).into()
    };
    let t_noise = tile(icons::WAVES, "Noise", noise, Message::ToggleMicNoise(!noise));
    let t_voices = tile(icons::HEADPHONES, "Background voices", voices, Message::ToggleMicBvc(!voices));
    let t_accent = tile(icons::SPARKLES, "Accent", accent, Message::ToggleAccent(!accent));
    let t_studio = tile(icons::SLIDERS, "Studio", studio, Message::ToggleMicStudio(!studio));
    // Side by side only when the card is wide enough; otherwise a 2 x 2 grid so no label is squeezed.
    // Width left inside the card: the sidebar layout loses the sidebar and the page padding, the tab layout less.
    let inner = (if win_w >= WIDE_MIN { win_w - 254.0 - 60.0 } else { win_w - 36.0 }) - 48.0;
    let tiles: Element<'_, Message> = if !wide && inner >= 480.0 {
        row![t_noise, t_voices, t_accent, t_studio].spacing(10).into()
    } else {
        column![row![t_noise, t_voices].spacing(10), row![t_accent, t_studio].spacing(10)].spacing(10).into()
    };
    let modes = column![text("ENHANCEMENT MODE").size(11).color(t.muted_strong).font(bold()), tiles].spacing(14);

    let muted = !mic_enabled;
    let mute = btn(
        container(
            row![glyph_icon(icons::MIC, 16.0, Color::WHITE), text(if muted { "Unmute mic" } else { "Mute mic" }).size(14).font(bold())]
                .spacing(10)
                .align_y(iced::Alignment::Center),
        )
        .center_x(Length::Fill)
        .center_y(Length::Fill),
    )
    .width(Length::FillPortion(3))
    .height(46)
    .on_press(Message::ToggleMute)
    .style(move |th, s| if muted { style::danger(th, s) } else { style::strong(th, s) });
    // The engine plays only the processed signal to the monitor; there is no before/after switch yet.
    let hear = tip(
        btn(
            container(
                row![glyph_icon(icons::SPEAKER, 16.0, t.muted), text("Hear before / after").size(14).font(bold()).color(t.muted)]
                    .spacing(10)
                    .align_y(iced::Alignment::Center),
            )
            .center_x(Length::Fill)
            .center_y(Length::Fill),
        )
        .width(Length::FillPortion(2))
        .height(46)
        .style(style::outline)
        .into(),
        "Not available yet. Use Devices > Monitor to hear your processed voice.",
        dark,
    );

    let columns: Element<'_, Message> = if wide {
        row![container(signal).width(Length::FillPortion(5)), container(modes).width(Length::FillPortion(6))].spacing(40).into()
    } else {
        column![signal, modes].spacing(20).into()
    };
    let mut mic_col = column![
        card_head(
            state,
            icons::MIC,
            style::Accent::Blue,
            "Your microphone",
            Some(chip(mic_chip.0, mic_chip.1, dark)),
            "What others hear on the call",
            Some(toggler(mic_enabled).on_toggle(|_| Message::ToggleMute).size(26).style(style::toggle).into()),
        ),
        hrule(),
        columns,
        hrule(),
        row![mute, hear].spacing(14),
    ]
    .spacing(20);
    if accent {
        mic_col = mic_col.push(
            row![
                glyph_icon(icons::SPARKLES, 15.0, amber_text(dark)),
                text("Accent conversion adds about 0.5 s of delay.").size(13).color(amber_text(dark)),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center),
        );
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
    let setup_btn: Option<Element<'_, Message>> = if ready {
        None
    } else {
        Some(icon_button(icons::SLIDERS, "Set up", Message::SelectTab(Tab::Devices), style::primary, Color::WHITE))
    };
    let mut spk_col = column![card_head(state, icons::PHONE, style::Accent::Amber, "Call audio", Some(chip(spk_chip.0, spk_chip.1, dark)), "What you hear from the caller", setup_btn)]
        .spacing(18);
    if !ready {
        let (abg, afg) = style::accent_colors(dark, style::Accent::Amber);
        spk_col = spk_col.push(
            row![
                container(glyph_icon(icons::SPEAKER_BOX, 16.0, afg))
                    .width(30)
                    .height(30)
                    .center_x(Length::Fixed(30.0))
                    .center_y(Length::Fixed(30.0))
                    .style(move |_| container::Style {
                        background: Some(abg.into()),
                        border: iced::Border { radius: 8.0.into(), ..Default::default() },
                        ..Default::default()
                    }),
                text("Incoming audio isn't being cleaned yet. Set up a virtual cable to get started.").size(13).color(t.muted_strong),
            ]
            .spacing(12)
            .align_y(iced::Alignment::Center),
        );
    } else {
        spk_col = spk_col.push(setting_row(
            state,
            "Clean the caller's side",
            Some("Noise and background voices on what you hear"),
            spk_enabled,
            Message::ToggleSpeakerMasterEnabled,
        ));
    }

    column![card(mic_col), card(spk_col)].spacing(20).into()
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

/// Select and its status line: side by side when there is room, stacked otherwise.
fn accent_row<'a>(wide: bool, select: Element<'a, Message>, line: Element<'a, Message>) -> Element<'a, Message> {
    if wide {
        row![select, line].spacing(18).align_y(iced::Alignment::Center).into()
    } else {
        column![select, line].spacing(8).into()
    }
}

fn microphone_tab(state: &MainState, wide: bool) -> Element<'_, Message> {
    let h = &state.handles;
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);

    let mut col = column![setting_row(state, "Noise removal", Some("Fans, typing, traffic"), h.mic_noise.is_on(), Message::ToggleMicNoise)].spacing(26);
    col = col.push(setting_row(
        state,
        "Background voice removal",
        Some("Removes other people talking near you. Choose the model in Advanced."),
        h.mic_bvc.is_on(),
        Message::ToggleMicBvc,
    ));
    if let Some(note) = bvc_note(state, false) {
        col = col.push(note);
    }
    col = col
        .push(setting_row(state, "Quiet nearby voices", Some("Fast. Lowers voices quieter than yours."), h.mic_voice_gate.is_on(), Message::ToggleMicVoiceGate))
        .push(setting_row(
            state,
            "Echo removal",
            Some("Removes call audio from your speakers out of your mic. Not needed with headphones."),
            h.echo_cancel.is_on(),
            Message::ToggleEchoCancel,
        ))
        .push(setting_row(state, "Accent conversion", Some("Indian \u{2192} American English \u{00B7} adds about 0.5 s delay"), state.settings.accent_on, Message::ToggleAccent));

    // What the accent stage is REALLY doing (the switch only records the click): green = converting, amber = starting,
    // red = on but not converting, grey = off.
    let status = h.accent.status();
    let label = accent_onnx::VOICE_CHOICES[(state.settings.accent_voice_index() - 1) as usize];
    let accent_color = match status.state {
        accent_onnx::pipeline::AccentState::Active => t.success,
        accent_onnx::pipeline::AccentState::Loading | accent_onnx::pipeline::AccentState::Starting => t.warning,
        accent_onnx::pipeline::AccentState::Paused | accent_onnx::pipeline::AccentState::Unavailable => t.danger,
        accent_onnx::pipeline::AccentState::Off => t.muted,
    };
    col = col.push(accent_row(
            wide,
            select_row(
                state,
                "Accent voice",
                pick_list(&accent_onnx::VOICE_CHOICES[..], Some(label), Message::SelectAccentVoice).padding([9, 12]).text_size(13).style(style::select).into(),
            ),
            text(status.line()).size(12).color(accent_color).into(),
    ));
    col = col.push(setting_row(state, "Studio voice", Some("Fuller, clearer tone"), h.mic_studio.is_on(), Message::ToggleMicStudio));
    col = col.push(select_row(
        state,
        "Studio preset",
        pick_list(
            &STUDIO_PRESET_NAMES[..],
            Some(studio_preset_name(studio_preset_from_index(h.mic_studio_preset.load()))),
            Message::SelectMicStudioPreset,
        )
        .padding([9, 12])
        .text_size(13)
        .style(style::select)
        .into(),
    ));
    card(col)
}

fn call_audio_tab(state: &MainState) -> Element<'_, Message> {
    let h = &state.handles;
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    let selected = tse_engine::SpeakerBvcBackend::from_index(h.speaker_bvc_backend.load());
    let note: Option<Element<'_, Message>> = match selected {
        tse_engine::SpeakerBvcBackend::Hush if !h.speaker_bvc_available => {
            Some(text("Background voice removal isn't available right now. Try another model.").size(12).style(text::danger).into())
        }
        tse_engine::SpeakerBvcBackend::DeepFilter if !h.speaker_deep_filter_available => {
            Some(text("Background voice removal isn't available right now. Try another model.").size(12).style(text::danger).into())
        }
        _ => None,
    };
    let ready = speaker_ready(state);
    let mut col = column![].spacing(22);
    if !ready {
        col = col.push(text("Call audio isn't set up yet. Choose your devices in the Devices tab.").size(13).color(amber_text(dark)));
    }
    let status_text = if ready { format!("Call audio: {}", status_label(state.speaker_status)) } else { "Call audio: Waiting for a source device".to_string() };
    col = col
        .push(row![text("\u{25CF}").size(10).color(t.success), text(status_text).size(15)].spacing(8).align_y(iced::Alignment::Center))
        .push(setting_row(state, "Call audio path enabled", None, h.speaker_master_enabled.is_on(), Message::ToggleSpeakerMasterEnabled))
        .push(setting_row(state, "Noise removal", None, h.speaker_noise.is_on(), Message::ToggleSpeakerNoise))
        .push(select_row(
            state,
            "Background voice model",
            pick_list(&SPEAKER_BVC_BACKEND_NAMES[..], Some(selected.name()), Message::SelectSpeakerBvcBackend)
                .padding([9, 12])
                .text_size(13)
                .style(style::select)
                .into(),
        ));
    if let Some(n) = note {
        col = col.push(n);
    }
    col = col
        .push(setting_row(state, "Studio voice", None, h.speaker_studio.is_on(), Message::ToggleSpeakerStudio))
        .push(select_row(
            state,
            "Studio preset",
            pick_list(
                &STUDIO_PRESET_NAMES[..],
                Some(studio_preset_name(studio_preset_from_index(h.speaker_studio_preset.load()))),
                Message::SelectSpeakerStudioPreset,
            )
            .padding([9, 12])
            .text_size(13)
            .style(style::select)
            .into(),
        ));
    if !ready {
        col = col.push(btn(text("Set up call audio").size(14).font(bold())).padding([12, 20]).on_press(Message::SelectTab(Tab::Devices)).style(style::primary));
    }
    container(card(col)).max_width(900).into()
}

fn transcript_tab(state: &MainState) -> Element<'_, Message> {
    use super::icons;
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    let status = state.handles.transcript.status();
    let color = match status.state {
        asr_onnx::pipeline::TranscriptState::Listening | asr_onnx::pipeline::TranscriptState::Transcribing => t.success,
        asr_onnx::pipeline::TranscriptState::Loading => t.warning,
        asr_onnx::pipeline::TranscriptState::Unavailable => t.danger,
        asr_onnx::pipeline::TranscriptState::Off => t.muted,
    };
    let mut col = column![row![
        column![text("Live transcript").size(16).font(bold()), text("Saved automatically for this call").size(13).color(t.muted)].spacing(4),
        toggler(state.settings.transcript_on).on_toggle(Message::ToggleTranscript).size(26).style(style::toggle),
        container(text("")).width(Length::Fill),
        text("\u{25CF}").size(11).color(color),
        text(status.line()).size(13).color(color),
    ]
    .spacing(24)
    .align_y(iced::Alignment::Center)]
    .spacing(20);
    if state.settings.transcript_on {
        let (lines, partial) = state.handles.transcript.snapshot(500);
        let all: Vec<(u32, String)> = lines.iter().map(|l| (l.at_secs, l.text.clone())).collect();
        let shown = ui_logic::filter_lines(&all, &state.transcript_query);
        let mut body = column![].spacing(7);
        if all.is_empty() && partial.is_empty() {
            body = body.push(text("Say something - your words appear here.").size(14).color(t.muted));
        } else if shown.is_empty() && partial.is_empty() {
            body = body.push(text("No lines match your search.").size(14).color(t.muted));
        }
        for (secs, line) in shown {
            body = body.push(text(format!("[{}] {}", asr_onnx::pipeline::fmt_time(*secs), line)).size(14));
        }
        if !partial.is_empty() && state.transcript_query.trim().is_empty() {
            body = body.push(text(format!("... {partial}")).size(14).color(t.muted));
        }
        let search = container(
            row![
                glyph_icon(icons::SEARCH, 16.0, t.muted),
                text_input("Search transcript", &state.transcript_query).on_input(Message::TranscriptSearch).padding(0).size(14).style(style::bare_input),
            ]
            .spacing(12)
            .align_y(iced::Alignment::Center),
        )
        .padding([13, 16])
        .width(Length::Fill)
        .style(style::field);
        col = col
            .push(search)
            .push(container(scrollable(body).height(Length::Fixed(300.0)).anchor_bottom()).padding([0, 6]).width(Length::Fill))
            .push(
                row![
                    icon_button(icons::COPY, "Copy", Message::CopyTranscript, style::primary, Color::WHITE),
                    icon_button(icons::DOWNLOAD, "Export", Message::ExportTranscript, style::primary, Color::WHITE),
                    icon_button(icons::FOLDER, "Open folder", Message::OpenTranscriptFolder, style::primary, Color::WHITE),
                    icon_button(icons::TRASH, "Clear", Message::ClearTranscript, style::danger, Color::WHITE),
                ]
                .spacing(10),
            )
            .push(text("Saved automatically for this call").size(12).color(t.muted));
    }
    if let Some(note) = &state.transcript_note {
        col = col.push(text(note.as_str()).size(12).color(t.muted));
    }
    card(col)
}

/// A few small bars that follow the microphone level (a live "signal detected" mark).
fn level_bars<'a>(level: f32, color: Color) -> Element<'a, Message> {
    let mut r = row![].spacing(3).align_y(iced::Alignment::Center);
    for (i, w) in [0.45f32, 0.8, 1.0, 0.7, 0.5].iter().enumerate() {
        let lit = (level * 6.0 * w).clamp(0.0, 1.0);
        let height = 6.0 + 16.0 * lit.max(0.15) * (1.0 - 0.05 * i as f32);
        r = r.push(container(text("")).width(3).height(height).style(move |_| container::Style {
            background: Some(color.into()),
            border: iced::Border { radius: 2.0.into(), ..Default::default() },
            ..Default::default()
        }));
    }
    r.into()
}

fn devices_tab(state: &MainState, wide: bool) -> Element<'_, Message> {
    use super::icons;
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    let ready = speaker_ready(state);
    let have_devices = !state.devices.capture.is_empty() || !state.devices.render.is_empty();
    let ok_mark = |show: bool| -> Option<Element<'_, Message>> { show.then(|| glyph_icon(icons::CHECK_CIRCLE, 20.0, t.success)) };

    let signal = state.input_level > 0.02;
    let mic_card = card(
        column![
            card_head(state, icons::MIC, style::Accent::Blue, "Microphone input", None, "What others hear from you", ok_mark(have_devices)),
            select_row(
                state,
                "Input device",
                device_select(
                    crate::device_rules::only_virtual(&state.devices.capture, false),
                    &state.settings.physical_mic_device_id,
                    "Default microphone",
                    Message::SelectPhysicalMic,
                ),
            ),
            hrule(),
            row![
                glyph_icon(icons::PULSE, 15.0, if signal { t.success } else { t.muted }),
                text(if signal { "Signal detected" } else { "No signal yet" }).size(13).color(if signal { t.success } else { t.muted }),
                container(text("")).width(Length::Fill),
                level_bars(state.input_level, t.primary),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(22),
    );
    let out_card = card(
        column![
            card_head(state, icons::SPEAKER, style::Accent::Green, "Speaker output", None, "What you hear from the caller", ok_mark(have_devices)),
            select_row(
                state,
                "Output device",
                device_select(
                    crate::device_rules::only_virtual(&state.devices.render, false),
                    &state.settings.real_speaker_output_device_id,
                    "Default speakers",
                    Message::SelectRealSpeakerOutput,
                ),
            ),
            hrule(),
            row![
                glyph_icon(icons::SPEAKER, 15.0, t.success),
                text("Output ready").size(13).color(t.success),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(22),
    );

    let capture_chip = if ready { chip("Selected", ChipKind::Ok, dark) } else { chip("Not selected", ChipKind::Warn, dark) };
    let capture = card(
        column![
            card_head(
                state,
                icons::SPEAKER_BOX,
                style::Accent::Amber,
                "Speaker-source capture",
                None,
                "Independent virtual cable for incoming call cleanup",
                Some(capture_chip),
            ),
            select_row(
                state,
                "Capture endpoint",
                device_select(
                    crate::device_rules::call_source_choices(&state.devices, state.settings.virtual_mic_target_id.as_deref()),
                    &state.settings.virtual_speaker_source_id,
                    "Not selected",
                    Message::SelectSpeakerSource,
                ),
            ),
            container(
                row![
                    glyph_icon(icons::HELP, 16.0, t.muted),
                    text("Use a separate virtual cable from your microphone input so the caller audio stays isolated.").size(13).color(t.muted_strong).width(Length::Fill),
                    btn(text("Install guide").size(12)).padding([7, 12]).on_press(Message::OpenInstallGuide).style(style::outline),
                ]
                .spacing(12)
                .align_y(iced::Alignment::Center),
            )
            .padding([13, 16])
            .width(Length::Fill)
            .style(style::soft_box),
        ]
        .spacing(22),
    );

    // Not on the product page but needed by the app: where the cleaned voice goes, and the test monitor.
    let virtual_mic = card(
        column![
            card_head(state, icons::MIC, style::Accent::Purple, "Virtual microphone", None, "Where ClearNAI sends your cleaned voice", None),
            select_row(
                state,
                "Send cleaned audio to",
                device_select(
                    crate::device_rules::only_virtual(&state.devices.render, true),
                    &state.settings.virtual_mic_target_id,
                    "Not selected",
                    Message::SelectMicTarget,
                ),
            ),
            container(text("In Zoom, Teams or your dialer, set the microphone to CABLE Output.").size(13).color(t.muted_strong))
                .padding([13, 16])
                .width(Length::Fill)
                .style(style::soft_box),
        ]
        .spacing(22),
    );
    let monitor = card(
        column![
            card_head(state, icons::HEADPHONES, style::Accent::Blue, "Monitor my processed mic", None, "For testing only. Plays your cleaned voice back to you", None),
            setting_row(state, "Hear my processed mic", None, state.handles.monitor.is_on(), Message::ToggleMonitor),
            select_row(
                state,
                "Play it on",
                device_select(state.devices.render.clone(), &state.settings.monitor_device_id, "Not selected", Message::SelectMonitorDevice),
            ),
        ]
        .spacing(22),
    );

    column![grid2(mic_card, out_card, wide), capture, grid2(virtual_mic, monitor, wide)].spacing(20).into()
}

fn advanced_tab(state: &MainState, wide: bool) -> Element<'_, Message> {
    use super::icons;
    let dark = state.settings.theme_dark;
    let t = style::tokens_for(dark);
    let h = &state.handles;
    let selected = BvcBackend::from_index(h.bvc_backend.load());

    let mut processing = column![
        card_head(state, icons::SLIDERS, style::Accent::Purple, "Processing", None, "Balance quality and responsiveness.", None),
        select_row(
            state,
            "Background voice model",
            pick_list(&BVC_BACKEND_NAMES[..], Some(selected.name()), Message::SelectBvcBackend).padding([9, 12]).text_size(13).style(style::select).into(),
        ),
    ]
    .spacing(18);
    if let Some(n) = bvc_note(state, true) {
        processing = processing.push(n);
    }
    // The call-audio model's technical reason belongs here too (its plain message is on the Call audio tab).
    let spk_reason = match tse_engine::SpeakerBvcBackend::from_index(h.speaker_bvc_backend.load()) {
        tse_engine::SpeakerBvcBackend::Hush if !h.speaker_bvc_available => h.speaker_bvc_unavailable_reason.clone(),
        tse_engine::SpeakerBvcBackend::DeepFilter if !h.speaker_deep_filter_available => h.speaker_deep_filter_unavailable_reason.clone(),
        _ => None,
    };
    if let Some(reason) = spk_reason {
        processing = processing.push(text(format!("Call audio model unavailable: {reason}")).size(12).style(text::danger));
    }
    processing = processing
        .push(disabled_select(state, "Accent conversion quality", "Balanced"))
        .push(disabled_select(state, "Processing priority", "Normal"))
        .push(
            row![
                btn(text("Low CPU mode").size(13)).padding([9, 14]).on_press(Message::EnableLowCpuMode).style(style::outline),
                text("Pauses background voice removal and Studio. Noise removal stays on.").size(12).color(t.muted).width(Length::Fill),
            ]
            .spacing(12)
            .align_y(iced::Alignment::Center),
        );
    if state.show_low_cpu_suggestion {
        processing = processing.push(text("\u{26A0} Your PC is struggling to keep up. Try Low CPU mode.").size(12).color(amber_text(dark)));
    }

    let behavior = column![
        card_head(state, icons::SHIELD, style::Accent::Blue, "App behavior", None, "Keep ClearNAI ready when you need it.", None),
        setting_row(state, "Start with computer", Some("Open ClearNAI when you sign in."), state.settings.start_with_windows, Message::ToggleStartWithWindows),
        disabled_setting_row(state, "Show notifications", "Get feedback when audio changes.", false),
        disabled_setting_row(state, "Share diagnostics", "Help improve ClearNAI with anonymous data.", false),
    ]
    .spacing(22);

    let cap = |s: &'static str| -> Element<'static, Message> {
        container(text(s).size(11).font(iced::Font::MONOSPACE)).padding([6, 9]).style(style::keycap).into()
    };
    let shortcuts = column![
        card_head(state, icons::CLIPBOARD, style::Accent::Amber, "Keyboard shortcuts", None, "Quick controls while you are on a call.", None),
        row![
            text("Mute microphone").size(14).color(t.muted_strong),
            cap("Ctrl"),
            text("+").size(12).color(t.muted),
            cap("Shift"),
            text("+").size(12).color(t.muted),
            cap("M"),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center),
    ]
    .spacing(20);

    // Diagnostics and reset: not on the product page, kept because they are how problems are found.
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
        row![container(text(label).size(13).color(t.muted_strong)).width(Length::Fill), text(value).size(13)].into()
    };
    let mut diagnostics = column![
        card_head(state, icons::PULSE, style::Accent::Green, "Diagnostics", None, "What the engine is doing right now.", None),
        row![
            text("\u{25CF}").size(14).color(Color::from_rgb(status_dot_color.r, status_dot_color.g, status_dot_color.b)),
            text(format!("Engine: {}", status_label(state.status))).size(13),
        ]
        .spacing(8),
        diag_row("Mic dropped frames", io.dropped_frames.to_string()),
        diag_row("Mic input overruns", io.input_overruns.to_string()),
        diag_row("Mic output underruns", io.output_underruns.to_string()),
        diag_row("Accent speed / added delay", accent_line),
        diag_row("Call audio underruns", sio.output_underruns.to_string()),
        diag_row("CPU (this app, all cores)", cpu_line),
        row![
            btn(text("Export logs").size(13)).padding([9, 14]).on_press(Message::ExportLogs).style(style::outline),
            btn(text(if state.reset_armed { "Click again to reset everything" } else { "Reset to defaults" }).size(13))
                .padding([9, 14])
                .on_press(Message::ResetDefaults)
                .style(move |th, s| if state.reset_armed { style::danger(th, s) } else { style::outline(th, s) }),
        ]
        .spacing(10),
    ]
    .spacing(14);
    if let Some(note) = &state.diag_note {
        diagnostics = diagnostics.push(text(note.as_str()).size(12).color(t.muted));
    }

    column![grid2(card(processing), card(behavior), wide), card(shortcuts), card(diagnostics)].spacing(20).into()
}
