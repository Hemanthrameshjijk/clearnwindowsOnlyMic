//! The ClearNAI look (taken from the product's web page): a cool grey page, white cards with a thin line and 12 px
//! corners, a blue accent, soft pill toggles and a calm dark variant. Everything the window draws takes its colours from
//! [`Tokens`], so light and dark stay consistent.

use iced::widget::{button as button_widget, container, pick_list, progress_bar, text_input, toggler, Button};
use iced::{Border, Color, Element, Shadow, Theme, Vector};

#[derive(Clone, Copy, Debug)]
pub struct Tokens {
    pub page: Color,
    pub surface: Color,
    pub soft: Color,
    pub line: Color,
    pub line_strong: Color,
    pub text: Color,
    pub muted: Color,
    pub muted_strong: Color,
    pub primary: Color,
    pub primary_soft: Color,
    pub success: Color,
    pub warning: Color,
    pub danger: Color,
}

const fn rgb(v: u32) -> Color {
    Color::from_rgb(((v >> 16) & 0xFF) as f32 / 255.0, ((v >> 8) & 0xFF) as f32 / 255.0, (v & 0xFF) as f32 / 255.0)
}

pub const fn tokens_for(dark: bool) -> Tokens {
    if dark {
        Tokens {
            page: rgb(0x161A22),
            surface: rgb(0x20252F),
            soft: rgb(0x272D38),
            line: rgb(0x363E4C),
            line_strong: rgb(0x4B5667),
            text: rgb(0xF2F5FB),
            muted: rgb(0xA1ADBF),
            muted_strong: rgb(0xC3CCDA),
            primary: rgb(0x6175FF),
            primary_soft: rgb(0x2A3355),
            success: rgb(0x63E0B4),
            warning: rgb(0xF6C457),
            danger: rgb(0xE5707E),
        }
    } else {
        Tokens {
            page: rgb(0xF3F6FB),
            surface: rgb(0xFFFFFF),
            soft: rgb(0xF7F9FD),
            line: rgb(0xDFE5EF),
            line_strong: rgb(0xCBD4E2),
            text: rgb(0x17213D),
            muted: rgb(0x70809D),
            muted_strong: rgb(0x53617C),
            primary: rgb(0x2563EB),
            primary_soft: rgb(0xEAF1FF),
            success: rgb(0x15936D),
            warning: rgb(0xDF9A22),
            danger: rgb(0xB64B59),
        }
    }
}

pub fn tokens(theme: &Theme) -> Tokens {
    tokens_for(theme.extended_palette().is_dark)
}

pub fn theme(dark: bool) -> Theme {
    let t = tokens_for(dark);
    Theme::custom(
        if dark { "ClearNAI Dark" } else { "ClearNAI Light" },
        iced::theme::Palette { background: t.page, text: t.text, primary: t.primary, success: t.success, warning: t.warning, danger: t.danger },
    )
}

/// A white (or dark grey) card with a hairline border, 12 px corners and a barely-there shadow.
pub fn card(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(t.surface.into()),
        border: Border { color: t.line, width: 1.0, radius: 12.0.into() },
        shadow: Shadow { color: Color::from_rgba(0.10, 0.17, 0.30, 0.04), offset: Vector::new(0.0, 4.0), blur_radius: 14.0 },
        ..Default::default()
    }
}

/// The left navigation column.
pub fn sidebar(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.surface.into()), border: Border { color: t.line, width: 1.0, radius: 0.0.into() }, ..Default::default() }
}

fn rounded(mut style: button_widget::Style, radius: f32) -> button_widget::Style {
    style.border.radius = radius.into();
    style
}

/// The default button of the app: blue, 10 px corners.
pub fn primary(theme: &Theme, status: button_widget::Status) -> button_widget::Style {
    rounded(button_widget::primary(theme, status), 10.0)
}

pub fn strong(theme: &Theme, status: button_widget::Status) -> button_widget::Style {
    let t = tokens(theme);
    let hovered = matches!(status, button_widget::Status::Hovered | button_widget::Status::Pressed);
    let base = if theme.extended_palette().is_dark { t.primary } else { rgb(0x17213D) };
    let lift = if hovered { 0.12 } else { 0.0 };
    button_widget::Style {
        background: Some(Color { r: (base.r + lift).min(1.0), g: (base.g + lift).min(1.0), b: (base.b + lift).min(1.0), a: 1.0 }.into()),
        text_color: Color::WHITE,
        border: Border { radius: 10.0.into(), ..Default::default() },
        ..Default::default()
    }
}

pub fn danger(theme: &Theme, status: button_widget::Status) -> button_widget::Style {
    rounded(button_widget::danger(theme, status), 10.0)
}

pub fn text_button(theme: &Theme, status: button_widget::Status) -> button_widget::Style {
    rounded(button_widget::text(theme, status), 10.0)
}

/// A sidebar entry: the current page is tinted blue, the rest are quiet text.
pub fn nav_item(selected: bool) -> impl Fn(&Theme, button_widget::Status) -> button_widget::Style {
    move |theme, status| {
        let t = tokens(theme);
        let hovered = matches!(status, button_widget::Status::Hovered | button_widget::Status::Pressed);
        button_widget::Style {
            background: if selected {
                Some(t.primary_soft.into())
            } else if hovered {
                Some(t.soft.into())
            } else {
                None
            },
            text_color: if selected { t.primary } else { t.muted_strong },
            border: Border { radius: 10.0.into(), ..Default::default() },
            ..Default::default()
        }
    }
}

/// `iced::widget::button` with the app's default look already applied (later `.style(..)` calls still override it).
pub fn button<'a, M: Clone + 'a>(content: impl Into<Element<'a, M>>) -> Button<'a, M> {
    button_widget(content).style(primary)
}

/// Pill switch: 22 px tall, blue when on, grey when off, white knob.
pub fn toggle(theme: &Theme, status: toggler::Status) -> toggler::Style {
    let t = tokens(theme);
    let (is_on, disabled) = match status {
        toggler::Status::Active { is_toggled } | toggler::Status::Hovered { is_toggled } => (is_toggled, false),
        toggler::Status::Disabled { is_toggled } => (is_toggled, true),
    };
    let off = if theme.extended_palette().is_dark { rgb(0x4B5667) } else { rgb(0xCBD3DF) };
    let mut track = if is_on { t.primary } else { off };
    if disabled {
        track.a = 0.45;
    }
    toggler::Style {
        background: track.into(),
        background_border_width: 0.0,
        background_border_color: Color::TRANSPARENT,
        foreground: Color::WHITE.into(),
        foreground_border_width: 0.0,
        foreground_border_color: Color::TRANSPARENT,
        text_color: Some(t.text),
        border_radius: None,
        padding_ratio: 0.14,
    }
}

/// A dropdown that looks like the product page's select: soft fill, hairline border, 8 px corners.
pub fn select(theme: &Theme, status: pick_list::Status) -> pick_list::Style {
    let t = tokens(theme);
    let hovered = matches!(status, pick_list::Status::Hovered | pick_list::Status::Opened { .. });
    pick_list::Style {
        text_color: t.text,
        placeholder_color: t.muted,
        handle_color: t.muted,
        background: t.soft.into(),
        border: Border { color: if hovered { t.line_strong } else { t.line }, width: 1.0, radius: 9.0.into() },
    }
}

/// A dropdown with no frame at all (used inside a framed card).
pub fn select_bare(theme: &Theme, _status: pick_list::Status) -> pick_list::Style {
    let t = tokens(theme);
    pick_list::Style {
        text_color: t.text,
        placeholder_color: t.muted,
        handle_color: t.muted,
        background: Color::TRANSPARENT.into(),
        border: Border { color: Color::TRANSPARENT, width: 0.0, radius: 8.0.into() },
    }
}

/// The search box: no frame of its own (the card around it draws it).
pub fn bare_input(theme: &Theme, _status: text_input::Status) -> text_input::Style {
    let t = tokens(theme);
    text_input::Style {
        background: Color::TRANSPARENT.into(),
        border: Border { color: Color::TRANSPARENT, width: 0.0, radius: 0.0.into() },
        icon: t.muted,
        placeholder: t.muted,
        value: t.text,
        selection: Color { a: 0.3, ..t.primary },
    }
}

/// A framed field: soft fill, hairline border, 9 px corners.
pub fn field(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(t.soft.into()),
        border: Border { color: t.line, width: 1.0, radius: 9.0.into() },
        ..Default::default()
    }
}

/// A flat soft box (the info strip inside a card).
pub fn soft_box(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.soft.into()), border: Border { radius: 9.0.into(), ..Default::default() }, ..Default::default() }
}

/// A thin horizontal rule colour.
pub fn rule_line(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style { background: Some(t.line.into()), ..Default::default() }
}

/// An outline button (white / dark surface, hairline border).
pub fn outline(theme: &Theme, status: button_widget::Status) -> button_widget::Style {
    let t = tokens(theme);
    let hovered = matches!(status, button_widget::Status::Hovered | button_widget::Status::Pressed);
    button_widget::Style {
        background: Some(if hovered { t.soft } else { t.surface }.into()),
        text_color: if matches!(status, button_widget::Status::Disabled) { t.muted } else { t.text },
        border: Border { color: if hovered { t.line_strong } else { t.line }, width: 1.0, radius: 10.0.into() },
        ..Default::default()
    }
}

/// A selectable tile (Noise / Background voices / ...): tinted with a blue outline when on.
pub fn tile(on: bool) -> impl Fn(&Theme, button_widget::Status) -> button_widget::Style {
    move |theme, status| {
        let t = tokens(theme);
        let hovered = matches!(status, button_widget::Status::Hovered | button_widget::Status::Pressed);
        button_widget::Style {
            background: Some(if on { t.primary_soft } else if hovered { t.soft } else { t.surface }.into()),
            text_color: if on { t.primary } else { t.muted_strong },
            border: Border { color: if on { t.primary } else { t.line }, width: 1.0, radius: 10.0.into() },
            ..Default::default()
        }
    }
}

/// A key cap (Ctrl, Shift, M).
pub fn keycap(theme: &Theme) -> container::Style {
    let t = tokens(theme);
    container::Style {
        background: Some(t.soft.into()),
        border: Border { color: t.line_strong, width: 1.0, radius: 6.0.into() },
        shadow: Shadow { color: Color::from_rgba(0.0, 0.0, 0.0, 0.08), offset: Vector::new(0.0, 1.0), blur_radius: 0.0 },
        ..Default::default()
    }
}

/// Soft tinted square behind a card's icon: (background, glyph) per accent.
#[derive(Clone, Copy)]
pub enum Accent {
    Blue,
    Amber,
    Green,
    Purple,
}

pub fn accent_colors(dark: bool, accent: Accent) -> (Color, Color) {
    match (accent, dark) {
        (Accent::Blue, false) => (rgb(0xEAF1FF), rgb(0x2563EB)),
        (Accent::Blue, true) => (rgb(0x2A3355), rgb(0x8EA0FF)),
        (Accent::Amber, false) => (rgb(0xFFF3D7), rgb(0xAA7117)),
        (Accent::Amber, true) => (rgb(0x423518), rgb(0xF6C457)),
        (Accent::Green, false) => (rgb(0xE2FAF2), rgb(0x07936D)),
        (Accent::Green, true) => (rgb(0x183B36), rgb(0x63E0B4)),
        (Accent::Purple, false) => (rgb(0xF0EAFF), rgb(0x8C5CE0)),
        (Accent::Purple, true) => (rgb(0x3A2F5C), rgb(0xB79BFF)),
    }
}

/// A flat surface-coloured strip (top bar, status bar).
pub fn surface(theme: &Theme) -> container::Style {
    container::Style { background: Some(tokens(theme).surface.into()), ..Default::default() }
}

/// The level meter: soft track, solid accent fill, fully rounded.
pub fn meter(theme: &Theme) -> progress_bar::Style {
    let t = tokens(theme);
    progress_bar::Style { background: t.line.into(), bar: t.primary.into(), border: Border { radius: 6.0.into(), ..Default::default() } }
}
