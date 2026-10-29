//! First-run "setup" checks shown before the main pipeline GUI: basic
//! system compatibility (OS version, architecture, COM) and BVC asset
//! (DLL + model) extraction from what is now embedded directly in this
//! binary - see `ensure_bvc_assets_extracted()`.
//!
//! Only the OS-version/COM checks are genuinely Windows-only
//! (`#[cfg(windows)]`); everything else here (the pure version-check math,
//! and the extraction logic) is ordinary portable Rust and builds/tests on
//! any platform, matching the rest of this crate's "portable unless it
//! truly must touch an OS API" pattern (see `gui.rs`/`shared.rs`/
//! `settings.rs`).
//!
//! # Why embedding instead of a next-to-the-exe download
//!
//! Earlier revisions of this module shipped `weya_nc.dll` and the ONNX
//! model bundle as separate files the user (or a packaging script) had to
//! place next to `clearnairt.exe`, with a first-run background HTTP
//! download as a fallback for the model. Real hands-on testing on Windows
//! found this confusing (a `.dll` isn't runnable and double-clicking it
//! does nothing useful) and not "production ready" for a single-binary
//! distribution goal. Both files are now `include_bytes!`-embedded directly
//! into `clearnairt.exe` at build time (see `app/assets/`) and
//! self-extracted into `%LOCALAPPDATA%\ClearNAI` on every launch - fast
//! (already in memory, just a size check + maybe a file write) enough to do
//! synchronously with no progress UI needed, and it means the packaged
//! folder now only needs the one `.exe` for BVC to work at all.
//!
//! # What has and hasn't been verified
//!
//! `check_system_compatibility()`'s real Windows branch (the `RtlGetVersion`
//! call and the COM init check) has never run on real Windows - only
//! type-checked via `cargo check --target x86_64-pc-windows-gnu`. Likewise
//! `ensure_bvc_assets_extracted()`'s real write into `%LOCALAPPDATA%` has
//! never been run on real Windows - only type/link-checked, plus unit
//! tests against an arbitrary temp directory (the extraction logic itself
//! is parameterized on the target directory for exactly this reason).

use std::io::Write;
use std::path::{Path, PathBuf};

/// The real `weya_nc.dll` (see `crates/bvc-hush`), embedded directly into
/// this binary at build time from `app/assets/weya_nc.dll` - a real,
/// unmodified copy built from github.com/pulp-vision/Hush's
/// `native/weya_nc_build/weya_nc` source, confirmed via `objdump` to export
/// exactly the symbols `bvc-hush/src/ffi.rs` binds to.
const EMBEDDED_DLL_BYTES: &[u8] = include_bytes!("../assets/weya_nc.dll");

/// The real ONNX model bundle, embedded directly into this binary at build
/// time from `app/assets/advanced_dfnet16k_model_best_onnx.tar.gz` - a real,
/// unmodified download from
/// <https://huggingface.co/weya-ai/hush/resolve/main/onnx/advanced_dfnet16k_model_best_onnx.tar.gz>
/// (Apache-2.0 licensed), confirmed to contain `enc.onnx`/`erb_dec.onnx`/
/// `df_dec.onnx`/`config.ini`/`version.txt`.
const EMBEDDED_MODEL_BYTES: &[u8] = include_bytes!("../assets/advanced_dfnet16k_model_best_onnx.tar.gz");

/// A real fine-tune of Hush's own upstream checkpoint (pulp-vision/Hush,
/// `deployment/models/model_best.ckpt`), supplied by the user and embedded
/// from `app/assets/bpo_v1_onnx.tar.gz` - see
/// `bvc_hush::BPO_V1_MODEL_BUNDLE_FILE_NAME`'s docs. Same tar.gz layout as
/// `EMBEDDED_MODEL_BYTES` above (loaded via the same
/// `weya_nc_model_load_from_path` call, just a different on-disk filename),
/// extracted alongside it - not a replacement.
const EMBEDDED_BPO_V1_MODEL_BYTES: &[u8] = include_bytes!("../assets/bpo_v1_onnx.tar.gz");

/// The official onnxruntime Windows x64 shared library (v1.19.2, from
/// Microsoft's own `microsoft/onnxruntime` GitHub Releases), embedded
/// directly into this binary from `app/assets/onnxruntime.dll` - needed by
/// the accent-conversion and live-transcript models (the `ort` crate's
/// `load-dynamic` mode). Same self-extraction pattern as
/// `EMBEDDED_DLL_BYTES` above.
const EMBEDDED_ONNXRUNTIME_DLL_BYTES: &[u8] = include_bytes!("../assets/onnxruntime.dll");

/// On-disk file name of the extracted onnxruntime library.
pub const ONNXRUNTIME_DLL_FILE_NAME: &str = "onnxruntime.dll";

/// Result of the one-time, fast system compatibility check run at every
/// launch (cheap enough to always redo, unlike the model download).
#[derive(Debug, Clone, PartialEq)]
pub struct CompatibilityReport {
    pub os_ok: bool,
    pub arch_ok: bool,
    pub com_ok: bool,
    pub details: Vec<String>,
}

impl CompatibilityReport {
    /// True if every individual check passed - callers use this to decide
    /// whether the setup screen needs to be shown at all for compatibility
    /// reasons (a missing model file is a separate, independent trigger).
    pub fn all_ok(&self) -> bool {
        self.os_ok && self.arch_ok && self.com_ok
    }
}

/// Pure logic factored out of the real `RtlGetVersion` call so it can be
/// unit-tested on any platform: Windows 10 and 11 both report major version
/// 10 (there is no separate "11" major version number at the Win32 API
/// level - the two are told apart by build number, which this app does not
/// need to distinguish for a basic compatibility gate).
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_major_version_is_supported(major: u32) -> bool {
    major >= 10
}

#[cfg(windows)]
fn check_os_version(details: &mut Vec<String>) -> bool {
    use windows::Wdk::System::SystemServices::RtlGetVersion;
    use windows::Win32::System::SystemInformation::OSVERSIONINFOEXW;

    // SAFETY: `RtlGetVersion` only reads/writes the `OSVERSIONINFOEXW` we
    // pass it; `dwOSVersionInfoSize` must be set to the struct size first,
    // per the documented contract, or the call fails with STATUS_INVALID_PARAMETER.
    let mut info = OSVERSIONINFOEXW {
        dwOSVersionInfoSize: std::mem::size_of::<OSVERSIONINFOEXW>() as u32,
        ..Default::default()
    };
    let status = unsafe { RtlGetVersion(&mut info as *mut _ as *mut _) };
    if status.is_err() {
        details.push(format!("Failed to query Windows version (RtlGetVersion status {status:?})"));
        return false;
    }
    let major = info.dwMajorVersion;
    let build = info.dwBuildNumber;
    let ok = windows_major_version_is_supported(major);
    if ok {
        details.push(format!("Windows 10/11 detected (build {build})"));
    } else {
        details.push(format!(
            "Unsupported Windows version detected (found major version {major}, need 10+)"
        ));
    }
    ok
}

#[cfg(windows)]
fn check_com(details: &mut Vec<String>) -> bool {
    // Must match `audio_io::initialize_com_for_this_thread`'s apartment
    // choice exactly (STA, not MTA) — this runs on the same main/GUI
    // thread that `winit` (via `iced`) later calls `OleInitialize` on for
    // window drag-and-drop support, and `OleInitialize` hard-requires STA.
    // Real hardware testing hit this directly: initializing MTA here (and
    // in `audio-io`) worked for WASAPI enumeration alone, but crashed the
    // whole app with `OleInitialize failed! RPC_E_CHANGED_MODE` as soon as
    // the GUI window tried to come up, because the thread had already
    // committed to a different (incompatible) apartment model. Calling the
    // same STA init twice on one thread (once here, once in
    // `audio_io::initialize_com_for_this_thread`) is safe — COM reference-
    // counts nested *compatible* `CoInitializeEx`/`OleInitialize` calls.
    let hr = wasapi::initialize_sta();
    if hr.is_ok() {
        details.push("COM (STA) initialized successfully".to_string());
        true
    } else {
        details.push(format!("COM (STA) initialization failed: {hr:?}"));
        false
    }
}

/// Runs the real system compatibility checks. On Windows this queries the
/// real OS version via `RtlGetVersion` (not the manifest-shimmed
/// `GetVersionEx`) and attempts a real COM MTA initialization. On any other
/// platform, returns an honest "not running on Windows" report rather than
/// pretending to have checked anything.
pub fn check_system_compatibility() -> CompatibilityReport {
    #[cfg(windows)]
    {
        let mut details = Vec::new();
        let os_ok = check_os_version(&mut details);
        let arch_ok = cfg!(target_arch = "x86_64");
        details.push(if arch_ok {
            "x86_64 architecture detected (supported)".to_string()
        } else {
            "Unsupported CPU architecture (this build only targets x86_64)".to_string()
        });
        let com_ok = check_com(&mut details);
        CompatibilityReport { os_ok, arch_ok, com_ok, details }
    }
    #[cfg(not(windows))]
    {
        CompatibilityReport {
            os_ok: false,
            arch_ok: true,
            com_ok: false,
            details: vec![
                "Not running on Windows: OS-version and COM checks are unavailable on this \
                 platform."
                    .to_string(),
            ],
        }
    }
}

/// The directory BVC assets (`weya_nc.dll` and the ONNX model bundle) are
/// extracted into on every launch: `%LOCALAPPDATA%\ClearNAI` on Windows -
/// a per-user, writable-without-admin-rights location that is the
/// conventional home for this kind of app-managed cache/support data (as
/// opposed to `%PROGRAMFILES%`, which a non-admin install typically can't
/// write to, or right next to the exe, which is what this whole change
/// moves away from).
///
/// If `LOCALAPPDATA` isn't set (unusual, but not impossible - e.g. a
/// stripped-down service account), falls back to `%TEMP%\ClearNAI` so the
/// app still has *somewhere* writable rather than failing outright; if
/// even that can't be resolved, falls back to `.\ClearNAI` (relative to
/// the current directory).
///
/// On non-Windows platforms (only ever exercised by `cargo check`/
/// `cargo test` from this Linux dev machine, never for real) this resolves
/// to a temp directory instead, purely so the portable parts of this
/// module have something sane to test against.
///
/// Creates the directory (`create_dir_all`) if it doesn't already exist;
/// callers don't need to do this themselves.
pub fn app_data_dir() -> PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("TEMP"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    #[cfg(not(windows))]
    let base = std::env::temp_dir();

    let dir = base.join("ClearNAI");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        crate::log_error!("[clearnairt] WARNING: failed to create app data dir {}: {e:#}", dir.display());
    }
    dir
}

/// Where the BVC model bundle lives once extracted: under
/// `app_data_dir()`, using the real filename `bvc-hush` already exports
/// (never re-hardcoded here, so the two crates can't silently drift apart).
/// Written as the raw, still-compressed `.tar.gz` - confirmed against the
/// real upstream deployment README that `weya_nc_model_load_from_path`
/// wants the archive itself, not an unpacked directory (see
/// `bvc_hush::MODEL_BUNDLE_FILE_NAME`'s docs for the full story, including
/// a briefly-reverted attempt to unpack it that turned out to be wrong).
pub fn model_bundle_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(bvc_hush::MODEL_BUNDLE_FILE_NAME)
}

/// Where the `bpo_v1` fine-tune's model bundle lives once extracted - see
/// `EMBEDDED_BPO_V1_MODEL_BYTES`'s docs. A second, independent file next to
/// `model_bundle_path`'s, never overwriting it.
pub fn bpo_v1_model_bundle_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(bvc_hush::BPO_V1_MODEL_BUNDLE_FILE_NAME)
}

/// Where the DLL lives once extracted: under `app_data_dir()`.
pub fn dll_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(bvc_hush::DLL_FILE_NAME)
}

/// Where the speaker path's own separate copy of the same DLL bytes lives
/// once extracted. See `bvc_hush::HushBvcStage::try_load_named`'s docs for
/// why this needs to be a distinct on-disk file rather than reusing
/// `dll_path`.
pub fn speaker_dll_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(bvc_hush::SPEAKER_DLL_FILE_NAME)
}

/// Where `onnxruntime.dll` lives once extracted: under `app_data_dir()` -
/// the path the accent and transcript engines load it from.
pub fn onnxruntime_dll_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(ONNXRUNTIME_DLL_FILE_NAME)
}

/// True only if the real bundle file exists and its content exactly
/// matches the embedded bytes - i.e. `ensure_bvc_assets_extracted` has
/// already run successfully against this directory. A zero-byte, partial,
/// or stale/mismatched file counts as NOT present so extraction is
/// retried. Compares full content, not just size - see
/// `extract_one_if_needed`'s docs for why a size-only check isn't safe
/// here.
///
/// Not called from `main.rs` any more (extraction now runs unconditionally
/// and reports its own `Result` - see `ensure_bvc_assets_extracted`), but
/// kept as a small, independently useful/testable query, matching this
/// module's existing habit of keeping honest presence-checks around.
#[allow(dead_code)]
pub fn model_bundle_present(app_data_dir: &Path) -> bool {
    std::fs::read(model_bundle_path(app_data_dir))
        .map(|existing| existing == EMBEDDED_MODEL_BYTES)
        .unwrap_or(false)
}

/// Whether a single embedded asset needed to be (re)written, returned per
/// file by `ensure_bvc_assets_extracted` so callers/tests can distinguish
/// "already there, nothing to do" from "just wrote it".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetState {
    /// The file already existed at the target path with a size matching
    /// the embedded bytes - left untouched.
    AlreadyPresent,
    /// The file was missing, or present with a mismatched size (e.g. a
    /// partial/corrupt leftover, or an older/newer embedded version), so it
    /// was (re)written from the embedded bytes.
    Extracted,
}

/// Result of `ensure_bvc_assets_extracted`: what happened to each of the
/// embedded files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractionOutcome {
    pub dll: AssetState,
    /// The speaker path's separate copy of the same DLL bytes - see
    /// `speaker_dll_path`.
    pub speaker_dll: AssetState,
    pub model: AssetState,
    /// The `bpo_v1` fine-tune's model bundle - see
    /// `bpo_v1_model_bundle_path`. Independent of `model` above; never
    /// overwrites or is overwritten by it.
    pub bpo_v1_model: AssetState,
    /// `onnxruntime.dll` - see `onnxruntime_dll_path`.
    pub onnxruntime_dll: AssetState,
}

impl ExtractionOutcome {
    /// True if at least one file was freshly written this run - useful for
    /// deciding whether to mention anything to the user at all versus
    /// silently proceeding straight past the setup screen.
    #[allow(dead_code)]
    pub fn wrote_anything(&self) -> bool {
        self.dll == AssetState::Extracted
            || self.speaker_dll == AssetState::Extracted
            || self.model == AssetState::Extracted
            || self.bpo_v1_model == AssetState::Extracted
            || self.onnxruntime_dll == AssetState::Extracted
    }
}

/// Writes `bytes` to `final_path` atomically: first to a `.part` sibling
/// file, then renamed into place only on complete success - the same
/// crash-safety property the old network download used, so a process
/// killed mid-write never leaves a corrupt file at `final_path` that a
/// later run would mistake for good (the size check in
/// `extract_one_if_needed` would in fact still catch a truncated `.part`
/// left over from an old, pre-embedding version of this app, since it only
/// ever inspects `final_path`, never `.part`).
fn write_atomically(final_path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let part_path = final_path.with_extension(match final_path.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });
    let result = (|| -> anyhow::Result<()> {
        if let Some(parent) = part_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("failed to create directory {}: {e}", parent.display()))?;
        }
        let mut file = std::fs::File::create(&part_path)
            .map_err(|e| anyhow::anyhow!("failed to create temp file {}: {e}", part_path.display()))?;
        file.write_all(bytes)
            .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", part_path.display()))?;
        file.sync_all().ok();
        drop(file);
        std::fs::rename(&part_path, final_path)
            .map_err(|e| anyhow::anyhow!("failed to move {} into place: {e}", final_path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&part_path);
    }
    result
}

/// Writes `bytes` to `path` only if it doesn't already exist there with
/// byte-identical content - avoids rewriting several MB of already-correct
/// data on every single launch.
///
/// Compares full contents, not just file size. An earlier revision of this
/// function compared size only, on the reasoning that the embedded bytes
/// never change between launches of the same build - true, but it silently
/// breaks across *different* builds whose embedded asset happens to be the
/// exact same byte length (not a hypothetical: the real
/// `advanced_dfnet16k_model_best_onnx.tar.gz` and `bpo_v1_onnx.tar.gz`
/// bundles this app ships differ by only a few hundred bytes out of ~8.5MB,
/// a near-miss that a less similar pair of models could easily land on
/// exactly). A size-only check in that case would see "right length,
/// extracted already" on an update that actually changed the model's
/// content and leave the stale, wrong file in place indefinitely, with no
/// error or warning. Reading the existing file back costs a few
/// milliseconds for assets this size; paid once per launch, that's a
/// correct trade against "silently serves a stale model forever."
fn extract_one_if_needed(path: &Path, bytes: &[u8]) -> anyhow::Result<AssetState> {
    let already_correct = std::fs::read(path).map(|existing| existing == bytes).unwrap_or(false);
    if already_correct {
        return Ok(AssetState::AlreadyPresent);
    }
    write_atomically(path, bytes)?;
    Ok(AssetState::Extracted)
}

/// Marker file written into `model_dir` after a successful unpack, holding
/// the byte length of the embedded (still-compressed) archive it was
/// unpacked from - a cheap way to detect "already unpacked from this exact
/// embedded build" on every subsequent launch without re-hashing or
/// re-walking every file the model bundle contains.
/// Self-extracts the embedded `weya_nc.dll` and ONNX model bundle into
/// `target_dir` (in practice, always `app_data_dir()` - factored out as a
/// parameter here purely so this logic is unit-testable against a plain
/// temp directory without touching `%LOCALAPPDATA%`).
///
/// Fast enough (bytes are already resident in the binary's own image; this
/// is at most two size-checks plus two small file writes) to run
/// synchronously on the GUI thread at every launch, unlike the old
/// network-download path this replaces - no background thread or progress
/// bar is needed.
///
/// Returns a real `Result`: disk-full, permission-denied (e.g. an
/// unwritable `%LOCALAPPDATA%`), or any other I/O failure is surfaced as an
/// `Err` rather than panicking, so the caller can show it on the setup
/// screen and offer "Continue anyway" - BVC simply stays unavailable in
/// that case, exactly like the DLL-not-found case already handled by
/// `bvc_hush::HushBvcStage::try_load`.
pub fn ensure_bvc_assets_extracted(target_dir: &Path) -> anyhow::Result<ExtractionOutcome> {
    let dll = extract_one_if_needed(&dll_path(target_dir), EMBEDDED_DLL_BYTES)
        .map_err(|e| anyhow::anyhow!("failed to extract {}: {e:#}", bvc_hush::DLL_FILE_NAME))?;
    // Same bytes, a second on-disk copy under a different filename - see
    // `speaker_dll_path`'s docs for why the speaker path loads its own
    // separate file instead of the mic path's `weya_nc.dll`.
    let speaker_dll = extract_one_if_needed(&speaker_dll_path(target_dir), EMBEDDED_DLL_BYTES)
        .map_err(|e| anyhow::anyhow!("failed to extract {}: {e:#}", bvc_hush::SPEAKER_DLL_FILE_NAME))?;
    let model = extract_one_if_needed(&model_bundle_path(target_dir), EMBEDDED_MODEL_BYTES)
        .map_err(|e| anyhow::anyhow!("failed to extract {}: {e:#}", bvc_hush::MODEL_BUNDLE_FILE_NAME))?;
    // The user-supplied bpo_v1 fine-tune - a separate file, never
    // overwriting `model` above.
    let bpo_v1_model = extract_one_if_needed(&bpo_v1_model_bundle_path(target_dir), EMBEDDED_BPO_V1_MODEL_BYTES)
        .map_err(|e| anyhow::anyhow!("failed to extract {}: {e:#}", bvc_hush::BPO_V1_MODEL_BUNDLE_FILE_NAME))?;
    let onnxruntime_dll = extract_one_if_needed(&onnxruntime_dll_path(target_dir), EMBEDDED_ONNXRUNTIME_DLL_BYTES)
        .map_err(|e| anyhow::anyhow!("failed to extract {}: {e:#}", ONNXRUNTIME_DLL_FILE_NAME))?;
    Ok(ExtractionOutcome { dll, speaker_dll, model, bpo_v1_model, onnxruntime_dll })
}

// ---------------------------------------------------------------------------------------------------------------
// Accent model bundle (`app_models`)
//
// Unlike the Hush bundle (8 MB, `include_bytes!`-embedded), the accent bundle is ~270 MB (a 194 MB encoder, a 27 MB
// decoder and a 52 MB vocoder, all fp32), far past what can be committed to git or reasonably embedded in the exe.
// It is shipped as a folder next to `clearnairt.exe` (`app_models/`, see `installer.nsi`) and installed into
// `%LOCALAPPDATA%\ClearNAI\app_models` the same way the Hush bundle is: a file is only (re)written when its FULL
// content differs from the shipped one (never a size-only check; see `extract_one_if_needed`'s docs for why).
// ---------------------------------------------------------------------------------------------------------------

/// Where the accent bundle lives once installed.
pub fn accent_models_dir(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("app_models")
}

/// Where the shipped accent bundle is looked up: `ACCENT_MODELS_SRC`, else `app_models` next to the exe.
pub fn accent_models_source_dir() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("ACCENT_MODELS_SRC") {
        return Some(PathBuf::from(p));
    }
    std::env::current_exe().ok()?.parent().map(|d| d.join("app_models"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccentBundleState {
    /// Installed target already matched the shipped files byte for byte (or no shipped copy exists to compare
    /// against and the installed one is complete).
    AlreadyPresent,
    /// At least one file was missing or differed and was (re)copied.
    Installed,
}

/// Streaming full-content comparison (1 MiB chunks), so a 194 MB file is never held in memory twice.
fn files_identical(a: &Path, b: &Path) -> std::io::Result<bool> {
    use std::io::Read;
    let (ma, mb) = (std::fs::metadata(a)?, std::fs::metadata(b)?);
    if ma.len() != mb.len() {
        return Ok(false);
    }
    let (mut fa, mut fb) = (std::fs::File::open(a)?, std::fs::File::open(b)?);
    let (mut ba, mut bb) = (vec![0u8; 1 << 20], vec![0u8; 1 << 20]);
    loop {
        let n = fa.read(&mut ba)?;
        if n == 0 {
            return Ok(true);
        }
        fb.read_exact(&mut bb[..n])?;
        if ba[..n] != bb[..n] {
            return Ok(false);
        }
    }
}

fn copy_atomically(src: &Path, dst: &Path) -> anyhow::Result<()> {
    let part = dst.with_extension(match dst.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent).map_err(|e| anyhow::anyhow!("failed to create {}: {e}", parent.display()))?;
    }
    let result = std::fs::copy(src, &part)
        .map_err(|e| anyhow::anyhow!("failed to copy {} to {}: {e}", src.display(), part.display()))
        .and_then(|_| {
            std::fs::rename(&part, dst).map_err(|e| anyhow::anyhow!("failed to move {} into place: {e}", dst.display()))
        });
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

/// Makes `target` hold exactly the files of the shipped bundle in `source`. Errors if neither a shipped copy nor a
/// complete installed copy exists. Called lazily, from the accent worker thread, the first time Accent is switched
/// on - a user who never enables it pays nothing at launch.
pub fn ensure_accent_models(source: Option<&Path>, target: &Path) -> anyhow::Result<AccentBundleState> {
    let files = accent_onnx::BUNDLE_FILES;
    let source = source.filter(|s| s.is_dir() && *s != target);
    let Some(source) = source else {
        let missing: Vec<&str> = files.iter().copied().filter(|f| !target.join(f).is_file()).collect();
        anyhow::ensure!(
            missing.is_empty(),
            "accent model bundle not found: no `app_models` folder next to the exe and {} is missing {}",
            target.display(),
            missing.join(", ")
        );
        return Ok(AccentBundleState::AlreadyPresent);
    };
    let missing: Vec<&str> = files.iter().copied().filter(|f| !source.join(f).is_file()).collect();
    anyhow::ensure!(missing.is_empty(), "shipped accent bundle {} is missing {}", source.display(), missing.join(", "));
    let mut changed = false;
    for f in files {
        let (s, t) = (source.join(f), target.join(f));
        let same = t.is_file() && files_identical(&s, &t).unwrap_or(false);
        if !same {
            copy_atomically(&s, &t)?;
            changed = true;
        }
    }
    Ok(if changed { AccentBundleState::Installed } else { AccentBundleState::AlreadyPresent })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_bundle(dir: &Path, fill: u8) {
        std::fs::create_dir_all(dir).unwrap();
        for (i, f) in accent_onnx::BUNDLE_FILES.iter().enumerate() {
            std::fs::write(dir.join(f), vec![fill.wrapping_add(i as u8); 3_000_000 + i]).unwrap();
        }
    }

    #[test]
    fn accent_bundle_installs_then_is_idempotent_and_compares_full_content() {
        let root = std::env::temp_dir().join(format!("clearnai_accent_bundle_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (src, dst) = (root.join("src"), root.join("dst"));
        fake_bundle(&src, 10);

        assert_eq!(ensure_accent_models(Some(&src), &dst).unwrap(), AccentBundleState::Installed);
        assert_eq!(ensure_accent_models(Some(&src), &dst).unwrap(), AccentBundleState::AlreadyPresent);

        // Same length, one byte different deep inside the file: a size check would miss it, a full compare must not.
        let victim = dst.join(accent_onnx::BUNDLE_FILES[4]);
        let mut bytes = std::fs::read(&victim).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&victim, &bytes).unwrap();
        assert_eq!(ensure_accent_models(Some(&src), &dst).unwrap(), AccentBundleState::Installed);
        assert!(files_identical(&src.join(accent_onnx::BUNDLE_FILES[4]), &victim).unwrap());

        // A newer shipped bundle replaces the installed one.
        fake_bundle(&src, 99);
        assert_eq!(ensure_accent_models(Some(&src), &dst).unwrap(), AccentBundleState::Installed);

        // No shipped copy, but a complete installed one: usable as is.
        assert_eq!(ensure_accent_models(None, &dst).unwrap(), AccentBundleState::AlreadyPresent);
        std::fs::remove_file(dst.join("catalog.json")).unwrap();
        assert!(ensure_accent_models(None, &dst).unwrap_err().to_string().contains("catalog.json"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn windows_10_and_11_major_version_is_supported() {
        assert!(windows_major_version_is_supported(10));
        assert!(windows_major_version_is_supported(11));
        assert!(windows_major_version_is_supported(12), "future major versions >= 10 are not rejected");
    }

    #[test]
    fn pre_windows_10_major_version_is_unsupported() {
        assert!(!windows_major_version_is_supported(6)); // Vista/7/8/8.1 era
        assert!(!windows_major_version_is_supported(0));
    }

    #[test]
    fn model_bundle_path_uses_the_real_shared_constant() {
        let dir = PathBuf::from("/some/exe/dir");
        let path = model_bundle_path(&dir);
        assert_eq!(path, dir.join(bvc_hush::MODEL_BUNDLE_FILE_NAME));
    }

    #[test]
    fn model_bundle_absent_when_no_file_exists() {
        let dir = std::env::temp_dir().join(format!("clearnai_setup_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!model_bundle_present(&dir));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_bundle_absent_when_file_is_zero_bytes() {
        let dir = std::env::temp_dir().join(format!("clearnai_setup_test_zero_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(model_bundle_path(&dir), []).unwrap();
        assert!(!model_bundle_present(&dir), "a zero-byte file must not count as present");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_bundle_absent_when_size_does_not_match_embedded_bytes() {
        let dir = std::env::temp_dir().join(format!("clearnai_setup_test_wrongsize_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(model_bundle_path(&dir), b"not a real tarball, and the wrong size too").unwrap();
        assert!(!model_bundle_present(&dir), "a size mismatch vs the embedded bytes must not count as present");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test for the exact bug `extract_one_if_needed`'s docs
    /// describe: a stale on-disk file that happens to match the embedded
    /// asset's SIZE but not its CONTENT must still be detected as stale and
    /// rewritten - a size-only check would wrongly call this
    /// `AlreadyPresent` and leave the wrong bytes in place forever.
    #[test]
    fn extract_one_if_needed_rewrites_a_same_size_but_wrong_content_file() {
        let dir = std::env::temp_dir().join(format!("clearnai_setup_test_same_size_wrong_content_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("asset.bin");

        let real_bytes = b"the real, correct embedded content";
        // Same length as `real_bytes`, every byte different.
        let stale_bytes: Vec<u8> = real_bytes.iter().map(|b| b.wrapping_add(1)).collect();
        assert_eq!(stale_bytes.len(), real_bytes.len(), "test setup: must be a genuine same-size mismatch");
        std::fs::write(&path, &stale_bytes).unwrap();

        let state = extract_one_if_needed(&path, real_bytes).expect("extraction must succeed");
        assert_eq!(
            state,
            AssetState::Extracted,
            "a same-size-but-different-content file must be treated as stale and rewritten, not AlreadyPresent"
        );
        assert_eq!(std::fs::read(&path).unwrap(), real_bytes, "the stale content must actually have been replaced");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_bundle_present_after_real_extraction() {
        let dir = std::env::temp_dir().join(format!("clearnai_setup_test_present_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        ensure_bvc_assets_extracted(&dir).unwrap();
        assert!(model_bundle_present(&dir));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_bvc_assets_extracted_writes_files_matching_embedded_bytes() {
        let dir = std::env::temp_dir().join(format!("clearnai_setup_test_extract_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let outcome = ensure_bvc_assets_extracted(&dir).expect("extraction into a fresh temp dir must succeed");
        assert_eq!(outcome.dll, AssetState::Extracted);
        assert_eq!(outcome.speaker_dll, AssetState::Extracted);
        assert_eq!(outcome.model, AssetState::Extracted);
        assert_eq!(outcome.bpo_v1_model, AssetState::Extracted);
        assert_eq!(outcome.onnxruntime_dll, AssetState::Extracted);
        assert!(outcome.wrote_anything());

        // Full content match, not just size - proves `extract_one_if_needed`
        // actually wrote the real embedded bytes, not merely a
        // correctly-sized file.
        assert_eq!(std::fs::read(dll_path(&dir)).unwrap(), EMBEDDED_DLL_BYTES);
        assert_eq!(std::fs::read(speaker_dll_path(&dir)).unwrap(), EMBEDDED_DLL_BYTES);
        assert_eq!(std::fs::read(model_bundle_path(&dir)).unwrap(), EMBEDDED_MODEL_BYTES);
        assert_eq!(std::fs::read(bpo_v1_model_bundle_path(&dir)).unwrap(), EMBEDDED_BPO_V1_MODEL_BYTES);
        assert_eq!(std::fs::read(onnxruntime_dll_path(&dir)).unwrap(), EMBEDDED_ONNXRUNTIME_DLL_BYTES);

        // No leftover `.part` files after a successful extraction.
        assert!(!dll_path(&dir).with_extension("dll.part").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_bvc_assets_extracted_is_idempotent_and_does_not_rewrite_correct_files() {
        let dir = std::env::temp_dir().join(format!("clearnai_setup_test_idempotent_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let first = ensure_bvc_assets_extracted(&dir).expect("first extraction must succeed");
        assert_eq!(first.dll, AssetState::Extracted);
        assert_eq!(first.speaker_dll, AssetState::Extracted);
        assert_eq!(first.model, AssetState::Extracted);
        assert_eq!(first.bpo_v1_model, AssetState::Extracted);
        assert_eq!(first.onnxruntime_dll, AssetState::Extracted);

        let second = ensure_bvc_assets_extracted(&dir).expect("second extraction must succeed");
        assert_eq!(second.dll, AssetState::AlreadyPresent, "a byte-identical file must not be rewritten");
        assert_eq!(second.speaker_dll, AssetState::AlreadyPresent, "a byte-identical file must not be rewritten");
        assert_eq!(second.model, AssetState::AlreadyPresent, "a byte-identical file must not be rewritten");
        assert_eq!(second.bpo_v1_model, AssetState::AlreadyPresent, "a byte-identical file must not be rewritten");
        assert_eq!(second.onnxruntime_dll, AssetState::AlreadyPresent, "a byte-identical file must not be rewritten");
        assert!(!second.wrote_anything());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compatibility_report_all_ok_requires_every_field() {
        let mut r = CompatibilityReport { os_ok: true, arch_ok: true, com_ok: true, details: Vec::new() };
        assert!(r.all_ok());
        r.com_ok = false;
        assert!(!r.all_ok());
    }

    #[cfg(not(windows))]
    #[test]
    fn non_windows_report_is_honestly_not_ok() {
        let report = check_system_compatibility();
        assert!(!report.os_ok);
        assert!(!report.com_ok);
        assert!(report.arch_ok);
        assert!(!report.details.is_empty());
    }

    #[test]
    fn app_data_dir_is_creatable_and_a_directory() {
        let dir = app_data_dir();
        assert!(dir.is_dir(), "app_data_dir() must create the directory it returns: {}", dir.display());
    }
}
