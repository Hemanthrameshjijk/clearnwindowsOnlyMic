//! User-mode side of the ClearNAI engine <-> virtual-audio-driver transport.
//!
//! # What this is
//!
//! A bounded, lock-free, single-writer/single-reader ring buffer of fixed-
//! size audio frames, backed by a named Windows shared-memory section
//! (`CreateFileMappingW`), so a *separate process* - in particular, a
//! future kernel-mode virtual audio driver reading via `ZwOpenSection` -
//! can consume what this process writes, without either side needing to
//! know about the other's internals beyond this file's documented layout.
//!
//! This is deliberately **only** the transport. It contains no audio
//! processing, no model inference, no knowledge of BVC/RNNoise/Studio -
//! see `docs/DRIVER_PLAN.md`'s architecture rule ("do not put AI inference
//! ... inside the Windows kernel driver"). The engine (`app/src/engine.rs`)
//! is responsible for calling `SharedRingWriter::push_frame` with already-
//! fully-processed 480-sample frames; this crate only gets them from this
//! process's address space into a place a driver could read them from.
//!
//! # Status
//!
//! `SOURCE-VERIFIED` / `UNIT-TESTED` on this (Windows) dev machine - real
//! `CreateFileMappingW`/`MapViewOfFile` calls, a real round-trip test
//! (writer and reader opening the same named section from within one
//! process, since that's what's exercisable here). **`RUNTIME_VERIFIED`
//! against an actual kernel-mode reader is NOT done** - no driver exists
//! yet to read this (see `docs/DRIVER_PLAN.md`). Do not claim this is
//! "the driver is now working" - it is the half of the transport that
//! doesn't require kernel code, built and tested on its own.
//!
//! # Wire format (this is the real contract a future driver must match)
//!
//! A named shared-memory section of exactly
//! `size_of::<RingHeader>() + capacity_frames * frame_samples * size_of::<f32>()`
//! bytes, laid out as:
//!
//! ```text
//! offset 0                              : RingHeader (see below), #[repr(C)]
//! offset size_of::<RingHeader>()        : frame slot 0 (frame_samples x f32, little-endian)
//! offset ... + 1 * frame_bytes          : frame slot 1
//! offset ... + (capacity_frames-1) * .. : frame slot (capacity_frames - 1)
//! ```
//!
//! `RingHeader` is plain-old-data (`#[repr(C)]`, only fixed-width integers
//! and `AtomicU64`, which have the same bit representation as a plain
//! `u64` - safe for a reader on the other side of a shared mapping to read
//! with ordinary volatile/atomic loads, in C or Rust). Field meanings:
//!
//! - `magic`: always `PROTOCOL_MAGIC` - a reader must check this before
//!   trusting anything else in the mapping (protects against opening some
//!   unrelated section that happens to share a name).
//! - `version`: `PROTOCOL_VERSION` - bump this, not the layout in place,
//!   if the wire format ever changes incompatibly.
//! - `sample_rate_hz`, `frame_samples`: must match `dsp_core::SAMPLE_RATE_HZ`/
//!   `dsp_core::FRAME_SAMPLES` for this writer; a reader should check these
//!   rather than assume 48000/480 forever.
//! - `capacity_frames`: number of frame slots physically present after the
//!   header - a reader needs this to know the mapping's total size before
//!   it can even map the rest.
//! - `write_index`/`read_index`: monotonically increasing frame counters
//!   (never wrapped/masked themselves - only `index % capacity_frames`
//!   is used to locate a slot). The gap `write_index - read_index` is the
//!   ring's current occupancy.
//! - `dropped_frames`: incremented by the writer every time it declines to
//!   overwrite an unread slot (see `push_frame`'s docs) - real, honest
//!   telemetry, not silently discarded data.
//!
//! # Concurrency model
//!
//! Single writer, single reader (SPSC), exactly like `dsp_core`'s in-
//! process ring buffer - just across a process boundary instead of across
//! threads. `write_index` is bumped with `Release` ordering *after* the
//! frame's samples are copied into its slot; a reader loads `write_index`
//! with `Acquire` before reading that slot's samples, which is what makes
//! "the data is visible once the index says it's there" actually true
//! under the C11/Rust memory model, not just true "in practice" on x86.
//!
//! Known, honestly-stated limitation: like any fixed-capacity ring with a
//! free-running writer, a reader that falls more than `capacity_frames`
//! frames behind can have its next read overwritten out from under it by
//! the time it gets there (the writer has no way to block - see
//! `push_frame`'s docs on why blocking is never acceptable here). This
//! crate does not detect that specific case (only "buffer full" is
//! detected, via `dropped_frames`); a large enough `capacity_frames` for
//! the reader's expected worst-case latency is how a real deployment
//! would need to size this, exactly as `dsp_core`'s in-process ring
//! buffers already are today (2 seconds of headroom).

use std::sync::atomic::{AtomicU64, Ordering};

/// Named section this project's engine writes its final, fully-processed
/// mic-path frames into. `Local\` (not `Global\`) scopes it to the current
/// user session - correct for a per-user desktop app; a kernel-mode reader
/// (which is session-0/system-wide) would need `Global\` instead, so this
/// exact name is a placeholder for the user-mode round-trip this crate
/// currently proves, not yet the real driver-facing name (see
/// `docs/DRIVER_PLAN.md` - that will need `Global\` and a security
/// descriptor granting the driver's access, neither of which exists yet).
pub const MIC_OUTPUT_SECTION_NAME: &str = "Local\\ClearNAI_MicOutput_v1";

/// 200 frames = 2 seconds of headroom at the standard 10ms frame period -
/// matches every in-process ring buffer already used elsewhere in this
/// workspace (`app::engine::make_ring_buffer(2.0)`), for the same reason:
/// generous slack against transient stalls without hiding a sustained
/// backlog (a consistently-full ring here still means `dropped_frames`
/// climbs, same honesty as everywhere else in this project).
pub const DEFAULT_CAPACITY_FRAMES: u32 = 200;

/// Identifies a valid ClearNAI shared-memory transport section, distinct
/// from any other named section that might coincidentally share a name.
/// ASCII "CNAG" (ClearNAI Audio) read as a little-endian u32.
pub const PROTOCOL_MAGIC: u32 = 0x47414E43;

/// Bump this - not the byte layout of `RingHeader` in place - if the wire
/// format ever needs to change incompatibly. A reader must refuse to
/// proceed if this doesn't match what it was built against.
pub const PROTOCOL_VERSION: u32 = 1;

/// Plain-old-data header at the start of the shared section. See this
/// module's doc comment for the full field-by-field contract - this is
/// the actual wire format, not just an implementation detail.
#[repr(C)]
pub struct RingHeader {
    pub magic: u32,
    pub version: u32,
    pub sample_rate_hz: u32,
    pub frame_samples: u32,
    pub capacity_frames: u32,
    _reserved: u32,
    pub write_index: AtomicU64,
    pub read_index: AtomicU64,
    pub dropped_frames: AtomicU64,
}

impl RingHeader {
    fn frame_bytes(frame_samples: u32) -> usize {
        frame_samples as usize * std::mem::size_of::<f32>()
    }

    /// Total byte size of the shared section for `capacity_frames` slots of
    /// `frame_samples` each - header plus every frame slot's raw bytes.
    pub fn total_size(capacity_frames: u32, frame_samples: u32) -> usize {
        std::mem::size_of::<RingHeader>() + capacity_frames as usize * Self::frame_bytes(frame_samples)
    }
}

#[cfg(test)]
mod portable_tests {
    use super::*;

    #[test]
    fn total_size_accounts_for_header_and_every_frame_slot() {
        let capacity = 10u32;
        let frame_samples = 480u32;
        let expected =
            std::mem::size_of::<RingHeader>() + 10 * 480 * std::mem::size_of::<f32>();
        assert_eq!(RingHeader::total_size(capacity, frame_samples), expected);
    }

    #[test]
    fn magic_and_version_are_stable_constants() {
        // Pinned values, not just "whatever the code currently computes" -
        // a future protocol version bump must be a deliberate edit here,
        // never an accidental drift.
        assert_eq!(PROTOCOL_MAGIC, 0x47414E43);
        assert_eq!(PROTOCOL_VERSION, 1);
    }
}

#[cfg(windows)]
mod windows_impl {
    use super::*;
    use anyhow::{bail, Context, Result};
    use std::ptr::NonNull;
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows::Win32::System::Memory::{
        CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS,
        MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
    };

    fn wide_name(name: &str) -> Vec<u16> {
        name.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// The writer side: owns the section, initializes the header, and
    /// appends frames. Constructed exactly once per process (the engine's
    /// mic pipeline thread, in practice) - this is the SPSC "S" (single
    /// writer).
    pub struct SharedRingWriter {
        _handle: HANDLE,
        view: MEMORY_MAPPED_VIEW_ADDRESS,
        header: NonNull<RingHeader>,
        data: NonNull<f32>,
        capacity_frames: u32,
        frame_samples: u32,
    }

    // SAFETY: the underlying shared memory is only ever mutated through
    // this type's own methods, which enforce the single-writer contract at
    // the type level (only one `SharedRingWriter` is ever constructed for
    // a given section name by this process); the OS guarantees the mapped
    // memory itself is valid for the handle's lifetime.
    unsafe impl Send for SharedRingWriter {}

    impl SharedRingWriter {
        /// Creates a brand-new named shared section (fails if one with this
        /// exact name already exists and is still open elsewhere - a stale
        /// leftover mapping must not be silently reused, since its
        /// `capacity_frames`/`frame_samples` might not match what this
        /// call is about to write into the header).
        pub fn create(name: &str, capacity_frames: u32, frame_samples: u32) -> Result<Self> {
            let total_size = RingHeader::total_size(capacity_frames, frame_samples);
            let wide = wide_name(name);

            // SAFETY: `wide` is a valid null-terminated UTF-16 buffer kept
            // alive for the duration of this call; `total_size` fits in
            // the (size_high, size_low) pair for any realistic frame
            // count/capacity this project would ever configure.
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    None,
                    PAGE_READWRITE,
                    (total_size >> 32) as u32,
                    (total_size & 0xFFFF_FFFF) as u32,
                    PCWSTR(wide.as_ptr()),
                )
            }
            .context("CreateFileMappingW failed")?;

            // SAFETY: `handle` was just created successfully above.
            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total_size) };
            if view.Value.is_null() {
                unsafe { CloseHandle(handle).ok() };
                bail!("MapViewOfFile returned a null view for a freshly created section");
            }

            let header_ptr = view.Value.cast::<RingHeader>();
            // SAFETY: `view` is a valid mapping of at least `total_size`
            // bytes, which is >= size_of::<RingHeader>() by construction;
            // `header_ptr` is non-null (checked above) and correctly
            // aligned (the OS maps at page granularity, far more aligned
            // than `RingHeader` requires).
            unsafe {
                header_ptr.write(RingHeader {
                    magic: PROTOCOL_MAGIC,
                    version: PROTOCOL_VERSION,
                    sample_rate_hz: dsp_core::SAMPLE_RATE_HZ,
                    frame_samples,
                    capacity_frames,
                    _reserved: 0,
                    write_index: AtomicU64::new(0),
                    read_index: AtomicU64::new(0),
                    dropped_frames: AtomicU64::new(0),
                });
            }

            let data_ptr = unsafe { view.Value.cast::<u8>().add(std::mem::size_of::<RingHeader>()) }.cast::<f32>();

            Ok(Self {
                _handle: handle,
                view,
                header: NonNull::new(header_ptr).expect("checked non-null above"),
                data: NonNull::new(data_ptr).expect("derived from a non-null, in-bounds view"),
                capacity_frames,
                frame_samples,
            })
        }

        /// Appends one full frame (`frame.len()` must equal the
        /// `frame_samples` this writer was created with). Never blocks: if
        /// the ring is full (the reader hasn't kept up), the frame is
        /// dropped and `dropped_frames` is incremented - the realtime
        /// audio thread calling this must never be made to wait on a
        /// reader process that might be slow, stalled, or gone. Returns
        /// `true` if the frame was actually written.
        pub fn push_frame(&self, frame: &[f32]) -> bool {
            debug_assert_eq!(frame.len(), self.frame_samples as usize);
            // SAFETY: `self.header` points at a valid, live `RingHeader`
            // for this mapping's entire lifetime (owned by `self`).
            let header = unsafe { self.header.as_ref() };

            let write_idx = header.write_index.load(Ordering::Relaxed);
            let read_idx = header.read_index.load(Ordering::Acquire);
            if write_idx.wrapping_sub(read_idx) >= self.capacity_frames as u64 {
                header.dropped_frames.fetch_add(1, Ordering::Relaxed);
                return false;
            }

            let slot = (write_idx % self.capacity_frames as u64) as usize;
            // SAFETY: `slot < capacity_frames`, and `self.data` points at
            // `capacity_frames * frame_samples` valid, writable `f32`s for
            // this mapping's lifetime; `frame.len() == frame_samples`
            // (debug-checked above, and the sole real caller in this
            // workspace - `app/src/engine.rs` - always passes exactly
            // `dsp_core::FRAME_SAMPLES` samples).
            unsafe {
                let dst = self.data.as_ptr().add(slot * self.frame_samples as usize);
                std::ptr::copy_nonoverlapping(frame.as_ptr(), dst, self.frame_samples as usize);
            }

            // Release: everything above (the sample copy) must be visible
            // to any reader that observes this incremented index.
            header.write_index.store(write_idx + 1, Ordering::Release);
            true
        }

        pub fn capacity_frames(&self) -> u32 {
            self.capacity_frames
        }
    }

    impl Drop for SharedRingWriter {
        fn drop(&mut self) {
            // SAFETY: `self.view`/`self._handle` were obtained from this
            // struct's own successful `create` call and have not been
            // freed elsewhere; this runs at most once (owned `Drop`).
            unsafe {
                let _ = UnmapViewOfFile(self.view);
                let _ = CloseHandle(self._handle);
            }
        }
    }

    /// The reader side: opens an *existing* section by name (created by a
    /// `SharedRingWriter` elsewhere - in this process for the round-trip
    /// test below, or in a real future driver's kernel-mode equivalent of
    /// this open+read logic) and consumes frames from it.
    pub struct SharedRingReader {
        _handle: HANDLE,
        view: MEMORY_MAPPED_VIEW_ADDRESS,
        header: NonNull<RingHeader>,
        data: NonNull<f32>,
        capacity_frames: u32,
        frame_samples: u32,
    }

    unsafe impl Send for SharedRingReader {}

    impl SharedRingReader {
        /// Opens the named section and validates `magic`/`version` before
        /// trusting anything else in it - see this module's doc comment.
        pub fn open(name: &str) -> Result<Self> {
            let wide = wide_name(name);
            // SAFETY: `wide` is a valid null-terminated UTF-16 buffer kept
            // alive for the duration of this call.
            let handle = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS.0, false, PCWSTR(wide.as_ptr())) }
                .context("OpenFileMappingW failed - has the writer created this section yet?")?;

            // Map just the header first to learn the real total size
            // before mapping the rest - we don't know `capacity_frames`/
            // `frame_samples` (and therefore the true total size) until
            // we've read the header.
            let header_probe_size = std::mem::size_of::<RingHeader>();
            // SAFETY: `handle` was just opened successfully; a partial
            // view of at least the header's own size is always a valid
            // sub-range of whatever the writer actually created (the
            // header is always the first thing in the section).
            let probe_view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, header_probe_size) };
            if probe_view.Value.is_null() {
                unsafe { CloseHandle(handle).ok() };
                bail!("MapViewOfFile (header probe) returned a null view");
            }
            let probe_header = probe_view.Value.cast::<RingHeader>();
            // SAFETY: mapped at least `header_probe_size` bytes above.
            let (magic, version, sample_rate_hz, frame_samples, capacity_frames) = unsafe {
                (
                    (*probe_header).magic,
                    (*probe_header).version,
                    (*probe_header).sample_rate_hz,
                    (*probe_header).frame_samples,
                    (*probe_header).capacity_frames,
                )
            };
            // SAFETY: `probe_view` was returned by the `MapViewOfFile`
            // call directly above and hasn't been unmapped yet.
            unsafe { let _ = UnmapViewOfFile(probe_view); }

            if magic != PROTOCOL_MAGIC {
                unsafe { CloseHandle(handle).ok() };
                bail!("section '{name}' exists but its magic ({magic:#x}) doesn't match this crate's protocol ({PROTOCOL_MAGIC:#x}) - refusing to trust it");
            }
            if version != PROTOCOL_VERSION {
                unsafe { CloseHandle(handle).ok() };
                bail!("section '{name}' is protocol version {version}, this build expects {PROTOCOL_VERSION}");
            }
            if sample_rate_hz != dsp_core::SAMPLE_RATE_HZ {
                unsafe { CloseHandle(handle).ok() };
                bail!("section '{name}' writer uses sample_rate_hz={sample_rate_hz}, this build expects {}", dsp_core::SAMPLE_RATE_HZ);
            }

            let total_size = RingHeader::total_size(capacity_frames, frame_samples);
            // SAFETY: `handle` refers to a section at least this large -
            // `total_size` was computed from the header's own declared
            // `capacity_frames`/`frame_samples`, i.e. exactly what the
            // writer sized the section to.
            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total_size) };
            if view.Value.is_null() {
                unsafe { CloseHandle(handle).ok() };
                bail!("MapViewOfFile (full view) returned a null view");
            }

            let header_ptr = view.Value.cast::<RingHeader>();
            let data_ptr = unsafe { view.Value.cast::<u8>().add(std::mem::size_of::<RingHeader>()) }.cast::<f32>();

            Ok(Self {
                _handle: handle,
                view,
                header: NonNull::new(header_ptr).expect("checked non-null above"),
                data: NonNull::new(data_ptr).expect("derived from a non-null, in-bounds view"),
                capacity_frames,
                frame_samples,
            })
        }

        /// Pops the oldest unread frame into `out` (`out.len()` must equal
        /// `frame_samples`). Returns `false` (leaving `out` untouched) if
        /// the ring is currently empty - never blocks.
        pub fn pop_frame(&self, out: &mut [f32]) -> bool {
            debug_assert_eq!(out.len(), self.frame_samples as usize);
            // SAFETY: `self.header` points at a valid, live `RingHeader`
            // for this mapping's entire lifetime (owned by `self`).
            let header = unsafe { self.header.as_ref() };

            let read_idx = header.read_index.load(Ordering::Relaxed);
            let write_idx = header.write_index.load(Ordering::Acquire);
            if read_idx >= write_idx {
                return false;
            }

            let slot = (read_idx % self.capacity_frames as u64) as usize;
            // SAFETY: `slot < capacity_frames`, and `self.data` points at
            // `capacity_frames * frame_samples` valid, readable `f32`s for
            // this mapping's lifetime; `out.len() == frame_samples`
            // (debug-checked above).
            unsafe {
                let src = self.data.as_ptr().add(slot * self.frame_samples as usize);
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), self.frame_samples as usize);
            }

            header.read_index.store(read_idx + 1, Ordering::Release);
            true
        }

        pub fn dropped_frames(&self) -> u64 {
            // SAFETY: see `pop_frame`.
            unsafe { self.header.as_ref() }.dropped_frames.load(Ordering::Relaxed)
        }
    }

    impl Drop for SharedRingReader {
        fn drop(&mut self) {
            // SAFETY: see `SharedRingWriter::drop` - identical reasoning.
            unsafe {
                let _ = UnmapViewOfFile(self.view);
                let _ = CloseHandle(self._handle);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn unique_name(tag: &str) -> String {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("Local\\ClearNAITransportTest_{tag}_{nonce}_{}", std::process::id())
        }

        #[test]
        fn writer_then_reader_round_trips_a_real_frame() {
            let name = unique_name("roundtrip");
            let writer = SharedRingWriter::create(&name, 8, dsp_core::FRAME_SAMPLES as u32).unwrap();
            let reader = SharedRingReader::open(&name).unwrap();

            let frame: Vec<f32> = (0..dsp_core::FRAME_SAMPLES).map(|i| i as f32 * 0.001).collect();
            assert!(writer.push_frame(&frame));

            let mut out = vec![0.0f32; dsp_core::FRAME_SAMPLES];
            assert!(reader.pop_frame(&mut out));
            assert_eq!(out, frame);

            // Empty now - a second pop must report "nothing there" rather
            // than returning stale/garbage data.
            let mut out2 = vec![-1.0f32; dsp_core::FRAME_SAMPLES];
            assert!(!reader.pop_frame(&mut out2));
            assert_eq!(out2, vec![-1.0f32; dsp_core::FRAME_SAMPLES], "must not touch `out` on empty");
        }

        #[test]
        fn full_ring_drops_and_counts_rather_than_blocking() {
            let name = unique_name("overflow");
            let writer = SharedRingWriter::create(&name, 2, 4).unwrap();
            let reader = SharedRingReader::open(&name).unwrap();

            assert!(writer.push_frame(&[1.0, 2.0, 3.0, 4.0]));
            assert!(writer.push_frame(&[5.0, 6.0, 7.0, 8.0]));
            // Ring (capacity 2) is now full without the reader ever having
            // popped - the third push must be dropped, not block.
            assert!(!writer.push_frame(&[9.0, 10.0, 11.0, 12.0]));
            assert_eq!(reader.dropped_frames(), 1);

            let mut out = [0.0f32; 4];
            assert!(reader.pop_frame(&mut out));
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0], "oldest frame first, not the dropped one");
        }

        #[test]
        fn wraps_around_past_capacity_correctly() {
            let name = unique_name("wraparound");
            let writer = SharedRingWriter::create(&name, 2, 1).unwrap();
            let reader = SharedRingReader::open(&name).unwrap();

            for n in 0..10 {
                assert!(writer.push_frame(&[n as f32]));
                let mut out = [0.0f32];
                assert!(reader.pop_frame(&mut out));
                assert_eq!(out, [n as f32], "frame {n} round-tripped correctly after wraparound");
            }
            assert_eq!(reader.dropped_frames(), 0, "reader kept up, nothing should have dropped");
        }

        #[test]
        fn reader_rejects_a_section_with_wrong_magic() {
            // Simulates "opened some unrelated section that happens to
            // share a name" - the reader must refuse it, not
            // misinterpret garbage bytes as a valid header.
            let name = unique_name("wrongmagic");
            {
                // Create a same-named section via the writer, then
                // corrupt its magic directly through a second mapping to
                // simulate a foreign/incompatible section.
                let writer = SharedRingWriter::create(&name, 4, 4).unwrap();
                // SAFETY: test-only direct header mutation to simulate
                // corruption/incompatibility, not a real usage pattern.
                unsafe {
                    (*writer.header.as_ptr()).magic = 0xDEADBEEF;
                }
            }
            let result = SharedRingReader::open(&name);
            assert!(result.is_err(), "must reject a section with the wrong magic");
        }
    }
}

#[cfg(windows)]
pub use windows_impl::{SharedRingReader, SharedRingWriter};
