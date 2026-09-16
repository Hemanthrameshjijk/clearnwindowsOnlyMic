# ClearNAI Virtual Audio Driver — Build Plan (NOT built, NOT tested)

**Status: SOURCE-VERIFIED plan + INF skeleton only. No .cpp/.h miniport code
is included here — see "Why no C++ is included" below.**

## What this document is

A concrete starting point for someone with WDK/kernel driver experience to
build a real "ClearNAI Microphone" virtual audio device, based on
Microsoft's own SYSVAD sample (`microsoft/Windows-driver-samples`,
`audio/sysvad`). It is not itself a working driver.

## Why no C++ miniport code is included

SYSVAD's actual device logic (`adapter.cpp`, `basetopo.cpp`, the per-endpoint
miniport files) implements WDM/PortCls kernel interfaces
(`IMiniportWaveRT`, `IPortTopology`, `IAdapterCommon`, etc.). Producing that
code correctly requires copying Microsoft's real source and modifying it in
place with the actual WDK headers open for reference - not reconstructing
it from a written description. Fabricating kernel-mode code from memory
risks subtly wrong interface implementations, which for a kernel driver
means real instability (up to a bugcheck/BSOD) on whatever machine loads it,
not just a compile error. That risk is not worth taking to produce
"source that looks plausible."

## Real starting point (do this first)

1. Clone `https://github.com/microsoft/Windows-driver-samples`.
2. Copy `audio/sysvad/EndpointsCommon/` and the `TabletAudioSample/`
   endpoint folder into a new folder, e.g. `audio/sysvad/ClearNAIMic/`.
3. Open it in Visual Studio with the WDK extension installed. Build the
   *unmodified* sample first, confirm it builds and loads (in test-signing
   mode, on a VM - never a machine you rely on) - this proves the toolchain
   works before any ClearNAI-specific change is made.
4. Only then start renaming: the endpoint's friendly name string, its
   hardware ID, its category GUID, and the INF's device description (see
   skeleton below) - these are plain string/GUID changes, not new kernel
   logic, and are safe to make without deep miniport knowledge.
5. The actual "capture ClearNAI's processed audio and expose it as this
   endpoint's input" logic requires either extending the existing sample's
   wave miniport to pull from a named shared-memory ring buffer (the
   `RtAudioEngine`-copying pattern the DDK docs describe) or writing a
   simpler user-mode "virtual audio endpoint" via a different, higher-level
   API - this is real, non-trivial driver engineering, not a rename.

## INF file skeleton (real syntax, safe to write - just needs real values)

This follows the same structure as `ComponentizedAudioSample.inf` in the
real repo. Placeholder GUIDs below are NOT real - generate real ones with
`[System.Guid]::NewGuid()` in PowerShell before ever installing this.

```inf
; ClearNAIMic.inf - adapted from Microsoft's SYSVAD ComponentizedAudioSample.inf
[Version]
Signature   = "$WINDOWS NT$"
Class       = MEDIA
ClassGuid   = {4d36e96c-e325-11ce-bfc1-08002be10318}
Provider    = %ProviderName%
CatalogFile = ClearNAIMic.cat
DriverVer   = ; set by build tooling, not by hand
PnpLockdown = 1

[Manufacturer]
%ManufacturerName% = ClearNAI, NT$ARCH$.10.0...16299

[ClearNAI.NT$ARCH$.10.0...16299]
%ClearNAIMic.DeviceDesc% = ClearNAIMic_Install, ROOT\ClearNAIMic

; --- The rest (service install section, coinstallers, wave/topology
; references) must mirror ComponentizedAudioSample.inf's real sections
; line-for-line except for names/GUIDs - copy that file, don't
; hand-retype it, so nothing subtly drifts from a working reference. ---

[Strings]
ProviderName        = "ClearNAI"
ManufacturerName    = "ClearNAI"
ClearNAIMic.DeviceDesc = "ClearNAI Microphone"
```

## Realistic scope reminder

See the earlier time estimate given in conversation: ~6-10 weeks with an
experienced Windows driver engineer, ~4-7 months without one, plus
Microsoft Hardware Dev Center attestation signing turnaround before it can
run on any machine outside test-signing mode. This plan shortens the
"where do I even start" question; it does not shorten that timeline.
