; ClearNAI Windows installer (NSIS script).
;
; Builds a proper Windows installer instead of the earlier raw-exe +
; uninstall.ps1 approach: a per-user install (no admin elevation needed),
; Start Menu + Desktop shortcuts, and a real entry in Windows'
; "Apps & Features" / "Add or Remove Programs" backed by an
; NSIS-auto-generated Uninstall.exe.
;
; Deliberately installs to the SAME directory the app already uses for its
; self-extracted BVC assets (%LOCALAPPDATA%\ClearNAI\weya_nc.dll, the model
; tar.gz, onnxruntime.dll, settings.json, clearnai.log - see
; app/src/setup.rs::app_data_dir)
; rather than Program Files: this app never needs admin rights for anything
; else it does, so an admin-requiring install would be an unnecessary extra
; UAC prompt, and installing to the very directory the app already treats
; as "its own" folder means uninstalling can just remove that one directory
; and be done, with no separate "app files" vs "user data" split to keep in
; sync.
;
; Built by CI via `makensis installer.nsi` (see .github/workflows/windows.yml)
; from the repo root's `app` directory, after `cargo build --release` has
; already produced target\release\clearnairt.exe two directories up.

!define APP_NAME "ClearNAI"
; Two programs (see app/src/lib.rs): the window and the background engine that keeps the audio running.
!define APP_EXE "clearnai.exe"
!define ENGINE_EXE "clearnai-engine.exe"
!define APP_PUBLISHER "ClearNAI"
!define APP_VERSION "0.1.0"
!define UNINSTALL_REG_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

; Third-party VB-Audio VB-CABLE driver installer, bundled here ONLY for
; this personal, non-commercial build (see the "VB-CABLE Virtual Audio
; Driver" section below for the licensing/elevation caveats - re-read them
; before ever reusing this for a build given to anyone else). Download the
; real installer yourself from VB-Audio's official site and place the
; extracted exe at this exact path before running makensis - this script
; does not fetch it for you, and `makensis` will fail with a clear
; "file not found" error if it's missing, rather than silently skipping it.
!define VBCABLE_INSTALLER "vendor\VBCABLE_Setup_x64.exe"

; ~500 MB of models: solid LZMA gives the smallest download (slower to build, fine for CI/release).
SetCompressor /SOLID lzma

!include LogicLib.nsh
!include x64.nsh
!include WinVer.nsh

; The VB-CABLE installer arguments. Empty = VB-Audio's own installer window (click "Install Driver"); this is the
; verified-safe default. The community-reported silent switches are `-i -h` (VB-Audio does not document them, and
; Windows still shows its own driver-trust prompt) - test on a clean machine and set this to "-i -h" only if it
; behaves. NEVER use `-u` here (that uninstalls).
!define VBCABLE_ARGS ""

; ClearNAI logo for the installer and uninstaller executables (generated from assets\logo_mark_light.svg).
Icon "assets\clearnai.ico"
UninstallIcon "assets\clearnai.ico"

Name "${APP_NAME}"
OutFile "..\target\release\ClearNAI-Setup.exe"
InstallDir "$LOCALAPPDATA\ClearNAI"
; Two-step install: ClearNAI itself installs per-user with NO admin prompt (the app only ever writes under the current
; user's %LOCALAPPDATA%\ClearNAI, so there is no profile mix-up from a different admin account). Only when the VB-CABLE
; driver is missing does the VB-CABLE step ask for admin (UAC prompt via "runas"); if it is already installed there
; is no prompt at all.
RequestExecutionLevel user
; (shell folders default to the current user, which is what we want, so no SetShellVarContext is needed.)

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "ClearNAI needs 64-bit Windows 10 or 11."
    Abort
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_OK|MB_ICONSTOP "ClearNAI needs Windows 10 or later."
    Abort
  ${EndIf}
FunctionEnd

; Fixed install folder (no directory page): the app keeps its settings, log and models in this same folder and the
; uninstaller removes the whole folder, so letting the user pick a shared folder such as Documents would be dangerous.
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

Section "Install"
  SetOutPath "$INSTDIR"
  ; Built by the `cargo build --release` step that runs immediately before
  ; this one in CI (see .github/workflows/windows.yml).
  ; Stop a running copy first (an update, or the old single-exe build), otherwise its files are locked.
  nsExec::Exec 'taskkill /F /IM ${ENGINE_EXE}'
  nsExec::Exec 'taskkill /F /IM ${APP_EXE}'
  nsExec::Exec 'taskkill /F /IM clearnairt.exe'
  Sleep 1000
  Delete "$INSTDIR\clearnairt.exe"
  File "..\target\release\${APP_EXE}"
  File "..\target\release\${ENGINE_EXE}"

  ; Accent-conversion model bundle (~270 MB), shipped as a folder next to the exe and installed into
  ; %LOCALAPPDATA%\ClearNAI\app_models on first use (setup.rs compares full contents). Put the files in
  ; app\assets\app_models\ before building the installer; /nonfatal = the installer still builds without
  ; them (Accent then reports "unavailable"). ecapa.onnx / librnnoise.dylib from the zip are not needed.
  SetOutPath "$INSTDIR\app_models"
  File /nonfatal "assets\app_models\catalog.json"
  File /nonfatal "assets\app_models\features.npz"
  File /nonfatal "assets\app_models\american_voices.npz"
  File /nonfatal "assets\app_models\converter_best_v6dmel.npz"
  File /nonfatal "assets\app_models\converter_best_v6dmel.onnx"
  File /nonfatal "assets\app_models\converter_best_v6dmel.dec.onnx"
  File /nonfatal "assets\app_models\vocoder_causal_hifigan_best.npz"
  File /nonfatal "assets\app_models\vocoder_causal_hifigan_best.onnx"
  SetOutPath "$INSTDIR"

  ; Live-transcript speech model (~210 MB). The app looks for it in %LOCALAPPDATA%\ClearNAI\asr_models (this
  ; folder). A reinstall removes the install folder, so without shipping it here the model vanished on every
  ; update ("speech model not found"). Put the two files in app\assets\asr_models\ before building the installer
  ; (fast_conformer_with_laughter_MatMul_qu8.ort and vocab.txt); /nonfatal = still builds without them.
  SetOutPath "$INSTDIR\asr_models"
  File /nonfatal "assets\asr_models\fast_conformer_with_laughter_MatMul_qu8.ort"
  File /nonfatal "assets\asr_models\vocab.txt"
  SetOutPath "$INSTDIR"

  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall ${APP_NAME}.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"

  ; The call-audio path needs a SECOND virtual cable (VB-CABLE A+B). VB-Audio's licence does not allow bundling A+B
  ; in another product's installer, so it is not shipped here: the user gets their official download page instead.
  WriteINIStr "$SMPROGRAMS\${APP_NAME}\Get VB-CABLE A+B (call audio).url" "InternetShortcut" "URL" "https://vb-audio.com/Cable/"

  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Start the background engine (hidden, in the tray) at every sign-in. The engine keeps this entry in line with the
  ; "Start with Windows" switch in Advanced.
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${APP_NAME}" '"$INSTDIR\${ENGINE_EXE}" --background'

  ; Per-user uninstall registry entry (HKCU, not HKLM - matches
  ; RequestExecutionLevel user above) so ClearNAI shows up in Windows'
  ; "Apps & Features" / "Add or Remove Programs" with a working Uninstall
  ; button, the same as any normally-installed app.
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "DisplayIcon" "$INSTDIR\${APP_EXE}"
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKCU "${UNINSTALL_REG_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTALL_REG_KEY}" "NoRepair" 1
SectionEnd

Section "VB-CABLE Virtual Audio Driver"
  ; Runs AFTER ClearNAI's own per-user install above (Section "Install") -
  ; "first ClearNAI, then the driver," per explicit instruction. This is a
  ; THIRD-PARTY kernel-mode driver (VB-Audio Software's VB-CABLE), not part
  ; of ClearNAI - see docs/VIRTUAL_DEVICES.md for why ClearNAI needs a
  ; virtual cable at all. Bundled directly into this installer ONLY
  ; because this specific build is for personal, non-commercial use -
  ; VB-Audio's donationware terms do not grant blanket redistribution
  ; rights for a commercial product, so this section must be re-evaluated
  ; (and VB-Audio contacted for explicit permission) before ever reusing
  ; this installer.nsi for a build distributed to other people.
  ;
  ; Driver installation genuinely requires admin rights - unlike
  ; ClearNAI's own per-user install (`RequestExecutionLevel user` above,
  ; deliberately unchanged), VB-CABLE's own installer requests its own
  ; elevation when launched, so Windows shows a UAC prompt for just this
  ; one step rather than requiring this whole installer to run elevated.
  ;
  ; Deliberately launches VB-CABLE's own installer UI rather than guessing
  ; an undocumented silent-install command-line flag - VB-Audio's real,
  ; current switches were not independently verified here, and getting one
  ; wrong (e.g. accidentally uninstalling instead of installing) is worse
  ; than one extra click. Click "Install Driver" in the window that opens.
  ;
  ; `/nonfatal`: this personal build's vendor/ file is deliberately NOT
  ; committed to the repo (see .gitignore) - CI (.github/workflows/windows.yml)
  ; builds this same script without it, and must keep succeeding, just
  ; without this section's behavior, rather than breaking the whole
  ; pipeline over a file that should never be checked in.
  ; Already installed? VB-CABLE's own setup, when it finds an existing install, offers to REMOVE the driver instead
  ; (per VB-Audio's reference manual), so never run it on a machine that already has it. Detection: the driver's
  ; Uninstall key (confirmed present on a machine with VB-CABLE installed: "VB:VBCABLE {87459874-1236-4469}").
  StrCpy $1 ""
  SetRegView 64
  ReadRegStr $1 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VB:VBCABLE {87459874-1236-4469}" "DisplayName"
  ${If} $1 == ""
    SetRegView 32
    ReadRegStr $1 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VB:VBCABLE {87459874-1236-4469}" "DisplayName"
  ${EndIf}
  SetRegView 32
  ${If} $1 != ""
    DetailPrint "VB-CABLE is already installed - skipping the driver step."
    Goto skip_vbcable
  ${EndIf}

  SetOutPath "$TEMP"
  File /nonfatal "${VBCABLE_INSTALLER}"

  IfFileExists "$TEMP\VBCABLE_Setup_x64.exe" 0 skip_vbcable
    DetailPrint "Launching the VB-CABLE driver installer - click Install Driver in the window that opens (it will show its own admin prompt)..."
    ; This installer is not elevated, and ExecWait cannot start a program that needs admin (error 740), so launch it
    ; with the "runas" verb: Windows shows the UAC prompt just for this step, and ExecShellWait waits for it to close
    ; (it reports no exit code).
    ExecShellWait "runas" "$TEMP\VBCABLE_Setup_x64.exe" "${VBCABLE_ARGS}" SW_SHOWNORMAL
    DetailPrint "VB-CABLE installer closed."
    Delete "$TEMP\VBCABLE_Setup_x64.exe"

    ; VB-Audio's own documentation recommends a reboot for the driver to
    ; register reliably - ask, never force a silent unexpected reboot.
    MessageBox MB_YESNO|MB_ICONQUESTION \
      "VB-CABLE setup has finished.$\r$\n$\r$\nA system restart is recommended before using the virtual cable with ClearNAI. Restart now?" \
      IDNO skip_reboot
    Reboot
    skip_reboot:
  skip_vbcable:
SectionEnd

; After everything is installed (including the optional driver step): start the engine, which opens the window once.
Function .onInstSuccess
  ${IfNot} ${Silent}
    Exec '"$INSTDIR\${ENGINE_EXE}"'
  ${EndIf}
FunctionEnd

Section "Uninstall"
  nsExec::Exec 'taskkill /F /IM ${APP_EXE}'
  nsExec::Exec 'taskkill /F /IM ${ENGINE_EXE}'
  nsExec::Exec 'taskkill /F /IM clearnairt.exe'
  Sleep 1000
  ; $INSTDIR is also where the running app self-extracts weya_nc.dll, the
  ; ONNX model bundle, settings.json, and clearnai.log (see
  ; app/src/setup.rs::app_data_dir) - removing the whole directory cleans
  ; all of that up in one step, matching what the earlier uninstall.ps1
  ; script did explicitly.
  ;
  ; Deliberately does NOT touch any VB-Audio Virtual Cable / VoiceMeeter
  ; driver install - those are separate third-party products the user
  ; installed themselves (see docs/VIRTUAL_DEVICES.md) and may still be
  ; needed by other apps.
  RMDir /r "$INSTDIR"

  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall ${APP_NAME}.lnk"
  RMDir "$SMPROGRAMS\${APP_NAME}"
  Delete "$DESKTOP\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Get VB-CABLE A+B (call audio).url"

  ; The app's optional "Start with Windows" setting writes this value (HKCU Run).
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${APP_NAME}"

  DeleteRegKey HKCU "${UNINSTALL_REG_KEY}"
SectionEnd
