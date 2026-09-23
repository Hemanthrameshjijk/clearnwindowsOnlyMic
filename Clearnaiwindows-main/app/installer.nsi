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
!define APP_EXE "clearnairt.exe"
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

Name "${APP_NAME}"
OutFile "..\target\release\ClearNAI-Setup.exe"
InstallDir "$LOCALAPPDATA\ClearNAI"
; Per-user install: no admin/UAC prompt required, matching this app's own
; behavior (it already only ever writes under the current user's
; %LOCALAPPDATA%, never Program Files or HKLM).
RequestExecutionLevel user

Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

Section "Install"
  SetOutPath "$INSTDIR"
  ; Built by the `cargo build --release` step that runs immediately before
  ; this one in CI (see .github/workflows/windows.yml).
  File "..\target\release\${APP_EXE}"

  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall ${APP_NAME}.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"

  WriteUninstaller "$INSTDIR\Uninstall.exe"

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
  SetOutPath "$TEMP"
  File /nonfatal "${VBCABLE_INSTALLER}"

  IfFileExists "$TEMP\VBCABLE_Setup_x64.exe" 0 skip_vbcable
    DetailPrint "Launching the VB-CABLE driver installer - click Install Driver in the window that opens (it will show its own admin prompt)..."
    ExecWait "$TEMP\VBCABLE_Setup_x64.exe" $0
    DetailPrint "VB-CABLE installer closed (exit code $0)."
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

Section "Uninstall"
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

  DeleteRegKey HKCU "${UNINSTALL_REG_KEY}"
SectionEnd
