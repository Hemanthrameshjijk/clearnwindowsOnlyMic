// Embeds the app icon into clearnairt.exe so Explorer, the taskbar and the installer shortcuts show the ClearNAI logo.
// Windows targets only; a failure to embed is a warning, never a build error.
fn main() {
    println!("cargo:rerun-if-changed=assets/clearnai.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/clearnai.ico");
        res.set("ProductName", "ClearNAI");
        res.set("FileDescription", "ClearNAI");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the app icon: {e}");
        }
    }
}
