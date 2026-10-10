use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=CRISPASR_LIB_DIR");

    #[cfg(target_os = "windows")]
    {
        if let Ok(lib_dir) = env::var("CRISPASR_LIB_DIR") {
            let lib_path = Path::new(&lib_dir);
            if lib_path.is_dir() {
                if let Ok(out_dir) = env::var("OUT_DIR") {
                    // target/<profile> is 3 levels up from OUT_DIR
                    if let Some(target_dir) = Path::new(&out_dir).ancestors().nth(3) {
                        if let Ok(entries) = fs::read_dir(lib_path) {
                            for entry in entries.flatten() {
                                let path = entry.path();
                                if path.extension().and_then(|s| s.to_str()) == Some("dll") {
                                    if let Some(name) = path.file_name() {
                                        let dest = target_dir.join(name);
                                        let _ = fs::copy(&path, dest);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
