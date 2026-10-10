use std::env;
use std::fs;
use std::path::Path;

fn copy_native_libs(src: &Path, target: &Path) {
    if let Ok(entries) = fs::read_dir(src) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                copy_native_libs(&path, target);
            } else {
                let should_copy = if cfg!(target_os = "windows") {
                    path.extension().and_then(|s| s.to_str()) == Some("dll")
                } else if cfg!(target_os = "linux") {
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .map(|name| name.contains(".so"))
                        .unwrap_or(false)
                } else {
                    false
                };

                if should_copy {
                    if let Some(name) = path.file_name() {
                        let dest = target.join(name);
                        let _ = fs::copy(&path, dest);
                    }
                }
            }
        }
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=CRISPASR_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CRISPASR_EXTRACT_DIR");

    let dirs = [
        env::var("CRISPASR_LIB_DIR").ok(),
        env::var("CRISPASR_EXTRACT_DIR").ok(),
    ];

    if let Ok(out_dir) = env::var("OUT_DIR") {
        if let Some(target_dir) = Path::new(&out_dir).ancestors().nth(3) {
            for dir in dirs.into_iter().flatten() {
                let trimmed = dir.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let path = Path::new(trimmed);
                if path.is_dir() {
                    copy_native_libs(path, target_dir);
                }
            }
        }
    }
}
