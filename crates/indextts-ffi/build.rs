// Build script for cbindgen - simplified

fn main() {
    // Skip cbindgen for now as it may not be installed
    // Manual header generation is recommended
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=indextts.h");
}
