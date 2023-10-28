# wasm_simd128 example

## Build

Add this to your `.cargo/config` before running `cargo build`
```toml
[build]
target = "wasm32-unknown-unknown"
rustflags = ["-C", "target-feature=+simd128"]
```

Alternatively you can set `RUSTFLAGS`
```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --release
```
