# Contributing

All pull requests are welcome.

## TODO

- Improve performance of "packing". <br>
Right now it may be slow due to unnecessary zero padding.
Loop ranges can be "stripped".

- Improve performance of `generic` kernels. <br>
Currently, genric 4x4/8x8 kernels can run 2-3 times slower than a `NeonKernel` with manual simd and loop unrolling.
It should be possible to help the rust compiler optimize better.

## Run CI locally

### Requirements

1. rustc 1.65+
2. cargo-make
3. node
4. firefox
5. CMake

### Run

Go to the project root and run:
```sh
cargo make all
```
