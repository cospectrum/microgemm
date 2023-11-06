# Contributing

All pull requests are welcome.

## TODO

- Improve performance of "packing". <br>
Right now it may be slow due to unnecessary zero padding.
Loop ranges can be "stripped".

## Run CI locally

### Requirements

1. rustc 1.65+
2. cargo-make
3. CMake
4. node
5. firefox

### Run

Go to the project root and run:
```sh
cargo make all
```
