# Contributing

All pull requests are welcome.

## TODO

- Improve performance of "packing". <br>
Explore possible optimizations.
- Add neon kernels for f64.

## Code style

1. No allocations.

2. Prefer `for el in it` instead of `it.for_each(|el| ..)` to make model-checking easier,
since `for_each` uses a while loop internally.

3. All used `simd intrinsics` must have a `mock` implementation, since model-checking cannot be
performed directly on real `simd intrinsics`. <br>
If you are using `simd intrinsic`, make sure to import it along with the `mock` implementation, for example:
```rust
#[cfg(any(kani, miri))]
use simd_mock::intrinsic;
#[cfg(not(any(kani, miri)))]
use simd::intrinsic;
```

4. All `microkernels` must have property-based tests and working CI job for their arhitecture.

5. Use `assert` to check the invariants that are necessary for `safety`, use `debug_assert` for all other invariants.
