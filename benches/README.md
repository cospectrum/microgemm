# Benches

## Gemm

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Cache Utils

Get cache info
```sh
sysctl -a | grep 'cachesize'
```

Run helper script for block sizes
```sh
python3 cache.py
```
