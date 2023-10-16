# Benches

## Gemm

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Cache

Cache info
```sh
sysctl -a | grep 'cachesize'
```

Helper script for block sizes
```sh
python3 cache.py
```
