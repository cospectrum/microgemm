#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct float32x4_t([f32; 4]);

pub unsafe fn vmovq_n_f32(value: f32) -> float32x4_t {
    float32x4_t([value; 4])
}

pub unsafe fn vld1q_f32(ptr: *const f32) -> float32x4_t {
    let mut out = vmovq_n_f32(0f32);
    for i in 0..4 {
        out.0[i] = *ptr.add(i);
    }
    out
}

pub unsafe fn vst1q_f32(ptr: *mut f32, a: float32x4_t) {
    for i in 0..4 {
        *ptr.add(i) = a.0[i];
    }
}

pub unsafe fn vaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    let mut out = vmovq_n_f32(0f32);
    for i in 0..4 {
        out.0[i] = a.0[i] + b.0[i];
    }
    out
}

pub unsafe fn vmulq_n_f32(a: float32x4_t, b: f32) -> float32x4_t {
    let mut out = vmovq_n_f32(0f32);
    for i in 0..4 {
        out.0[i] = a.0[i] * b;
    }
    out
}

pub unsafe fn vfmaq_laneq_f32<const LANE: i32>(
    float32x4_t(a): float32x4_t,
    float32x4_t(b): float32x4_t,
    float32x4_t(c): float32x4_t,
) -> float32x4_t {
    assert!(0 <= LANE && LANE < 4);
    let scalar = c[usize::try_from(LANE).unwrap()];
    let mut out = vmovq_n_f32(0f32);
    for i in 0..4 {
        out.0[i] = a[i] + b[i] * scalar;
    }
    out
}

pub unsafe fn vfmaq_f32(
    float32x4_t(a): float32x4_t,
    float32x4_t(b): float32x4_t,
    float32x4_t(c): float32x4_t,
) -> float32x4_t {
    let mut out = vmovq_n_f32(0f32);
    for i in 0..4 {
        out.0[i] = c[i] + a[i] * b[i];
    }
    out
}
