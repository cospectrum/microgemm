fn add_f32(a: f32, b: f32) -> f32 {
    a + b
}

fn mul_f32(a: f32, b: f32) -> f32 {
    a * b
}

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

pub unsafe fn vaddq_f32(float32x4_t(a): float32x4_t, float32x4_t(b): float32x4_t) -> float32x4_t {
    let mut out = vmovq_n_f32(0f32);
    for i in 0..4 {
        out.0[i] = add_f32(a[i], b[i]);
    }
    out
}

pub unsafe fn vmulq_n_f32(float32x4_t(a): float32x4_t, b: f32) -> float32x4_t {
    let mut out = vmovq_n_f32(0f32);
    for (dst, value) in out.0.iter_mut().zip(a) {
        *dst = mul_f32(value, b);
    }
    out
}

pub unsafe fn vfmaq_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    float32x4_t(c): float32x4_t,
) -> float32x4_t {
    assert!(0 <= LANE && LANE < 4);
    let scalar = c[usize::try_from(LANE).unwrap()];
    fma_scalar(a, b, scalar)
}

// A non-generic arithmetic boundary lets Kani abstract values while retaining
// the lane assertion/indexing above (Kani 0.65 cannot stub const generics).
fn fma_scalar(
    float32x4_t(a): float32x4_t,
    float32x4_t(b): float32x4_t,
    scalar: f32,
) -> float32x4_t {
    let mut out = float32x4_t([0.0; 4]);
    for i in 0..4 {
        out.0[i] = add_f32(a[i], mul_f32(b[i], scalar));
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
        out.0[i] = add_f32(a[i], mul_f32(b[i], c[i]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            #[cfg(miri)]
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn loads_and_stores_preserve_all_bits(bits in any::<[u32; 4]>(), offset in 0usize..4) {
            let mut input = [0.0; 7];
            for (dst, bits) in input[offset..offset + 4].iter_mut().zip(bits) {
                *dst = f32::from_bits(bits);
            }
            let mut output = [f32::from_bits(0x7fc0_1234); 9];
            // SAFETY: each pointer covers four initialized f32 values and the
            // destination is a separate, exclusively borrowed allocation.
            unsafe {
                let lanes = vld1q_f32(input.as_ptr().add(offset));
                vst1q_f32(output.as_mut_ptr().add(offset + 1), lanes);
            }
            for (i, value) in output.iter().enumerate() {
                let expected = if (offset + 1..offset + 5).contains(&i) {
                    bits[i - offset - 1]
                } else {
                    0x7fc0_1234
                };
                prop_assert_eq!(value.to_bits(), expected);
            }
        }

        #[test]
        fn arithmetic_and_lane_selection(
            a in prop::array::uniform4(-16i16..=16),
            b in prop::array::uniform4(-16i16..=16),
            c in prop::array::uniform4(-16i16..=16),
        ) {
            let a = a.map(f32::from);
            let b = b.map(f32::from);
            let c = c.map(f32::from);
            let (av, bv, cv) = (float32x4_t(a), float32x4_t(b), float32x4_t(c));
            // Small integers make every operation exact, so this checks lane
            // selection without claiming that the unfused mock models FMA rounding.
            // SAFETY: arithmetic mocks have no pointer preconditions; every lane is valid.
            unsafe {
                prop_assert_eq!(vmovq_n_f32(c[0]).0, [c[0]; 4]);
                prop_assert_eq!(vaddq_f32(av, bv).0, core::array::from_fn(|i| a[i] + b[i]));
                prop_assert_eq!(vmulq_n_f32(av, c[0]).0, a.map(|v| v * c[0]));
                prop_assert_eq!(vfmaq_f32(av, bv, cv).0, core::array::from_fn(|i| a[i] + b[i] * c[i]));
                prop_assert_eq!(vfmaq_laneq_f32::<0>(av, bv, cv).0, core::array::from_fn(|i| a[i] + b[i] * c[0]));
                prop_assert_eq!(vfmaq_laneq_f32::<1>(av, bv, cv).0, core::array::from_fn(|i| a[i] + b[i] * c[1]));
                prop_assert_eq!(vfmaq_laneq_f32::<2>(av, bv, cv).0, core::array::from_fn(|i| a[i] + b[i] * c[2]));
                prop_assert_eq!(vfmaq_laneq_f32::<3>(av, bv, cv).0, core::array::from_fn(|i| a[i] + b[i] * c[3]));

                #[cfg(not(miri))]
                {
                    use core::arch::aarch64 as neon;
                    // SAFETY: AArch64 NEON is enabled for these tests and each
                    // array supplies four initialized elements for loads/stores.
                    let na = neon::vld1q_f32(a.as_ptr());
                    let nb = neon::vld1q_f32(b.as_ptr());
                    let nc = neon::vld1q_f32(c.as_ptr());
                    let mut actual = [0.0; 4];
                    macro_rules! compare {
                        ($real:expr, $mock:expr) => {
                            neon::vst1q_f32(actual.as_mut_ptr(), $real);
                            prop_assert_eq!(actual, $mock.0);
                        };
                    }
                    compare!(neon::vmovq_n_f32(c[0]), vmovq_n_f32(c[0]));
                    compare!(neon::vaddq_f32(na, nb), vaddq_f32(av, bv));
                    compare!(neon::vmulq_n_f32(na, c[0]), vmulq_n_f32(av, c[0]));
                    compare!(neon::vfmaq_f32(na, nb, nc), vfmaq_f32(av, bv, cv));
                    compare!(neon::vfmaq_laneq_f32::<0>(na, nb, nc), vfmaq_laneq_f32::<0>(av, bv, cv));
                    compare!(neon::vfmaq_laneq_f32::<1>(na, nb, nc), vfmaq_laneq_f32::<1>(av, bv, cv));
                    compare!(neon::vfmaq_laneq_f32::<2>(na, nb, nc), vfmaq_laneq_f32::<2>(av, bv, cv));
                    compare!(neon::vfmaq_laneq_f32::<3>(na, nb, nc), vfmaq_laneq_f32::<3>(av, bv, cv));
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "0 <= LANE && LANE < 4")]
    fn rejects_negative_lane() {
        let v = float32x4_t([1.0; 4]);
        // The mock explicitly checks lane bounds before indexing.
        unsafe { vfmaq_laneq_f32::<-1>(v, v, v) };
    }

    #[test]
    #[should_panic(expected = "0 <= LANE && LANE < 4")]
    fn rejects_lane_past_end() {
        let v = float32x4_t([1.0; 4]);
        // The mock explicitly checks lane bounds before indexing.
        unsafe { vfmaq_laneq_f32::<4>(v, v, v) };
    }
}

// Explicit stubs for memory proofs. Whole-vector arithmetic is independent of
// addresses; abstracting it avoids thousands of irrelevant scalar operations.
// Pointer loads/stores remain the real mocks above. Lane bounds remain checked.
#[cfg(kani)]
#[allow(dead_code)]
pub(super) mod memory_arithmetic {
    use super::float32x4_t;

    pub fn add(_: float32x4_t, _: float32x4_t) -> float32x4_t {
        float32x4_t(kani::any())
    }

    pub fn scale(_: float32x4_t, _: f32) -> float32x4_t {
        float32x4_t(kani::any())
    }

    pub fn fma(_: float32x4_t, _: float32x4_t, _: float32x4_t) -> float32x4_t {
        float32x4_t(kani::any())
    }

    pub fn scalar_fma(_: float32x4_t, _: float32x4_t, _: f32) -> float32x4_t {
        float32x4_t(kani::any())
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    // No arithmetic stubs. Every sign pattern over {-1, 1} is exact under
    // both this scalar model and fused NEON FMA; all four lanes are checked.
    #[kani::proof]
    #[kani::unwind(5)]
    fn exact_vector_fma() {
        let abits: [bool; 4] = kani::any();
        let bbits: [bool; 4] = kani::any();
        let cbits: [bool; 4] = kani::any();
        let mut a = [0.0; 4];
        let mut b = [0.0; 4];
        let mut c = [0.0; 4];
        for i in 0..4 {
            a[i] = if abits[i] { 1.0 } else { -1.0 };
            b[i] = if bbits[i] { 1.0 } else { -1.0 };
            c[i] = if cbits[i] { 1.0 } else { -1.0 };
        }
        // SAFETY: this arithmetic model has no pointer or CPU preconditions.
        let result = unsafe { vfmaq_f32(float32x4_t(a), float32x4_t(b), float32x4_t(c)) };
        let lane: usize = kani::any_where(|&i| i < 4);
        let expected = match (abits[lane], bbits[lane] == cbits[lane]) {
            (true, true) => 2.0,
            (false, false) => -2.0,
            _ => 0.0,
        };
        assert_eq!(result.0[lane], expected);
    }
}
