//! Axis types for specifying how sample points map to the complex plane.
//!
//! Free functions [`hz`], [`rad`], [`real`], and [`complex`] construct typed
//! axis wrappers that are passed to [`Model::fit`](crate::Model::fit) and
//! evaluation helpers.
//!
//! The [`c`] function is a shorthand for [`Complex64::new`].

use std::borrow::Cow;
use std::marker::PhantomData;

use num_complex::Complex64;

/// Trait for types that describe how sample points map to `Complex64`.
pub trait IntoAxis {
    /// The user-facing sample point type passed to response closures.
    type Point: Copy;

    /// Convert a slice of user points into the complex axis used internally.
    fn into_axis(points: &[Self::Point]) -> Vec<Complex64>;
}

/// A sample axis paired with its interpretation.
#[derive(Debug, Clone, Copy)]
pub struct Axis<'a, A: IntoAxis> {
    /// The raw sample points.
    pub points: &'a [A::Point],
    _marker: PhantomData<A>,
}

impl<'a, A: IntoAxis> Axis<'a, A> {
    /// Build the complex axis used internally by the fitting algorithm.
    pub fn to_complex(&self) -> Vec<Complex64> {
        A::into_axis(self.points)
    }

    /// Return the number of sample points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Return whether the axis is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Marker types
// ---------------------------------------------------------------------------

/// Marker: axis values are already `Complex64`.
pub struct ComplexAxis;

impl IntoAxis for ComplexAxis {
    type Point = Complex64;

    fn into_axis(points: &[Complex64]) -> Vec<Complex64> {
        points.to_vec()
    }
}

/// Marker: axis values are frequency in hertz, mapped to `j·2π·f`.
pub struct Hz;

impl IntoAxis for Hz {
    type Point = f64;

    fn into_axis(points: &[f64]) -> Vec<Complex64> {
        points
            .iter()
            .map(|f| Complex64::new(0.0, 2.0 * std::f64::consts::PI * f))
            .collect()
    }
}

/// Marker: axis values are angular frequency in rad/s, mapped to `j·ω`.
pub struct RadPerSec;

impl IntoAxis for RadPerSec {
    type Point = f64;

    fn into_axis(points: &[f64]) -> Vec<Complex64> {
        points.iter().map(|w| Complex64::new(0.0, *w)).collect()
    }
}

/// Marker: axis values are real-valued, mapped onto the real line.
pub struct RealAxis;

impl IntoAxis for RealAxis {
    type Point = f64;

    fn into_axis(points: &[f64]) -> Vec<Complex64> {
        points.iter().map(|x| Complex64::new(*x, 0.0)).collect()
    }
}

// ---------------------------------------------------------------------------
// Free-function constructors
// ---------------------------------------------------------------------------

/// Wrap a frequency-in-hertz slice as an axis (mapped to `j·2π·f`).
pub fn hz(freq: &[f64]) -> Axis<'_, Hz> {
    Axis {
        points: freq,
        _marker: PhantomData,
    }
}

/// Wrap an angular-frequency slice as an axis (mapped to `j·ω`).
pub fn rad(omega: &[f64]) -> Axis<'_, RadPerSec> {
    Axis {
        points: omega,
        _marker: PhantomData,
    }
}

/// Wrap a real-valued slice as an axis (mapped to the real line).
pub fn real(x: &[f64]) -> Axis<'_, RealAxis> {
    Axis {
        points: x,
        _marker: PhantomData,
    }
}

/// Wrap a `Complex64` slice as an axis (used directly).
pub fn complex(s: &[Complex64]) -> Axis<'_, ComplexAxis> {
    Axis {
        points: s,
        _marker: PhantomData,
    }
}

/// Shorthand for `Complex64::new(re, im)`.
///
/// Most real constants (`1.0`, `5.0`, etc.) work directly in arithmetic with
/// `Complex64` values — use `c` only when you need a genuinely complex constant.
///
/// ```rust
/// use vecfit::c;
/// use num_complex::Complex64;
///
/// let s = Complex64::new(0.0, 10.0);
/// // With c():
/// let h = 0.05 + c(0.8, -0.5) / (s + c(5.0, -50.0));
/// // Without:
/// let h2 = 0.05 + Complex64::new(0.8, -0.5) / (s + Complex64::new(5.0, -50.0));
/// assert_eq!(h, h2);
/// ```
pub fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

// ---------------------------------------------------------------------------
// Evaluation axis trait
// ---------------------------------------------------------------------------

/// Anything that can serve as a complex evaluation axis.
///
/// This trait is implemented for `&[Complex64]`, `&Vec<Complex64>`, and all
/// typed [`Axis`] wrappers, so evaluation methods accept any of them directly.
pub trait AsComplexAxis {
    /// Borrow or build the underlying complex slice.
    fn as_complex_axis(&self) -> Cow<'_, [Complex64]>;
}

impl<'a> AsComplexAxis for &'a [Complex64] {
    fn as_complex_axis(&self) -> Cow<'a, [Complex64]> {
        Cow::Borrowed(self)
    }
}

impl<'a> AsComplexAxis for &'a Vec<Complex64> {
    fn as_complex_axis(&self) -> Cow<'a, [Complex64]> {
        Cow::Borrowed(self.as_slice())
    }
}

impl<'a, A: IntoAxis> AsComplexAxis for Axis<'a, A> {
    fn as_complex_axis(&self) -> Cow<'a, [Complex64]> {
        Cow::Owned(self.to_complex())
    }
}
