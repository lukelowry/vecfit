//! `vecfit` is a pure-Rust crate for relaxed vector fitting of scalar, vector,
//! matrix, and tensor-valued frequency responses.
//!
//! The primary API is intentionally small:
//! - [`Model`] owns a fitted rational approximation
//! - [`FitOptions`] controls the fit
//! - [`Shape`] and [`Layout`] describe flattened outputs
//! - [`CsvSamples`] handles EMT-style magnitude/phase CSV input
//!
//! # Minimal Examples
//!
//! Scalar fit on a complex sample axis:
//!
//! ```rust
//! use num_complex::Complex64;
//! use vecfit::{FitOptions, Model};
//!
//! let axis = (0..80)
//!     .map(|k| Complex64::new(0.0, 1.0 + k as f64))
//!     .collect::<Vec<_>>();
//!
//! let model = Model::fit(
//!     &axis,
//!     |s| {
//!         Complex64::new(0.05, 0.0)
//!             + Complex64::new(1.2, 0.0) / (s + Complex64::new(3.0, 0.0))
//!             + Complex64::new(0.4, 0.0) / (s + Complex64::new(15.0, 0.0))
//!     },
//!     FitOptions::new().poles(4),
//! )?;
//!
//! let fitted = model.evaluate_scalar(&axis)?;
//! assert_eq!(fitted.len(), axis.len());
//! # Ok::<(), vecfit::VecfitError>(())
//! ```
//!
//! Vector-valued fit on a `jw` axis in hertz:
//!
//! ```rust
//! use vecfit::{FitOptions, Model};
//!
//! let frequency_hz = (1..60).map(|k| k as f64).collect::<Vec<_>>();
//! let model = Model::fit_hz(
//!     &frequency_hz,
//!     |hz| {
//!         let w = 2.0 * std::f64::consts::PI * hz;
//!         [1.0 / (1.0 + w), 0.6 / (3.0 + w), 0.3 / (9.0 + w)]
//!     },
//!     FitOptions::new().poles(4),
//! )?;
//!
//! assert_eq!(model.channels(), 3);
//! # Ok::<(), vecfit::VecfitError>(())
//! ```
//!
//! CSV ingestion followed by fitting:
//!
//! ```rust
//! use vecfit::{CsvSamples, FitOptions};
//!
//! let csv =
//!     "freq_Hz,|Y1|,ang_Y1\n1,10,0\n3,8,-10\n10,4,-45\n30,2,-60\n100,1,-80\n300,0.5,-85\n";
//! let parsed = CsvSamples::from_csv(csv)?;
//! let model = parsed.fit(FitOptions::new().poles(3))?;
//!
//! assert!(model.shape().is_scalar());
//! # Ok::<(), vecfit::VecfitError>(())
//! ```
//!
//! JSON round-trip for interchange with external tools:
//!
//! ```rust
//! use num_complex::Complex64;
//! use vecfit::{FitOptions, Model};
//!
//! let axis = (0..48)
//!     .map(|k| Complex64::new(0.0, 2.0 + k as f64))
//!     .collect::<Vec<_>>();
//! let model = Model::fit(
//!     &axis,
//!     |s| vec![
//!         Complex64::new(1.0, 0.0) / (s + Complex64::new(2.0, 0.0)),
//!         Complex64::new(0.5, 0.0) / (s + Complex64::new(8.0, 0.0)),
//!     ],
//!     FitOptions::new().poles(3),
//! )?;
//!
//! let json = model.to_json()?;
//! let loaded = Model::from_json(&json)?;
//! assert_eq!(loaded.channels(), model.channels());
//! # Ok::<(), vecfit::VecfitError>(())
//! ```
//!
//! For runnable examples, see `examples/fit_scalar.rs`,
//! `examples/fit_vector.rs`, `examples/fit_csv.rs`,
//! `examples/json_roundtrip.rs`, and `examples/export_matrix_emt.rs`.

pub mod emt;
pub mod error;
pub mod fit;
pub mod io;
pub mod model;
pub mod shape;

pub use crate::emt::{
    ChannelStateSpace, DiscreteChannelStateSpace, DiscreteStateSpaceModel, DiscretizationMethod,
    RealSection, RealSectionChannel, RealSectionModel, StateSpaceModel,
};
pub use crate::error::{Result, VecfitError};
pub use crate::fit::{
    AutoPoles, FitOptions, ProblemRef, Report, SampleMatrix, SampleMatrixRef, SolverPolicy,
    SolverUsed, WeightStrategy,
};
pub use crate::io::{ComplexModelJson, CsvSamples, RealKernelJsonModel, RealKernelPoleJson};
pub use crate::model::{Model, ModelParts};
pub use crate::shape::{FlatResponse, IntoResponse, Layout, ResponseSample, ResponseScalar, Shape};
