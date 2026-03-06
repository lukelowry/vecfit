//! Pure-Rust [relaxed vector fitting](https://en.wikipedia.org/wiki/Vector_fitting)
//! for rational approximation of frequency-domain data.
//! Fits scalar, vector, and matrix-valued responses.
//!
//! # API overview
//!
//! - [`Model`] — fitted rational approximation (poles, residues, evaluation)
//! - [`Options`] — solver configuration (pole count, convergence, weighting)
//! - [`Csv`] and [`Touchstone`] — parse sampled data from common file formats
//! - [`hz`], [`rad`], [`real`], [`complex`] — axis wrappers for unit conversion
//! - [`Shape`] and [`Layout`] — describe the structure of flattened response data
//!
//! # Examples
//!
//! Fit sampled data from CSV:
//!
//! ```rust
//! use vecfit::{Csv, Options};
//!
//! let csv = "freq_Hz,re_f1,im_f1\n1,0.95,-0.31\n5,0.35,-0.72\n10,0.10,-0.98\n50,0.02,-1.0\n100,0.01,-1.0\n";
//! let model = Csv::from_csv(csv)?.fit(Options::new().poles(2))?;
//!
//! assert!(model.shape().is_scalar());
//! # Ok::<(), vecfit::VecfitError>(())
//! ```
//!
//! Fit an analytic transfer function via closure:
//!
//! ```rust
//! use num_complex::Complex64;
//! use vecfit::{Options, Model, complex};
//!
//! let s: Vec<Complex64> = (0..80)
//!     .map(|k| Complex64::new(0.0, 1.0 + k as f64))
//!     .collect();
//!
//! let model = Model::fit(
//!     complex(&s),
//!     |s| 0.05 + 1.2 / (s + 3.0) + 0.4 / (s + 15.0),
//!     Options::new().poles(4),
//! )?;
//!
//! let fitted = model.eval_scalar(&s)?;
//! assert_eq!(fitted.len(), s.len());
//! # Ok::<(), vecfit::VecfitError>(())
//! ```
//!
//! Multi-channel vector fit using the `hz` axis wrapper:
//!
//! ```rust
//! use vecfit::{Options, Model, hz};
//!
//! let freq: Vec<f64> = (1..60).map(|k| k as f64).collect();
//! let model = Model::fit(
//!     hz(&freq),
//!     |f| {
//!         let w = 2.0 * std::f64::consts::PI * f;
//!         [1.0 / (1.0 + w), 0.6 / (3.0 + w), 0.3 / (9.0 + w)]
//!     },
//!     Options::new().poles(4),
//! )?;
//!
//! assert_eq!(model.channels(), 3);
//! # Ok::<(), vecfit::VecfitError>(())
//! ```
//!
//! JSON round-trip:
//!
//! ```rust
//! use num_complex::Complex64;
//! use vecfit::{Options, Model, complex};
//!
//! let s: Vec<Complex64> = (0..48)
//!     .map(|k| Complex64::new(0.0, 2.0 + k as f64))
//!     .collect();
//! let model = Model::fit(
//!     complex(&s),
//!     |s| vec![1.0 / (s + 2.0), 0.5 / (s + 8.0)],
//!     Options::new().poles(3),
//! )?;
//!
//! let json = model.to_json()?;
//! let loaded = Model::from_json(&json)?;
//! assert_eq!(loaded.channels(), model.channels());
//! # Ok::<(), vecfit::VecfitError>(())
//! ```

pub mod axis;
pub mod emt;
pub mod error;
pub mod fit;
pub mod io;
pub mod model;
pub mod shape;
pub mod touchstone;

pub use crate::axis::{
    AsComplexAxis, Axis, ComplexAxis, Hz, IntoAxis, RadPerSec, RealAxis, c, complex, hz, rad, real,
};
pub use crate::emt::{
    ChannelStateSpace, DiscreteChannelStateSpace, DiscreteStateSpaceModel, DiscretizationMethod,
    RealSection, RealSectionChannel, RealSectionModel, StateSpaceModel,
};
pub use crate::error::{Result, VecfitError};
pub use crate::fit::{
    AutoPoles, Options, ProblemRef, Report, SampleMatrix, SampleMatrixRef, SolverPolicy,
    SolverUsed, WeightStrategy,
};
pub use crate::io::{
    ComplexModelJson, Csv, ParsedSamples, RealKernelJsonModel, RealKernelPoleJson,
};
pub use crate::model::{ChannelErrors, Model, ModelParts};
pub use crate::shape::{FlatResponse, IntoResponse, Layout, ResponseSample, ResponseScalar, Shape};
pub use crate::touchstone::{
    DataFormat, FrequencyUnit, ParameterType, Touchstone, TouchstoneOptions,
};
