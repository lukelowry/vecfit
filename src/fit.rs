use faer::{Mat, prelude::SolveLstsq};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::error::{Result, VecfitError};
use crate::shape::{Layout, Shape};

/// Borrowed flat `(sample, channel)` response data.
#[derive(Debug, Clone, Copy)]
pub struct SampleMatrixRef<'a> {
    pub values: &'a [Complex64],
    pub samples: usize,
    pub channels: usize,
}

impl<'a> SampleMatrixRef<'a> {
    pub fn new(values: &'a [Complex64], samples: usize, channels: usize) -> Result<Self> {
        if samples == 0 {
            return Err(VecfitError::Dimension(
                "sample matrix must have at least one row".to_string(),
            ));
        }
        if channels == 0 {
            return Err(VecfitError::Dimension(
                "sample matrix must have at least one channel".to_string(),
            ));
        }
        if values.len() != samples * channels {
            return Err(VecfitError::Dimension(format!(
                "sample matrix length {} does not match {samples}x{channels}",
                values.len()
            )));
        }
        Ok(Self {
            values,
            samples,
            channels,
        })
    }
}

/// Owned flat `(sample, channel)` response data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleMatrix {
    pub values: Vec<Complex64>,
    pub samples: usize,
    pub channels: usize,
}

impl SampleMatrix {
    pub fn new(values: Vec<Complex64>, samples: usize, channels: usize) -> Result<Self> {
        SampleMatrixRef::new(&values, samples, channels)?;
        Ok(Self {
            values,
            samples,
            channels,
        })
    }

    pub fn as_ref(&self) -> SampleMatrixRef<'_> {
        SampleMatrixRef {
            values: &self.values,
            samples: self.samples,
            channels: self.channels,
        }
    }

    pub fn row(&self, idx: usize) -> &[Complex64] {
        let start = idx * self.channels;
        &self.values[start..start + self.channels]
    }
}

/// Complete borrowed fitting problem description.
#[derive(Debug, Clone, Copy)]
pub struct ProblemRef<'a> {
    pub axis: &'a [Complex64],
    pub response: SampleMatrixRef<'a>,
    pub weights: Option<&'a [f64]>,
    pub shape: &'a Shape,
    pub layout: Layout,
}

impl<'a> ProblemRef<'a> {
    pub fn validate(&self) -> Result<()> {
        if self.axis.is_empty() {
            return Err(VecfitError::InvalidInput(
                "sample axis cannot be empty".to_string(),
            ));
        }
        if self.response.samples != self.axis.len() {
            return Err(VecfitError::Dimension(format!(
                "response rows {} do not match sample length {}",
                self.response.samples,
                self.axis.len()
            )));
        }
        if self.response.channels != self.shape.channels() {
            return Err(VecfitError::Dimension(format!(
                "response channels {} do not match shape {:?}",
                self.response.channels,
                self.shape.dims()
            )));
        }
        validate_weights(self.weights, self.axis.len())?;
        Ok(())
    }
}

/// Least-squares backend preference used during fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SolverPolicy {
    #[default]
    Auto,
    ColPivQr,
    SvdOnly,
}

/// Least-squares backend that actually produced the final solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverUsed {
    ColPivQr,
    Svd,
}

/// Automatic per-sample weighting strategy for the fitting objective.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum WeightStrategy {
    /// No automatic weighting (default). Minimizes absolute error.
    #[default]
    None,
    /// Weight each sample by `1 / max_channel(|f(s_k)|)`, floored to avoid
    /// division by zero.  Minimizes relative error uniformly across frequencies.
    InverseMagnitude,
}

/// Automatically search for the best pole count within a range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoPoles {
    /// Smallest pole count to try (must be >= 1).
    pub min_poles: usize,
    /// Largest pole count to try.
    pub max_poles: usize,
    /// Stop early when relative RMSE drops below this target.
    pub target_rel_rmse: f64,
}

impl Default for AutoPoles {
    fn default() -> Self {
        Self {
            min_poles: 2,
            max_poles: 30,
            target_rel_rmse: 1e-3,
        }
    }
}

/// Tuning knobs for relaxed vector fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Options {
    /// Number of poles to fit.
    pub poles: usize,
    /// User-supplied starting poles (must match `poles` in length).
    pub initial_poles: Option<Vec<Complex64>>,
    /// Maximum pole-relocation iterations before stopping.
    pub max_iterations: usize,
    /// Convergence threshold on the relative pole shift.
    pub tolerance: f64,
    /// Whether to include a constant (d) term in the model.
    pub fit_constant: bool,
    /// Whether to include a proportional (e * s) term in the model.
    pub fit_proportional: bool,
    /// Constrain all poles to the real axis.
    pub real_only: bool,
    /// Explicit per-sample weights (length must match the sample axis).
    pub weights: Option<Vec<f64>>,
    /// Least-squares backend preference.
    pub solver: SolverPolicy,
    /// Automatic per-sample weighting strategy.
    pub weight_strategy: WeightStrategy,
    /// Maximum multi-start restart attempts.
    pub max_restarts: usize,
    /// Relative RMSE below which restarts stop early.
    pub restart_threshold: f64,
    /// Automatic pole-count search configuration.
    pub auto_poles: Option<AutoPoles>,
    /// Record pole positions at each iteration for migration diagnostics.
    pub track_pole_history: bool,
    /// Memory layout for flattened output (default: RowMajor).
    pub layout: Layout,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            poles: 6,
            initial_poles: None,
            max_iterations: 30,
            tolerance: 1e-9,
            fit_constant: true,
            fit_proportional: false,
            real_only: false,
            weights: None,
            solver: SolverPolicy::Auto,
            weight_strategy: WeightStrategy::None,
            max_restarts: 3,
            restart_threshold: 0.05,
            auto_poles: None,
            track_pole_history: false,
            layout: Layout::RowMajor,
        }
    }
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn poles(mut self, poles: usize) -> Self {
        self.poles = poles;
        self
    }

    pub fn initial_poles(mut self, poles: Vec<Complex64>) -> Self {
        self.initial_poles = Some(poles);
        self
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn fit_constant(mut self, fit_constant: bool) -> Self {
        self.fit_constant = fit_constant;
        self
    }

    pub fn fit_proportional(mut self, fit_proportional: bool) -> Self {
        self.fit_proportional = fit_proportional;
        self
    }

    pub fn real_only(mut self, real_only: bool) -> Self {
        self.real_only = real_only;
        self
    }

    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn solver(mut self, solver: SolverPolicy) -> Self {
        self.solver = solver;
        self
    }

    pub fn weight_strategy(mut self, strategy: WeightStrategy) -> Self {
        self.weight_strategy = strategy;
        self
    }

    pub fn max_restarts(mut self, max_restarts: usize) -> Self {
        self.max_restarts = max_restarts;
        self
    }

    pub fn restart_threshold(mut self, threshold: f64) -> Self {
        self.restart_threshold = threshold;
        self
    }

    pub fn auto_poles(mut self, auto_poles: AutoPoles) -> Self {
        self.auto_poles = Some(auto_poles);
        self
    }

    pub fn track_pole_history(mut self, track: bool) -> Self {
        self.track_pole_history = track;
        self
    }

    pub fn layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    /// Shorthand for `Options::new().poles(n)`.
    pub fn with_poles(n: usize) -> Self {
        Self::new().poles(n)
    }

    /// Automatic pole-count search with default configuration.
    pub fn auto() -> Self {
        Self::new().auto_poles(AutoPoles::default())
    }

    /// Real-only fit with the given pole count.
    pub fn real(n: usize) -> Self {
        Self::new().poles(n).real_only(true)
    }

    /// Weighted fit with inverse-magnitude strategy.
    pub fn weighted(n: usize) -> Self {
        Self::new()
            .poles(n)
            .weight_strategy(WeightStrategy::InverseMagnitude)
    }

    /// Set convergence parameters (max iterations and tolerance).
    pub fn convergence(mut self, max_iter: usize, tol: f64) -> Self {
        self.max_iterations = max_iter;
        self.tolerance = tol;
        self
    }

    /// Set multi-start restart parameters.
    pub fn restarts(mut self, max: usize, threshold: f64) -> Self {
        self.max_restarts = max;
        self.restart_threshold = threshold;
        self
    }
}

/// Summary statistics describing the last fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Whether the pole-relocation loop converged within `max_iterations`.
    pub converged: bool,
    /// Number of pole-relocation iterations performed.
    pub iterations: usize,
    /// Absolute root-mean-square error between the model and the reference data.
    pub abs_rmse: f64,
    /// Relative root-mean-square error (absolute RMSE divided by signal RMS).
    pub rel_rmse: f64,
    /// Largest relative pole shift in the final iteration.
    pub max_pole_shift: f64,
    /// Relative pole shift at each iteration (for convergence diagnostics).
    pub pole_shifts: Vec<f64>,
    /// Which least-squares backend produced the final solution.
    pub solver_used: SolverUsed,
    /// Whether the SVD fallback was triggered during the fit.
    pub svd_fallback_used: bool,
    /// Whether per-sample weights were applied.
    pub weighted: bool,
    /// Whether all poles have non-positive real parts.
    pub stable: bool,
    /// Whether the model can be exported as real first/second-order sections.
    pub real_sections_valid: bool,
    /// Number of multi-start restarts performed.
    pub restarts: usize,
    /// Per-channel absolute RMSE. Length = channels.
    pub channel_abs_rmse: Vec<f64>,
    /// Per-channel relative RMSE. Length = channels.
    pub channel_rel_rmse: Vec<f64>,
    /// Pole snapshots at each iteration. Only populated when Options::track_pole_history is true.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pole_history: Vec<Vec<[f64; 2]>>,
}

impl Default for Report {
    fn default() -> Self {
        Self {
            converged: false,
            iterations: 0,
            abs_rmse: 0.0,
            rel_rmse: 0.0,
            max_pole_shift: 0.0,
            pole_shifts: Vec::new(),
            solver_used: SolverUsed::ColPivQr,
            svd_fallback_used: false,
            weighted: false,
            stable: false,
            real_sections_valid: false,
            restarts: 0,
            channel_abs_rmse: Vec::new(),
            channel_rel_rmse: Vec::new(),
            pole_history: Vec::new(),
        }
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "VecFit Report")?;
        writeln!(f, "  converged:    {}", self.converged)?;
        writeln!(f, "  iterations:   {}", self.iterations)?;
        writeln!(f, "  restarts:     {}", self.restarts)?;
        writeln!(f, "  abs RMSE:     {:.6e}", self.abs_rmse)?;
        writeln!(f, "  rel RMSE:     {:.6e}", self.rel_rmse)?;
        writeln!(f, "  max pole shift: {:.6e}", self.max_pole_shift)?;
        writeln!(f, "  solver:       {:?}", self.solver_used)?;
        writeln!(f, "  SVD fallback: {}", self.svd_fallback_used)?;
        writeln!(f, "  weighted:     {}", self.weighted)?;
        writeln!(f, "  stable:       {}", self.stable)?;
        write!(f, "  real sections: {}", self.real_sections_valid)?;
        if !self.channel_abs_rmse.is_empty() {
            writeln!(f)?;
            writeln!(f, "  per-channel abs RMSE:")?;
            for (i, rmse) in self.channel_abs_rmse.iter().enumerate() {
                write!(f, "    ch {}: {:.6e}", i, rmse)?;
                if i + 1 < self.channel_abs_rmse.len() {
                    writeln!(f)?;
                }
            }
        }
        if !self.channel_rel_rmse.is_empty() {
            writeln!(f)?;
            writeln!(f, "  per-channel rel RMSE:")?;
            for (i, rmse) in self.channel_rel_rmse.iter().enumerate() {
                write!(f, "    ch {}: {:.6e}", i, rmse)?;
                if i + 1 < self.channel_rel_rmse.len() {
                    writeln!(f)?;
                }
            }
        }
        if !self.pole_history.is_empty() {
            writeln!(f)?;
            write!(
                f,
                "  pole history: {} iterations tracked",
                self.pole_history.len()
            )?;
        }
        Ok(())
    }
}

pub(crate) fn matrix_from_row_major_slice(
    values: &[Complex64],
    rows: usize,
    cols: usize,
) -> Mat<Complex64> {
    Mat::from_fn(rows, cols, |row, col| values[row * cols + col])
}

pub(crate) fn pole_basis_matrix(
    axis: &[Complex64],
    poles: &[Complex64],
    fit_constant: bool,
    fit_proportional: bool,
) -> Mat<Complex64> {
    let cols = poles.len() + usize::from(fit_constant) + usize::from(fit_proportional);
    Mat::from_fn(axis.len(), cols, |row, col| {
        if col < poles.len() {
            Complex64::new(1.0, 0.0) / (axis[row] - poles[col])
        } else if fit_constant && col == poles.len() {
            Complex64::new(1.0, 0.0)
        } else {
            axis[row]
        }
    })
}

pub(crate) fn geometric_space(start: f64, stop: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![(start * stop).sqrt()],
        _ => {
            let log_start = start.log10();
            let log_stop = stop.log10();
            (0..count)
                .map(|idx| {
                    let blend = idx as f64 / (count as f64 - 1.0);
                    10f64.powf(log_start + blend * (log_stop - log_start))
                })
                .collect()
        }
    }
}

/// Generate initial poles for the VF iteration.
///
/// When `real_only` is false, generates complex conjugate pairs with imaginary
/// parts spanning the frequency range of the sample axis — matching the
/// Gustavsen & Semlyen approach for resonant systems.  When `real_only` is
/// true, generates purely real negative poles (suitable for monotonic responses).
pub(crate) fn initial_poles(axis: &[Complex64], poles: usize, real_only: bool) -> Vec<Complex64> {
    let mut magnitudes = axis
        .iter()
        .map(|value| value.im.abs().max(value.norm()))
        .filter(|value| *value > 1e-15)
        .collect::<Vec<_>>();
    if magnitudes.is_empty() {
        magnitudes.push(1.0);
    }
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = (magnitudes[0] * 0.5).max(1e-6);
    let hi = magnitudes[magnitudes.len() - 1] * 1.5;

    if real_only || poles < 2 {
        geometric_space(lo, hi, poles)
            .into_iter()
            .map(|value| Complex64::new(-value * 0.01, 0.0))
            .collect()
    } else {
        let pair_count = poles / 2;
        let has_extra = poles % 2 == 1;
        let n_pts = pair_count + usize::from(has_extra);
        let pts = geometric_space(lo, hi, n_pts);
        let mut result = Vec::with_capacity(poles);
        for (i, &beta) in pts.iter().enumerate() {
            if has_extra && i == n_pts - 1 {
                result.push(Complex64::new(-beta * 0.01, 0.0));
            } else {
                result.push(Complex64::new(-beta * 0.01, beta));
                result.push(Complex64::new(-beta * 0.01, -beta));
            }
        }
        result.truncate(poles);
        result
    }
}

pub(crate) fn apply_sample_weights(
    matrix: &mut Mat<Complex64>,
    weights: Option<&[f64]>,
    rows_per_sample: usize,
) {
    if let Some(weights) = weights {
        let rows_per_sample = rows_per_sample.max(1);
        for row in 0..matrix.nrows() {
            let weight = weights[row / rows_per_sample].sqrt();
            for col in 0..matrix.ncols() {
                matrix[(row, col)] *= weight;
            }
        }
    }
}

pub(crate) fn solve_least_squares(
    system: &Mat<Complex64>,
    rhs: &Mat<Complex64>,
    solver_policy: SolverPolicy,
) -> Result<(Mat<Complex64>, SolverUsed, bool)> {
    if system.nrows() < system.ncols() {
        return Err(VecfitError::InvalidInput(format!(
            "least-squares system is underdetermined ({} rows for {} unknowns); provide more samples or reduce the number of fitted terms",
            system.nrows(),
            system.ncols()
        )));
    }

    match solver_policy {
        SolverPolicy::SvdOnly => {
            let svd = system.as_ref().thin_svd()?;
            let solution = svd.solve_lstsq(rhs.as_ref());
            Ok((solution, SolverUsed::Svd, false))
        }
        SolverPolicy::ColPivQr => {
            let qr = system.as_ref().col_piv_qr();
            let solution = qr.solve_lstsq(rhs.as_ref());
            Ok((solution, SolverUsed::ColPivQr, false))
        }
        SolverPolicy::Auto => {
            let qr = system.as_ref().col_piv_qr();
            let solution = qr.solve_lstsq(rhs.as_ref());
            let all_finite = (0..solution.nrows()).all(|row| {
                (0..solution.ncols()).all(|col| {
                    let v = solution[(row, col)];
                    v.re.is_finite() && v.im.is_finite()
                })
            });
            if !all_finite {
                let svd = system.as_ref().thin_svd()?;
                let fallback = svd.solve_lstsq(rhs.as_ref());
                return Ok((fallback, SolverUsed::Svd, true));
            }
            Ok((solution, SolverUsed::ColPivQr, false))
        }
    }
}

/// Solve a column-scaled least-squares system for better conditioning.
///
/// Scales each column of `system` to unit norm before solving, then unscales
/// the solution.  Returns the same tuple as `solve_least_squares`.
pub(crate) fn solve_least_squares_scaled(
    system: &Mat<Complex64>,
    rhs: &Mat<Complex64>,
    solver_policy: SolverPolicy,
) -> Result<(Mat<Complex64>, SolverUsed, bool)> {
    let cols = system.ncols();
    let mut scaled = system.clone();
    let mut col_norms = vec![0.0f64; cols];
    for j in 0..cols {
        let mut norm_sq = 0.0;
        for i in 0..scaled.nrows() {
            norm_sq += scaled[(i, j)].norm_sqr();
        }
        col_norms[j] = norm_sq.sqrt().max(1e-30);
        let inv = 1.0 / col_norms[j];
        for i in 0..scaled.nrows() {
            scaled[(i, j)] *= inv;
        }
    }
    let (mut solution, solver_used, fallback) = solve_least_squares(&scaled, rhs, solver_policy)?;
    // solution shape: (system_cols, rhs_cols) — unscale each row
    for j in 0..cols {
        let inv = 1.0 / col_norms[j];
        for k in 0..solution.ncols() {
            solution[(j, k)] *= inv;
        }
    }
    Ok((solution, solver_used, fallback))
}

/// Compute inverse-magnitude weights from a flat response buffer.
///
/// For each sample, takes the maximum channel magnitude and returns
/// `1 / max(|f_ch|, floor)`.  The floor prevents division by zero at
/// transmission zeros.
pub(crate) fn compute_inverse_magnitude_weights(
    values: &[Complex64],
    samples: usize,
    channels: usize,
) -> Vec<f64> {
    let mut weights = Vec::with_capacity(samples);
    let mut max_mag = 0.0f64;
    for k in 0..samples {
        let mut sample_max = 0.0f64;
        for ch in 0..channels {
            sample_max = sample_max.max(values[k * channels + ch].norm());
        }
        max_mag = max_mag.max(sample_max);
        weights.push(sample_max);
    }
    let floor = max_mag * 1e-8;
    for w in &mut weights {
        *w = 1.0 / (*w).max(floor);
    }
    weights
}

pub(crate) fn validate_weights(weights: Option<&[f64]>, samples: usize) -> Result<()> {
    if let Some(weights) = weights {
        if weights.len() != samples {
            return Err(VecfitError::Dimension(format!(
                "weights length {} does not match sample length {}",
                weights.len(),
                samples
            )));
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(VecfitError::InvalidInput(
                "weights must be finite and nonnegative".to_string(),
            ));
        }
    }
    Ok(())
}
