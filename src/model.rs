use faer::Mat;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::axis::{AsComplexAxis, Axis, IntoAxis};
use crate::emt::{RealSectionModel, StateSpaceModel, supports_real_sections};
use crate::error::{Result, VecfitError};
use crate::fit::{
    AutoPoles, Options, ProblemRef, Report, SampleMatrix, SampleMatrixRef, WeightStrategy,
    apply_sample_weights, compute_inverse_magnitude_weights, initial_poles,
    matrix_from_row_major_slice, pole_basis_matrix, solve_least_squares_scaled, validate_weights,
};
use crate::shape::{IntoResponse, Layout, ResponseSample, Shape};

/// Prevents division by zero when scaling the relocation solution.
const RELOCATION_SCALE_FLOOR: f64 = 1e-30;

/// Stabilizes the relative pole-shift calculation near the origin.
const SHIFT_NORM_EPSILON: f64 = 1e-15;

/// Tolerance for averaging nearby poles into conjugate pairs.
const CONJUGATE_MATCH_TOLERANCE_SCALE: f64 = 1e-6;

/// Poles with real part at or below this threshold are considered stable.
const POLE_STABILITY_THRESHOLD: f64 = 1e-12;

/// Fractional part of the golden ratio, used for deterministic perturbation.
const GOLDEN_RATIO_FRACT: f64 = 0.618_033_988;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectedResponseBatch {
    values: Vec<Complex64>,
    shape: Shape,
    layout: Layout,
}

/// Component parts for constructing a [`Model`] directly via [`Model::from_parts`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParts {
    pub poles: Vec<Complex64>,
    pub residues: Vec<Complex64>,
    pub channels: usize,
    pub constant_terms: Vec<Complex64>,
    pub proportional_terms: Vec<Complex64>,
    pub shape: Shape,
    pub layout: Layout,
    pub report: Report,
}

/// Fitted rational model with shared poles and per-channel residues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub(crate) poles: Vec<Complex64>,
    pub(crate) residues: Vec<Complex64>,
    pub(crate) channels: usize,
    pub(crate) constant_terms: Vec<Complex64>,
    pub(crate) proportional_terms: Vec<Complex64>,
    pub(crate) shape: Shape,
    pub(crate) layout: Layout,
    pub(crate) report: Report,
}

/// Per-channel error metrics returned by [`Model::channel_errors`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelErrors {
    /// Absolute RMSE per channel.
    pub abs_rmse: Vec<f64>,
    /// Relative RMSE per channel.
    pub rel_rmse: Vec<f64>,
}

impl Model {
    /// Build a model from explicit coefficients and metadata.
    pub fn from_parts(parts: ModelParts) -> Result<Self> {
        let model = Self {
            poles: parts.poles,
            residues: parts.residues,
            channels: parts.channels,
            constant_terms: parts.constant_terms,
            proportional_terms: parts.proportional_terms,
            shape: parts.shape,
            layout: parts.layout,
            report: parts.report,
        };
        model.validate()?;
        Ok(model)
    }

    /// Fit a model from sample points and a response closure.
    ///
    /// The axis wrapper determines how sample points map to the complex plane.
    /// Use [`hz`](crate::hz), [`rad`](crate::rad), [`real`](crate::real),
    /// or [`complex`](crate::complex) to construct the axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vecfit::{Model, Options, hz};
    ///
    /// let freq: Vec<f64> = (1..=60).map(|k| k as f64).collect();
    /// let model = Model::fit(
    ///     hz(&freq),
    ///     |f| 1.0 / (1.0 + 2.0 * std::f64::consts::PI * f),
    ///     Options::new().poles(4),
    /// )?;
    /// # Ok::<(), vecfit::VecfitError>(())
    /// ```
    pub fn fit<A, R, F>(axis: Axis<'_, A>, response_for: F, options: Options) -> Result<Self>
    where
        A: IntoAxis,
        R: IntoResponse,
        F: Fn(A::Point) -> R,
    {
        let complex_axis = axis.to_complex();
        let collected = collect_response_batch(axis.points, |sample| response_for(*sample))?;
        let problem = build_fit_problem(
            &complex_axis,
            &collected.values,
            &collected.shape,
            collected.layout,
        )?;
        Self::fit_problem(problem, options)
    }

    /// Fit from pre-computed flat sample data with an axis wrapper.
    ///
    /// Use this when you already have the complex response values as a flat buffer
    /// rather than a closure.
    pub fn fit_samples<A: IntoAxis>(
        axis: Axis<'_, A>,
        flat_values: &[Complex64],
        shape: Shape,
        options: Options,
    ) -> Result<Self> {
        let complex_axis = axis.to_complex();
        let layout = options.layout;
        let problem = build_fit_problem(&complex_axis, flat_values, &shape, layout)?;
        Self::fit_problem(problem, options)
    }

    /// Fit a model from an explicitly flattened problem description.
    ///
    /// This is the main entry point that handles auto-poles, multi-start, and
    /// weight strategy before delegating to the core fitting loop.
    pub fn fit_problem(problem: ProblemRef<'_>, options: Options) -> Result<Self> {
        problem.validate()?;

        if let Some(ref auto) = options.auto_poles {
            return Self::fit_problem_auto(problem, options.clone(), auto.clone());
        }

        Self::fit_problem_core(problem, options)
    }

    /// Auto-poles: try increasing pole counts until target RMSE is met.
    fn fit_problem_auto(
        problem: ProblemRef<'_>,
        base_options: Options,
        auto: AutoPoles,
    ) -> Result<Self> {
        let min = auto.min_poles.max(1);
        let max = auto.max_poles.max(min);
        let mut best_model: Option<Model> = None;

        let mut n = min;
        while n <= max {
            let mut opts = base_options.clone();
            opts.poles = n;
            opts.auto_poles = None;
            let model = Self::fit_problem_core(problem, opts)?;
            let is_better = best_model
                .as_ref()
                .is_none_or(|prev| model.report.rel_rmse < prev.report.rel_rmse);
            if is_better {
                if model.report.rel_rmse <= auto.target_rel_rmse {
                    return Ok(model);
                }
                best_model = Some(model);
            }
            n += if base_options.real_only { 1 } else { 2 };
        }

        best_model.ok_or_else(|| {
            VecfitError::InvalidInput("auto-poles produced no valid fit".to_string())
        })
    }

    /// Core fitting loop with multi-start support.
    fn fit_problem_core(problem: ProblemRef<'_>, options: Options) -> Result<Self> {
        if options.poles == 0 {
            return Err(VecfitError::InvalidInput(
                "poles must be at least 1".to_string(),
            ));
        }

        // Resolve weights: explicit > strategy-derived > none
        let auto_weights;
        let weights = if let Some(w) = problem.weights.or(options.weights.as_deref()) {
            Some(w)
        } else {
            match options.weight_strategy {
                WeightStrategy::InverseMagnitude => {
                    auto_weights = compute_inverse_magnitude_weights(
                        problem.response.values,
                        problem.response.samples,
                        problem.response.channels,
                    );
                    Some(auto_weights.as_slice())
                }
                WeightStrategy::None => None,
            }
        };
        validate_weights(weights, problem.axis.len())?;

        let channels = problem.response.channels;
        let sample_matrix = matrix_from_row_major_slice(
            problem.response.values,
            problem.response.samples,
            channels,
        );

        let starting_poles = match &options.initial_poles {
            Some(user_poles) => {
                if user_poles.len() != options.poles {
                    return Err(VecfitError::InvalidInput(format!(
                        "initial pole count {} does not match poles {}",
                        user_poles.len(),
                        options.poles
                    )));
                }
                user_poles.clone()
            }
            None => initial_poles(problem.axis, options.poles, options.real_only),
        };

        // First attempt
        let mut best = run_single_fit(
            problem.axis,
            &sample_matrix,
            &starting_poles,
            weights,
            &options,
            problem,
        )?;

        // Multi-start: perturb and retry if RMSE is still high
        if options.max_restarts > 0 && best.report.rel_rmse > options.restart_threshold {
            for trial in 1..=options.max_restarts {
                let perturbed = perturb_poles(&best.poles, trial);
                match run_single_fit(
                    problem.axis,
                    &sample_matrix,
                    &perturbed,
                    weights,
                    &options,
                    problem,
                ) {
                    Ok(candidate) => {
                        if candidate.report.rel_rmse < best.report.rel_rmse {
                            best = candidate;
                            if best.report.rel_rmse <= options.restart_threshold {
                                best.report.restarts = trial;
                                break;
                            }
                        }
                    }
                    Err(_) => continue,
                }
                best.report.restarts = trial;
            }
        }

        Ok(best)
    }

    /// Validate that the model coefficients satisfy the invariants assumed by evaluation and export paths.
    pub fn validate(&self) -> Result<()> {
        if self.channels == 0 {
            return Err(VecfitError::Dimension(
                "model must have at least one channel".to_string(),
            ));
        }
        if self.shape.channels() != self.channels {
            return Err(VecfitError::Dimension(format!(
                "model shape {:?} expects {} channels but model declares {}",
                self.shape.dims(),
                self.shape.channels(),
                self.channels
            )));
        }

        let expected_residues = self.poles.len().checked_mul(self.channels).ok_or_else(|| {
            VecfitError::InvalidInput("model residue matrix is too large".to_string())
        })?;
        if self.residues.len() != expected_residues {
            return Err(VecfitError::Dimension(format!(
                "model residue count {} does not match {} poles x {} channels",
                self.residues.len(),
                self.poles.len(),
                self.channels
            )));
        }
        if self.constant_terms.len() != self.channels {
            return Err(VecfitError::Dimension(format!(
                "constant term count {} does not match channel count {}",
                self.constant_terms.len(),
                self.channels
            )));
        }
        if self.proportional_terms.len() != self.channels {
            return Err(VecfitError::Dimension(format!(
                "proportional term count {} does not match channel count {}",
                self.proportional_terms.len(),
                self.channels
            )));
        }
        Ok(())
    }

    /// Return the fitted poles (shared across all channels).
    pub fn poles(&self) -> &[Complex64] {
        &self.poles
    }

    /// Return the number of fitted poles.
    pub fn pole_count(&self) -> usize {
        self.poles.len()
    }

    /// Return the per-channel residues in row-major order: `residues[pole * channels + channel]`.
    pub fn residues(&self) -> &[Complex64] {
        &self.residues
    }

    /// Return the number of output channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Return the per-channel constant (d) terms.
    pub fn constant_terms(&self) -> &[Complex64] {
        &self.constant_terms
    }

    /// Return the per-channel proportional (e) terms.
    pub fn proportional_terms(&self) -> &[Complex64] {
        &self.proportional_terms
    }

    /// Return the recorded output shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Return the memory layout used for flattened output.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Return the fit report with convergence and error statistics.
    pub fn report(&self) -> &Report {
        &self.report
    }

    /// Return the absolute root-mean-square error of the fit.
    pub fn abs_rmse(&self) -> f64 {
        self.report.abs_rmse
    }

    /// Return the relative root-mean-square error of the fit.
    pub fn rel_rmse(&self) -> f64 {
        self.report.rel_rmse
    }

    /// Evaluate the model into a flat `(sample, channel)` matrix.
    ///
    /// Accepts `&[Complex64]`, `&Vec<Complex64>`, or any typed axis wrapper
    /// (`hz(&freq)`, `rad(&omega)`, `complex(&s)`, etc.).
    pub fn eval_flat(&self, axis: impl AsComplexAxis) -> Result<SampleMatrix> {
        self.validate()?;
        let a = axis.as_complex_axis();
        self.eval_flat_raw(&a)
    }

    /// Evaluate the model and reconstruct each sample into its recorded output shape.
    pub fn eval(&self, axis: impl AsComplexAxis) -> Result<Vec<ResponseSample<Complex64>>> {
        let a = axis.as_complex_axis();
        let flat = self.eval_flat_raw(&a)?;
        (0..flat.samples)
            .map(|row| ResponseSample::new(flat.row(row).to_vec(), self.shape.clone(), self.layout))
            .collect()
    }

    /// Evaluate the model as scalar samples.
    pub fn eval_scalar(&self, axis: impl AsComplexAxis) -> Result<Vec<Complex64>> {
        self.eval(axis)?
            .into_iter()
            .map(ResponseSample::into_scalar)
            .collect()
    }

    /// Evaluate the model as vector samples.
    pub fn eval_vector(&self, axis: impl AsComplexAxis) -> Result<Vec<Vec<Complex64>>> {
        self.eval(axis)?
            .into_iter()
            .map(ResponseSample::into_vector)
            .collect()
    }

    /// Evaluate the model as matrix samples.
    pub fn eval_matrix(&self, axis: impl AsComplexAxis) -> Result<Vec<Vec<Vec<Complex64>>>> {
        self.eval(axis)?
            .into_iter()
            .map(ResponseSample::into_matrix)
            .collect()
    }

    /// Evaluate the model and return per-channel magnitude in decibels.
    ///
    /// Returns `Vec<Vec<f64>>` where the outer index is channel and the inner
    /// index is frequency point: `result[channel][sample]`.
    pub fn magnitude_db(&self, axis: impl AsComplexAxis) -> Result<Vec<Vec<f64>>> {
        let a = axis.as_complex_axis();
        let flat = self.eval_flat_raw(&a)?;
        let mut result = vec![Vec::with_capacity(flat.samples); self.channels];
        for sample_idx in 0..flat.samples {
            for (ch, ch_vec) in result.iter_mut().enumerate() {
                let val = flat.values[sample_idx * self.channels + ch];
                ch_vec.push(20.0 * val.norm().log10());
            }
        }
        Ok(result)
    }

    /// Evaluate the model and return per-channel phase in degrees.
    ///
    /// Returns `Vec<Vec<f64>>` where the outer index is channel and the inner
    /// index is frequency point: `result[channel][sample]`.
    pub fn phase_deg(&self, axis: impl AsComplexAxis) -> Result<Vec<Vec<f64>>> {
        let a = axis.as_complex_axis();
        let flat = self.eval_flat_raw(&a)?;
        let mut result = vec![Vec::with_capacity(flat.samples); self.channels];
        for sample_idx in 0..flat.samples {
            for (ch, ch_vec) in result.iter_mut().enumerate() {
                let val = flat.values[sample_idx * self.channels + ch];
                ch_vec.push(val.arg().to_degrees());
            }
        }
        Ok(result)
    }

    /// Return the residue for a specific pole and channel.
    ///
    /// Indexed as `residues[pole_idx * channels + channel_idx]`.
    pub fn residue(&self, pole_idx: usize, channel_idx: usize) -> Complex64 {
        self.residues[pole_idx * self.channels + channel_idx]
    }

    /// Return whether all poles have non-positive real parts (stable model).
    pub fn is_stable(&self) -> bool {
        self.report.stable
    }

    /// Export the model as real first-order and second-order sections.
    pub fn real_sections(&self) -> Result<RealSectionModel> {
        RealSectionModel::from_model(self)
    }

    /// Export the model as a continuous-time state-space realization.
    pub fn state_space(&self) -> Result<StateSpaceModel> {
        StateSpaceModel::from_model(self)
    }

    /// Per-sample error magnitude between the model and reference data.
    ///
    /// Returns a vector of length `axis.len()` where each entry is the
    /// RMS error across channels at that frequency point.
    pub fn frequency_error(
        &self,
        axis: impl AsComplexAxis,
        reference_values: &[Complex64],
    ) -> Result<Vec<f64>> {
        let a = axis.as_complex_axis();
        let predicted = self.eval_flat_raw(&a)?;
        let channels = self.channels;
        let mut errors = Vec::with_capacity(a.len());
        for sample_idx in 0..a.len() {
            let mut sample_error_sq = 0.0;
            for ch in 0..channels {
                let idx = sample_idx * channels + ch;
                let diff = reference_values[idx] - predicted.values[idx];
                sample_error_sq += diff.norm_sqr();
            }
            errors.push((sample_error_sq / channels as f64).sqrt());
        }
        Ok(errors)
    }

    /// Per-sample, per-channel complex error between the model and reference data.
    ///
    /// Returns a `SampleMatrix` where each element is `reference - predicted`.
    pub fn frequency_error_matrix(
        &self,
        axis: impl AsComplexAxis,
        reference_values: &[Complex64],
    ) -> Result<SampleMatrix> {
        let a = axis.as_complex_axis();
        let predicted = self.eval_flat_raw(&a)?;
        let errors: Vec<Complex64> = reference_values
            .iter()
            .zip(predicted.values.iter())
            .map(|(r, p)| r - p)
            .collect();
        SampleMatrix::new(errors, a.len(), self.channels)
    }

    /// Per-channel RMSE recomputed against the given reference data.
    ///
    /// Useful for evaluating fit quality on a validation axis different
    /// from the one used during fitting.
    pub fn channel_errors(
        &self,
        axis: impl AsComplexAxis,
        reference_values: &[Complex64],
    ) -> Result<ChannelErrors> {
        let a = axis.as_complex_axis();
        let predicted = self.eval_flat_raw(&a)?;
        let channels = self.channels;
        let samples = a.len();
        let mut error_energy = vec![0.0f64; channels];
        let mut signal_energy = vec![0.0f64; channels];
        for sample_idx in 0..samples {
            for ch in 0..channels {
                let idx = sample_idx * channels + ch;
                let diff = reference_values[idx] - predicted.values[idx];
                error_energy[ch] += diff.norm_sqr();
                signal_energy[ch] += reference_values[idx].norm_sqr();
            }
        }
        let n = samples.max(1) as f64;
        Ok(ChannelErrors {
            abs_rmse: error_energy.iter().map(|e| (e / n).sqrt()).collect(),
            rel_rmse: error_energy
                .iter()
                .zip(signal_energy.iter())
                .map(|(e, s)| (e / n).sqrt() / ((s / n).sqrt().max(1e-15)))
                .collect(),
        })
    }

    /// Retrieve pole history as Complex64 values.
    ///
    /// Returns `None` if `Options::track_pole_history` was not enabled.
    /// Each inner vector corresponds to one iteration of the relocation loop.
    pub fn pole_history(&self) -> Option<Vec<Vec<Complex64>>> {
        if self.report.pole_history.is_empty() {
            return None;
        }
        Some(
            self.report
                .pole_history
                .iter()
                .map(|snapshot| {
                    snapshot
                        .iter()
                        .map(|&[re, im]| Complex64::new(re, im))
                        .collect()
                })
                .collect(),
        )
    }

    /// Return a human-readable summary of the fitted model.
    pub fn summary(&self) -> String {
        format!(
            "Model: {} poles, {} channels, shape {:?}\n\
             RMSE: {:.3e} (abs), {:.3e} (rel)\n\
             Converged: {} ({} iterations, {} restarts)\n\
             Stable: {}, Real-section export: {}",
            self.pole_count(),
            self.channels(),
            self.shape().dims(),
            self.abs_rmse(),
            self.rel_rmse(),
            self.report().converged,
            self.report().iterations,
            self.report().restarts,
            self.report().stable,
            self.report().real_sections_valid,
        )
    }
}

// ---------------------------------------------------------------------------
// Private evaluation core
// ---------------------------------------------------------------------------

impl Model {
    /// Core evaluation on a raw `&[Complex64]` axis. Used by all public eval
    /// methods and internal error computation.
    fn eval_flat_raw(&self, axis: &[Complex64]) -> Result<SampleMatrix> {
        let pole_response = Mat::from_fn(axis.len(), self.poles.len(), |sample_idx, pole_idx| {
            Complex64::new(1.0, 0.0) / (axis[sample_idx] - self.poles[pole_idx])
        });
        let residues = matrix_from_row_major_slice(&self.residues, self.poles.len(), self.channels);
        let mut values = Vec::with_capacity(axis.len() * self.channels);
        for (sample_idx, sample) in axis.iter().enumerate() {
            for channel_idx in 0..self.channels {
                let mut value = self.constant_terms[channel_idx]
                    + *sample * self.proportional_terms[channel_idx];
                for pole_idx in 0..self.poles.len() {
                    value +=
                        pole_response[(sample_idx, pole_idx)] * residues[(pole_idx, channel_idx)];
                }
                values.push(value);
            }
        }
        SampleMatrix::new(values, axis.len(), self.channels)
    }
}

// ---------------------------------------------------------------------------
// Single-fit execution (iteration loop + residue identification)
// ---------------------------------------------------------------------------

fn run_single_fit(
    axis: &[Complex64],
    sample_matrix: &Mat<Complex64>,
    starting_poles: &[Complex64],
    weights: Option<&[f64]>,
    options: &Options,
    problem: ProblemRef<'_>,
) -> Result<Model> {
    let channels = sample_matrix.ncols();
    let mut poles = starting_poles.to_vec();
    let mut report = Report {
        weighted: weights.is_some(),
        ..Report::default()
    };

    // --- Pole relocation iteration ---
    for iteration in 0..options.max_iterations {
        let previous_poles = poles.clone();
        poles = relocate_poles(axis, sample_matrix, &poles, weights, options)?;
        if options.track_pole_history {
            report.pole_history.push(
                poles.iter().map(|p| [p.re, p.im]).collect()
            );
        }
        report.iterations = iteration + 1;
        let relative_shift = poles
            .iter()
            .zip(previous_poles.iter())
            .map(|(next, previous)| {
                (*next - *previous).norm() / (previous.norm() + SHIFT_NORM_EPSILON)
            })
            .fold(0.0f64, f64::max);
        report.pole_shifts.push(relative_shift);
        report.max_pole_shift = relative_shift;
        if relative_shift < options.tolerance {
            report.converged = true;
            break;
        }
    }

    // --- Enforce conjugate structure (deferred until post-convergence) ---
    if options.real_only {
        for pole in &mut poles {
            pole.im = 0.0;
        }
    } else {
        poles = complete_conjugate_pairs(&poles);
    }

    // --- Residue identification ---
    let mut fit_basis =
        pole_basis_matrix(axis, &poles, options.fit_constant, options.fit_proportional);
    apply_sample_weights(&mut fit_basis, weights, 1);
    let mut weighted_response = sample_matrix.clone();
    apply_sample_weights(&mut weighted_response, weights, 1);

    let (coefficients, solver_used, svd_fallback_used) =
        solve_least_squares_scaled(&fit_basis, &weighted_response, options.solver)?;
    report.solver_used = solver_used;
    report.svd_fallback_used = svd_fallback_used;

    let pole_count = poles.len();
    let residues = {
        let sub = coefficients.get(..pole_count, ..);
        let mut out = Vec::with_capacity(pole_count * channels);
        for row in 0..pole_count {
            for col in 0..channels {
                out.push(sub[(row, col)]);
            }
        }
        out
    };
    let constant_terms =
        extract_constant_terms(&coefficients, pole_count, channels, options.fit_constant);
    let proportional_terms = extract_proportional_terms(
        &coefficients,
        pole_count,
        channels,
        options.fit_constant,
        options.fit_proportional,
    );

    let mut model = Model {
        poles,
        residues,
        channels,
        constant_terms,
        proportional_terms,
        shape: problem.shape.clone(),
        layout: problem.layout,
        report,
    };
    update_model_report(&mut model, axis, problem.response.values)?;
    Ok(model)
}

// ---------------------------------------------------------------------------
// Pole relocation (one iteration of relaxed vector fitting)
// ---------------------------------------------------------------------------

/// Perform one iteration of relaxed vector fitting pole relocation.
///
/// Uses QR-based orthogonal projection (I - QQ^H) to isolate the weight
/// function from the signal subspace, following the Gustavsen formulation
/// as implemented in sgwt / fd-lines.
fn relocate_poles(
    axis: &[Complex64],
    sample_matrix: &Mat<Complex64>,
    poles: &[Complex64],
    weights: Option<&[f64]>,
    options: &Options,
) -> Result<Vec<Complex64>> {
    let samples = sample_matrix.nrows();
    let channels = sample_matrix.ncols();
    let pole_count = poles.len();
    let n1 = pole_count + 1; // phi_sig columns: N Cauchy + 1 constant

    // Phase 1: Build bases
    // phi_sig = [1/(s-p1), ..., 1/(s-pN), 1]  — always includes constant
    let phi_sig = pole_basis_matrix(axis, poles, true, false);
    // fit_basis = [1/(s-p1), ..., 1/(s-pN), (d), (e)]
    let fit_basis = pole_basis_matrix(axis, poles, options.fit_constant, options.fit_proportional);

    // Phase 2: QR factorization of fit_basis → thin Q
    let qr = fit_basis.as_ref().qr();
    #[allow(non_snake_case)]
    let q = qr.compute_thin_Q();

    // Phase 3: Build flat weighted matrix and project out signal subspace
    //   flat[k, ch*n1 + b] = f[k, ch] * phi_sig[k, b]
    //   residual = (I - Q·Q^H) · flat
    //   Then reshape to (K*channels, n1) with per-sample weights
    let mut flat = Mat::<Complex64>::zeros(samples, channels * n1);
    for k in 0..samples {
        for ch in 0..channels {
            let f_val = sample_matrix[(k, ch)];
            for b in 0..n1 {
                flat[(k, ch * n1 + b)] = f_val * phi_sig[(k, b)];
            }
        }
    }

    // (I - QQ^H) · flat  =  flat - Q·(Q^H · flat)
    let qh_flat = q.as_ref().adjoint() * flat.as_ref();
    let projected = &q * &qh_flat;

    let mut rows = Mat::<Complex64>::zeros(samples * channels, n1);
    for k in 0..samples {
        let w = weights.map_or(1.0, |ws| ws[k].sqrt());
        for ch in 0..channels {
            for b in 0..n1 {
                let residual = flat[(k, ch * n1 + b)] - projected[(k, ch * n1 + b)];
                rows[(k * channels + ch, b)] = residual * w;
            }
        }
    }

    // Phase 4: Augment with normalization constraint
    let norm_weight = (rows.nrows() as f64).sqrt();
    let mut system = Mat::<Complex64>::zeros(rows.nrows() + 1, n1);
    for i in 0..rows.nrows() {
        for j in 0..n1 {
            system[(i, j)] = rows[(i, j)];
        }
    }
    for b in 0..n1 {
        let mut constraint = Complex64::new(0.0, 0.0);
        for k in 0..samples {
            let w = weights.map_or(1.0, |ws| ws[k].sqrt());
            constraint += phi_sig[(k, b)] * w;
        }
        system[(rows.nrows(), b)] = constraint * norm_weight;
    }

    let mut rhs = Mat::<Complex64>::zeros(rows.nrows() + 1, 1);
    rhs[(rows.nrows(), 0)] = Complex64::new(samples as f64 * norm_weight, 0.0);

    // Phase 5: Column-scaled solve
    let (solution, _, _) = solve_least_squares_scaled(&system, &rhs, options.solver)?;

    // Phase 6: Extract new poles via eigendecomposition of companion matrix
    let d_tilde = solution[(pole_count, 0)] + Complex64::new(RELOCATION_SCALE_FLOOR, 0.0);
    let mut h_matrix = Mat::<Complex64>::zeros(pole_count, pole_count);
    for i in 0..pole_count {
        h_matrix[(i, i)] = poles[i];
        for j in 0..pole_count {
            h_matrix[(i, j)] -= solution[(j, 0)] / d_tilde;
        }
    }

    let eigenvalues = match h_matrix.as_ref().eigenvalues() {
        Ok(eigs) => eigs,
        Err(_) => return Ok(poles.to_vec()), // NaN guard
    };

    // NaN guard: if any eigenvalue is non-finite, keep previous poles
    if eigenvalues
        .iter()
        .any(|e| !e.re.is_finite() || !e.im.is_finite())
    {
        return Ok(poles.to_vec());
    }

    // Phase 7: Stabilize — force negative real parts
    let mut next_poles: Vec<Complex64> = eigenvalues
        .into_iter()
        .map(|pole| {
            Complex64::new(
                -pole.re.abs(),
                if options.real_only { 0.0 } else { pole.im },
            )
        })
        .collect();
    sort_poles_by_location(&mut next_poles);
    Ok(next_poles)
}

// ---------------------------------------------------------------------------
// Multi-start perturbation
// ---------------------------------------------------------------------------

/// Perturb poles for multi-start restart.
///
/// Each pole's magnitude is scaled by a deterministic factor that depends on
/// the trial number, giving different starting points on each restart.
fn perturb_poles(poles: &[Complex64], trial: usize) -> Vec<Complex64> {
    poles
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            // Deterministic but varied: mix pole index and trial
            let phase = ((i as f64 + 1.0) * (trial as f64 + 1.0) * GOLDEN_RATIO_FRACT) % 1.0;
            let scale = 0.7 + 0.6 * phase; // range [0.7, 1.3]
            Complex64::new(p.re * scale, p.im * scale)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn build_fit_problem<'a>(
    axis: &'a [Complex64],
    flat_values: &'a [Complex64],
    shape: &'a Shape,
    layout: Layout,
) -> Result<ProblemRef<'a>> {
    let response = SampleMatrixRef::new(flat_values, axis.len(), shape.channels())?;
    Ok(ProblemRef {
        axis,
        response,
        weights: None,
        shape,
        layout,
    })
}

fn collect_response_batch<X, R, F>(samples: &[X], response_for: F) -> Result<CollectedResponseBatch>
where
    R: IntoResponse,
    F: Fn(&X) -> R,
{
    let mut values = Vec::new();
    let mut shape: Option<Shape> = None;
    let mut layout = Layout::RowMajor;

    for sample in samples {
        let flattened = response_for(sample).into_response()?;
        if let Some(expected_shape) = &shape {
            if expected_shape != &flattened.shape {
                return Err(VecfitError::Shape(format!(
                    "response shape changed from {:?} to {:?}",
                    expected_shape.dims(),
                    flattened.shape.dims()
                )));
            }
            if layout != flattened.layout {
                return Err(VecfitError::Shape(format!(
                    "response layout changed from {:?} to {:?}",
                    layout, flattened.layout
                )));
            }
        } else {
            layout = flattened.layout;
            shape = Some(flattened.shape.clone());
        }
        values.extend(flattened.values);
    }

    Ok(CollectedResponseBatch {
        values,
        shape: shape.unwrap_or_else(Shape::scalar),
        layout,
    })
}

fn extract_constant_terms(
    coefficients: &Mat<Complex64>,
    pole_count: usize,
    channels: usize,
    fit_constant: bool,
) -> Vec<Complex64> {
    if fit_constant {
        (0..channels)
            .map(|channel_idx| coefficients[(pole_count, channel_idx)])
            .collect()
    } else {
        vec![Complex64::new(0.0, 0.0); channels]
    }
}

fn extract_proportional_terms(
    coefficients: &Mat<Complex64>,
    pole_count: usize,
    channels: usize,
    fit_constant: bool,
    fit_proportional: bool,
) -> Vec<Complex64> {
    if fit_proportional {
        let row = pole_count + usize::from(fit_constant);
        (0..channels)
            .map(|channel_idx| coefficients[(row, channel_idx)])
            .collect()
    } else {
        vec![Complex64::new(0.0, 0.0); channels]
    }
}

fn update_model_report(
    model: &mut Model,
    axis: &[Complex64],
    reference_values: &[Complex64],
) -> Result<()> {
    let predicted = model.eval_flat_raw(axis)?;
    let channels = model.channels;
    let samples = axis.len();

    // Per-channel accumulators
    let mut channel_error_energy = vec![0.0f64; channels];
    let mut channel_signal_energy = vec![0.0f64; channels];

    for sample_idx in 0..samples {
        for ch in 0..channels {
            let idx = sample_idx * channels + ch;
            let error = reference_values[idx] - predicted.values[idx];
            channel_error_energy[ch] += error.norm_sqr();
            channel_signal_energy[ch] += reference_values[idx].norm_sqr();
        }
    }

    // Aggregate
    let total_error: f64 = channel_error_energy.iter().sum();
    let total_signal: f64 = channel_signal_energy.iter().sum();
    let total_count = reference_values.len().max(1) as f64;
    model.report.abs_rmse = (total_error / total_count).sqrt();
    model.report.rel_rmse =
        model.report.abs_rmse / ((total_signal / total_count).sqrt().max(1e-15));

    // Per-channel
    let samples_f64 = samples.max(1) as f64;
    model.report.channel_abs_rmse = channel_error_energy
        .iter()
        .map(|e| (e / samples_f64).sqrt())
        .collect();
    model.report.channel_rel_rmse = channel_error_energy
        .iter()
        .zip(channel_signal_energy.iter())
        .map(|(e, s)| (e / samples_f64).sqrt() / ((s / samples_f64).sqrt().max(1e-15)))
        .collect();

    model.report.stable = model
        .poles
        .iter()
        .all(|pole| pole.re <= POLE_STABILITY_THRESHOLD);
    model.report.real_sections_valid =
        supports_real_sections(model, crate::emt::CONJUGATE_PAIR_TOLERANCE);
    Ok(())
}

fn sort_poles_by_location(poles: &mut [Complex64]) {
    poles.sort_by(|left, right| {
        left.re
            .partial_cmp(&right.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                left.im
                    .partial_cmp(&right.im)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
}

fn complete_conjugate_pairs(poles: &[Complex64]) -> Vec<Complex64> {
    let max_magnitude = poles.iter().map(|pole| pole.norm()).fold(0.0f64, f64::max);
    let tol = CONJUGATE_MATCH_TOLERANCE_SCALE * (max_magnitude + 1.0);
    let mut pole_parts = poles
        .iter()
        .map(|pole| (pole.re, pole.im.abs()))
        .collect::<Vec<_>>();
    pole_parts.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    let mut completed = Vec::with_capacity(poles.len());
    let mut idx = 0usize;
    while idx < pole_parts.len() {
        if pole_parts[idx].1 < tol || idx + 1 == pole_parts.len() {
            completed.push(Complex64::new(pole_parts[idx].0, 0.0));
            idx += 1;
        } else {
            let real_part = (pole_parts[idx].0 + pole_parts[idx + 1].0) / 2.0;
            let imag_part = (pole_parts[idx].1 + pole_parts[idx + 1].1) / 2.0;
            completed.push(Complex64::new(real_part, imag_part));
            completed.push(Complex64::new(real_part, -imag_part));
            idx += 2;
        }
    }

    sort_poles_by_location(&mut completed);
    completed.truncate(poles.len());
    while completed.len() < poles.len() {
        completed.push(Complex64::new(
            completed.last().map_or(-1.0, |pole| pole.re),
            0.0,
        ));
    }
    completed
}

pub(crate) fn has_complete_conjugates(poles: &[Complex64]) -> bool {
    let tol = CONJUGATE_MATCH_TOLERANCE_SCALE
        * (poles.iter().map(|p| p.norm()).fold(0.0f64, f64::max) + 1.0);
    let mut sorted = poles.to_vec();
    sorted.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut i = 0;
    while i < sorted.len() {
        if sorted[i].im.abs() <= tol {
            i += 1;
        } else if i + 1 < sorted.len()
            && (sorted[i + 1].re - sorted[i].re).abs() <= tol
            && (sorted[i + 1].im + sorted[i].im).abs() <= tol
        {
            i += 2;
        } else {
            return false;
        }
    }
    true
}
