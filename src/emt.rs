use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, prelude::Solve};
use serde::{Deserialize, Serialize};

use crate::error::{Result, VecfitError};
use crate::model::{Model, has_complete_conjugates};
use crate::shape::{Layout, Shape};

/// Tolerance for matching conjugate pole pairs during real-section extraction.
pub(crate) const CONJUGATE_PAIR_TOLERANCE: f64 = 1e-8;

/// Proportional terms below this magnitude are treated as zero for state-space export.
const PROPORTIONAL_TERM_FLOOR: f64 = 1e-12;

/// Continuous-to-discrete transform used for EMT state-space export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscretizationMethod {
    BackwardEuler,
    Tustin,
}

/// Real first-order or second-order section for one fitted channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealSection {
    FirstOrder { pole: f64, residue: f64 },
    SecondOrder { a1: f64, a0: f64, b1: f64, b0: f64 },
}

/// Per-channel collection of real sections plus direct/proportional terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSectionChannel {
    pub sections: Vec<RealSection>,
    pub direct: f64,
    pub proportional: f64,
}

/// Real-section representation of a fitted multi-channel model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealSectionModel {
    pub shape: Shape,
    pub layout: Layout,
    pub channels: Vec<RealSectionChannel>,
}

/// Continuous-time state-space model for one fitted channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStateSpace {
    pub a: Vec<f64>,
    pub n_states: usize,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: f64,
    pub proportional: f64,
}

/// Continuous-time state-space export for a fitted multi-channel model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpaceModel {
    pub shape: Shape,
    pub layout: Layout,
    pub channels: Vec<ChannelStateSpace>,
}

/// Discrete-time state-space model for one fitted channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteChannelStateSpace {
    pub a: Vec<f64>,
    pub n_states: usize,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: f64,
    pub dt: f64,
    pub method: DiscretizationMethod,
}

/// Discrete-time state-space export for a fitted multi-channel model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteStateSpaceModel {
    pub shape: Shape,
    pub layout: Layout,
    pub channels: Vec<DiscreteChannelStateSpace>,
}

impl ChannelStateSpace {
    /// Validate that the state-space arrays match `n_states`.
    pub fn validate(&self) -> Result<()> {
        let expected_a = self.n_states.checked_mul(self.n_states).ok_or_else(|| {
            VecfitError::InvalidInput("state-space matrix is too large".to_string())
        })?;
        if self.a.len() != expected_a {
            return Err(VecfitError::Dimension(format!(
                "state-space A length {} does not match {}x{}",
                self.a.len(),
                self.n_states,
                self.n_states
            )));
        }
        if self.b.len() != self.n_states {
            return Err(VecfitError::Dimension(format!(
                "state-space B length {} does not match n_states {}",
                self.b.len(),
                self.n_states
            )));
        }
        if self.c.len() != self.n_states {
            return Err(VecfitError::Dimension(format!(
                "state-space C length {} does not match n_states {}",
                self.c.len(),
                self.n_states
            )));
        }
        Ok(())
    }
}

impl RealSectionModel {
    /// Build a real-section export from a fitted model.
    pub fn from_model(model: &Model) -> Result<Self> {
        validate_real_sections(model, CONJUGATE_PAIR_TOLERANCE)?;

        let mut channels = vec![
            RealSectionChannel {
                sections: Vec::new(),
                direct: 0.0,
                proportional: 0.0,
            };
            model.channels
        ];
        for (idx, value) in model.constant_terms.iter().enumerate() {
            channels[idx].direct = value.re;
        }
        for (idx, value) in model.proportional_terms.iter().enumerate() {
            channels[idx].proportional = value.re;
        }

        let tol = CONJUGATE_PAIR_TOLERANCE;
        let mut used = vec![false; model.poles.len()];
        for pole_idx in 0..model.poles.len() {
            if used[pole_idx] {
                continue;
            }
            let pole = model.poles[pole_idx];
            if pole.im.abs() <= tol {
                used[pole_idx] = true;
                for (channel_idx, channel) in channels.iter_mut().enumerate().take(model.channels) {
                    let residue = model.residues[pole_idx * model.channels + channel_idx];
                    channel.sections.push(RealSection::FirstOrder {
                        pole: pole.re,
                        residue: residue.re,
                    });
                }
            } else {
                let conj_idx = ((pole_idx + 1)..model.poles.len()).find(|other_idx| {
                    !used[*other_idx]
                        && (model.poles[*other_idx].re - pole.re).abs() <= tol
                        && (model.poles[*other_idx].im + pole.im).abs() <= tol
                });
                let conj_idx = conj_idx.ok_or_else(|| {
                    VecfitError::InvalidInput(
                        "missing conjugate pole for second-order section".to_string(),
                    )
                })?;
                let conjugate = model.poles[conj_idx];
                used[pole_idx] = true;
                used[conj_idx] = true;
                for (channel_idx, channel) in channels.iter_mut().enumerate().take(model.channels) {
                    let residue = model.residues[pole_idx * model.channels + channel_idx];
                    let a1 = -2.0 * pole.re;
                    let a0 = pole.norm_sqr();
                    let b1 = 2.0 * residue.re;
                    let b0 = -2.0 * (residue * conjugate).re;
                    channel
                        .sections
                        .push(RealSection::SecondOrder { a1, a0, b1, b0 });
                }
            }
        }
        Ok(Self {
            shape: model.shape.clone(),
            layout: model.layout,
            channels,
        })
    }
}

impl StateSpaceModel {
    /// Build a continuous-time state-space export from a fitted model.
    pub fn from_model(model: &Model) -> Result<Self> {
        let sections = RealSectionModel::from_model(model)?;
        let channels = sections
            .channels
            .into_iter()
            .map(channel_sections_to_state_space)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shape: sections.shape,
            layout: sections.layout,
            channels,
        })
    }

    /// Discretize each channel using the requested method and sample time.
    pub fn discretize(
        &self,
        dt: f64,
        method: DiscretizationMethod,
    ) -> Result<DiscreteStateSpaceModel> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(VecfitError::InvalidInput(
                "discretization step must be positive".to_string(),
            ));
        }
        let channels = self
            .channels
            .iter()
            .map(|channel| discretize_channel(channel, dt, method))
            .collect::<Result<Vec<_>>>()?;
        Ok(DiscreteStateSpaceModel {
            shape: self.shape.clone(),
            layout: self.layout,
            channels,
        })
    }
}

pub(crate) fn supports_real_sections(model: &Model, tol: f64) -> bool {
    validate_real_sections(model, tol).is_ok()
}

fn validate_real_sections(model: &Model, tol: f64) -> Result<()> {
    model.validate()?;
    if !has_complete_conjugates(&model.poles) {
        return Err(VecfitError::InvalidInput(
            "model does not admit a real-section export".to_string(),
        ));
    }
    if model
        .constant_terms
        .iter()
        .chain(model.proportional_terms.iter())
        .any(|value| value.im.abs() > tol)
    {
        return Err(VecfitError::InvalidInput(
            "model does not admit a real-section export".to_string(),
        ));
    }

    let mut unmatched = vec![true; model.poles.len()];
    for pole_idx in 0..model.poles.len() {
        if !unmatched[pole_idx] {
            continue;
        }
        let pole = model.poles[pole_idx];
        if pole.im.abs() <= tol {
            unmatched[pole_idx] = false;
            for channel_idx in 0..model.channels {
                let residue = model.residues[pole_idx * model.channels + channel_idx];
                if residue.im.abs() > tol {
                    return Err(VecfitError::InvalidInput(
                        "real pole has non-real residue".to_string(),
                    ));
                }
            }
            continue;
        }

        let conj_idx = ((pole_idx + 1)..model.poles.len()).find(|other_idx| {
            unmatched[*other_idx]
                && (model.poles[*other_idx].re - pole.re).abs() <= tol
                && (model.poles[*other_idx].im + pole.im).abs() <= tol
        });
        let conj_idx = conj_idx.ok_or_else(|| {
            VecfitError::InvalidInput("missing conjugate pole for second-order section".to_string())
        })?;
        unmatched[pole_idx] = false;
        unmatched[conj_idx] = false;

        for channel_idx in 0..model.channels {
            let residue = model.residues[pole_idx * model.channels + channel_idx];
            let residue_conj = model.residues[conj_idx * model.channels + channel_idx];
            if (residue_conj.re - residue.re).abs() > tol
                || (residue_conj.im + residue.im).abs() > tol
            {
                return Err(VecfitError::InvalidInput(
                    "complex residue pair is not conjugate".to_string(),
                ));
            }
        }
    }

    Ok(())
}

fn channel_sections_to_state_space(channel: RealSectionChannel) -> Result<ChannelStateSpace> {
    if channel.proportional.abs() > PROPORTIONAL_TERM_FLOOR {
        return Err(VecfitError::InvalidInput(
            "state-space export does not support nonzero proportional term".to_string(),
        ));
    }

    let state_count = channel
        .sections
        .iter()
        .map(|section| match section {
            RealSection::FirstOrder { .. } => 1,
            RealSection::SecondOrder { .. } => 2,
        })
        .sum::<usize>();

    let mut a = Mat::<f64>::zeros(state_count, state_count);
    let mut b = vec![0.0; state_count];
    let mut c = vec![0.0; state_count];
    let mut offset = 0usize;

    for section in channel.sections {
        match section {
            RealSection::FirstOrder { pole, residue } => {
                a[(offset, offset)] = pole;
                b[offset] = 1.0;
                c[offset] = residue;
                offset += 1;
            }
            RealSection::SecondOrder { a1, a0, b1, b0 } => {
                a[(offset, offset)] = 0.0;
                a[(offset, offset + 1)] = 1.0;
                a[(offset + 1, offset)] = -a0;
                a[(offset + 1, offset + 1)] = -a1;
                b[offset + 1] = 1.0;
                c[offset] = b0;
                c[offset + 1] = b1;
                offset += 2;
            }
        }
    }

    Ok(ChannelStateSpace {
        a: mat_to_row_major_f64(&a),
        n_states: state_count,
        b,
        c,
        d: channel.direct,
        proportional: channel.proportional,
    })
}

fn discretize_channel(
    channel: &ChannelStateSpace,
    dt: f64,
    method: DiscretizationMethod,
) -> Result<DiscreteChannelStateSpace> {
    channel.validate()?;
    if channel.proportional.abs() > PROPORTIONAL_TERM_FLOOR {
        return Err(VecfitError::InvalidInput(
            "discretization does not support nonzero proportional term".to_string(),
        ));
    }
    let n = channel.n_states;
    let a = Mat::from_fn(n, n, |i, j| channel.a[i * n + j]);
    let b = Mat::from_fn(n, 1, |i, _| channel.b[i]);
    let c = Mat::from_fn(1, n, |_, j| channel.c[j]);
    let identity = Mat::identity(n, n);
    let (ad, bd, cd, dd) = match method {
        DiscretizationMethod::BackwardEuler => {
            let lhs = &identity - faer::Scale(dt) * &a;
            let lu = lhs.as_ref().partial_piv_lu();
            let ad = lu.solve(identity.as_ref());
            let bd = lu.solve((faer::Scale(dt) * &b).as_ref());
            (ad, bd, c.clone(), channel.d)
        }
        DiscretizationMethod::Tustin => {
            let lhs = &identity - faer::Scale(0.5 * dt) * &a;
            let rhs = &identity + faer::Scale(0.5 * dt) * &a;
            let lu = lhs.as_ref().partial_piv_lu();
            let ad = lu.solve(rhs.as_ref());
            let bd = lu.solve((faer::Scale(dt) * &b).as_ref());
            let inv_lhs = lu.inverse();
            let cd = &c * inv_lhs.as_ref();
            let feed = (&c * lu.solve(b.as_ref()))[(0, 0)];
            (ad, bd, cd, channel.d + 0.5 * dt * feed)
        }
    };

    Ok(DiscreteChannelStateSpace {
        a: mat_to_row_major_f64(&ad),
        n_states: channel.n_states,
        b: mat_to_row_major_f64(&bd),
        c: mat_to_row_major_f64(&cd),
        d: dd,
        dt,
        method,
    })
}

fn mat_to_row_major_f64(mat: &Mat<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(mat.nrows() * mat.ncols());
    for row in 0..mat.nrows() {
        for col in 0..mat.ncols() {
            out.push(mat[(row, col)]);
        }
    }
    out
}
