use std::io::Read;
use std::path::Path;
use std::str::FromStr;

use csv::StringRecord;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::error::{Result, VecfitError};
use crate::fit::{Options, OutputRepresentation, Report, SampleMatrix};
use crate::model::{
    ModalStateSpace, ModalStateSpaceParts, Model, matrix_terms_from_flat, residue_matrix_dims,
};
use crate::shape::{Layout, ResponseSample, Shape};

/// Tolerance for checking whether coefficients are purely real in real-kernel export.
const REAL_COEFFICIENT_TOLERANCE: f64 = 1e-10;

/// Real-valued pole entry used by the real-kernel JSON interchange format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealKernelPoleJson {
    #[serde(rename = "q")]
    pub pole: f64,
    #[serde(rename = "r")]
    pub residues: Vec<f64>,
}

/// Real-kernel JSON model used by EMT-oriented interchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealKernelJsonModel {
    pub description: Option<String>,
    #[serde(rename = "nfuncs")]
    pub n_functions: usize,
    #[serde(rename = "npoles")]
    pub n_poles: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Shape>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<Layout>,
    #[serde(rename = "d")]
    pub direct_terms: Vec<f64>,
    #[serde(rename = "e")]
    pub proportional_terms: Option<Vec<f64>>,
    pub poles: Vec<RealKernelPoleJson>,
}

/// Complex-valued JSON model with shared poles.
///
/// The residue form uses `residues[pole][channel]`. The modal state-space form
/// uses `C[output_row][state]` and `B[state][input_col]` for
/// `H(s) = C(sI - diag(poles))^-1B + D + sE`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexModelJson {
    pub poles: Vec<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residues: Option<Vec<Vec<[f64; 2]>>>,
    #[serde(
        rename = "C",
        alias = "c",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub c: Option<Vec<Vec<[f64; 2]>>>,
    #[serde(
        rename = "B",
        alias = "b",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub b: Option<Vec<Vec<[f64; 2]>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Shape>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<Layout>,
    #[serde(rename = "d")]
    pub direct_terms: Vec<[f64; 2]>,
    #[serde(rename = "e")]
    pub proportional_terms: Vec<[f64; 2]>,
    #[serde(rename = "rmse")]
    pub abs_rmse: f64,
    #[serde(rename = "iters")]
    pub iterations: usize,
}

/// Detected column format for CSV parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CsvFormat {
    /// Magnitude/phase pairs: `|Y1|, ang_Y1, |Y2|, ang_Y2, ...`
    MagnitudePhase,
    /// Real/imaginary pairs: `re_Y1, im_Y1, re_Y2, im_Y2, ...`
    RealImag,
}

/// Format-agnostic parsed frequency-domain samples ready for fitting.
///
/// All format parsers ([`Csv`], [`Touchstone`](crate::Touchstone), etc.) produce
/// this type or delegate to it via `Deref`.
#[derive(Debug, Clone)]
pub struct ParsedSamples {
    frequency_hz: Vec<f64>,
    axis: Vec<Complex64>,
    samples: SampleMatrix,
    shape: Shape,
    layout: Layout,
}

impl ParsedSamples {
    /// Build from pre-computed components.
    pub fn new(
        frequency_hz: Vec<f64>,
        axis: Vec<Complex64>,
        samples: SampleMatrix,
        shape: Shape,
        layout: Layout,
    ) -> Result<Self> {
        if frequency_hz.len() != axis.len() {
            return Err(VecfitError::Dimension(format!(
                "frequency_hz length {} does not match axis length {}",
                frequency_hz.len(),
                axis.len()
            )));
        }
        if samples.samples != axis.len() {
            return Err(VecfitError::Dimension(format!(
                "sample matrix rows {} do not match axis length {}",
                samples.samples,
                axis.len()
            )));
        }
        Ok(Self {
            frequency_hz,
            axis,
            samples,
            shape,
            layout,
        })
    }

    /// Override the inferred output shape.
    pub fn with_shape(mut self, shape: Shape) -> Result<Self> {
        if shape.channels() != self.samples.channels {
            return Err(VecfitError::Dimension(format!(
                "shape {:?} expects {} channels but data has {}",
                shape.dims(),
                shape.channels(),
                self.samples.channels
            )));
        }
        self.shape = shape;
        Ok(self)
    }

    /// Return the original frequency axis in hertz.
    pub fn frequency_hz(&self) -> &[f64] {
        &self.frequency_hz
    }

    /// Return the mapped complex sample axis used for fitting.
    pub fn axis(&self) -> &[Complex64] {
        &self.axis
    }

    /// Return the raw flat `(sample, channel)` sample matrix.
    pub fn samples(&self) -> &SampleMatrix {
        &self.samples
    }

    /// Return the number of parsed samples.
    pub fn len(&self) -> usize {
        self.samples.samples
    }

    /// Return whether the parsed data contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.samples == 0
    }

    /// Return the number of response channels per sample.
    pub fn channels(&self) -> usize {
        self.samples.channels
    }

    /// Return the interpreted output shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Return the interpreted flattened layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Reconstruct each parsed sample into its recorded output shape.
    pub fn responses(&self) -> Result<Vec<ResponseSample<Complex64>>> {
        (0..self.samples.samples)
            .map(|row| {
                ResponseSample::new(
                    self.samples.row(row).to_vec(),
                    self.shape.clone(),
                    self.layout,
                )
            })
            .collect()
    }

    /// Reconstruct the parsed data as scalar samples.
    pub fn scalars(&self) -> Result<Vec<Complex64>> {
        self.responses()?
            .into_iter()
            .map(ResponseSample::into_scalar)
            .collect()
    }

    /// Reconstruct the parsed data as vector samples.
    pub fn vectors(&self) -> Result<Vec<Vec<Complex64>>> {
        self.responses()?
            .into_iter()
            .map(ResponseSample::into_vector)
            .collect()
    }

    /// Reconstruct the parsed data as matrix samples.
    pub fn matrices(&self) -> Result<Vec<Vec<Vec<Complex64>>>> {
        self.responses()?
            .into_iter()
            .map(ResponseSample::into_matrix)
            .collect()
    }

    /// Interpret the parsed samples as scalars.
    pub fn scalar(self) -> Result<Self> {
        self.with_shape(Shape::scalar())
    }

    /// Interpret the parsed samples as a vector of the given length.
    pub fn vector(self, len: usize) -> Result<Self> {
        self.with_shape(Shape::vector(len)?)
    }

    /// Interpret the parsed samples as a matrix with the given dimensions.
    pub fn matrix(self, rows: usize, cols: usize) -> Result<Self> {
        self.with_shape(Shape::matrix(rows, cols)?)
    }

    /// Interpret the parsed samples as a tensor with the given dimensions.
    pub fn tensor<I>(self, dims: I) -> Result<Self>
    where
        I: IntoIterator<Item = usize>,
    {
        self.with_shape(Shape::tensor(dims)?)
    }

    /// Override the flattened layout used when reconstructing samples.
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    /// Compare a fitted model against this parsed data and return per-channel errors.
    pub fn compare(&self, model: &Model) -> Result<crate::model::ChannelErrors> {
        model.channel_errors(&self.axis as &[Complex64], &self.samples.values)
    }

    /// Fit a model directly from the parsed data.
    pub fn fit(&self, options: Options) -> Result<Model> {
        Model::fit_samples(
            crate::axis::complex(&self.axis),
            &self.samples.values,
            self.shape.clone(),
            options,
        )
    }
}

/// Parsed CSV input with both the original frequency axis and mapped imaginary-axis samples.
///
/// Supports two column formats (auto-detected from headers):
/// - **Magnitude/phase**: `freq_Hz, |Y1|, ang_Y1, ...` (angle in degrees)
/// - **Rectangular**: `freq_Hz, re_Y1, im_Y1, ...`
///
/// Also supports tab-separated, semicolon-separated, and custom delimiters
/// via [`from_tsv`](Csv::from_tsv), [`from_ssv`](Csv::from_ssv), and
/// [`from_delimited`](Csv::from_delimited).
#[derive(Debug, Clone)]
pub struct Csv {
    inner: ParsedSamples,
}

impl std::ops::Deref for Csv {
    type Target = ParsedSamples;

    fn deref(&self) -> &ParsedSamples {
        &self.inner
    }
}

impl Csv {
    /// Convert into the format-agnostic parsed samples.
    pub fn into_parsed(self) -> ParsedSamples {
        self.inner
    }

    /// Parse CSV text into complex samples. Format is auto-detected from headers.
    pub fn from_csv(csv_text: &str) -> Result<Self> {
        Self::from_reader_inner(csv_text.as_bytes(), b',')
    }

    /// Parse CSV data from a filesystem path. Format is auto-detected from headers.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let csv_text = std::fs::read_to_string(path)?;
        Self::from_csv(&csv_text)
    }

    /// Parse CSV data from any reader. Format is auto-detected from headers.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self> {
        Self::from_reader_inner(reader, b',')
    }

    /// Parse tab-separated values.
    pub fn from_tsv(text: &str) -> Result<Self> {
        Self::from_delimited(text, b'\t')
    }

    /// Parse semicolon-separated values.
    pub fn from_ssv(text: &str) -> Result<Self> {
        Self::from_delimited(text, b';')
    }

    /// Parse delimited text with a custom field separator.
    pub fn from_delimited(text: &str, delimiter: u8) -> Result<Self> {
        Self::from_reader_inner(text.as_bytes(), delimiter)
    }

    /// Parse delimited data from a filesystem path with a custom field separator.
    pub fn from_path_delimited<P: AsRef<Path>>(path: P, delimiter: u8) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_delimited(&text, delimiter)
    }

    /// Parse and fit in one call.
    pub fn fit_csv(csv_text: &str, options: Options) -> Result<Model> {
        Self::from_csv(csv_text)?.fit(options)
    }

    /// Parse from a file path and fit in one call.
    pub fn fit_path<P: AsRef<Path>>(path: P, options: Options) -> Result<Model> {
        Self::from_path(path)?.fit(options)
    }

    fn from_reader_inner<R: Read>(reader: R, delimiter: u8) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(delimiter)
            .from_reader(reader);
        let headers = reader
            .headers()
            .map_err(|err| VecfitError::Csv(err.to_string()))?
            .clone();
        let (format, channels) = detect_csv_format(&headers)?;

        let mut frequency_hz = Vec::new();
        let mut axis = Vec::new();
        let mut sample_values = Vec::new();
        for record in reader.records() {
            let record = record?;
            let frequency = parse_frequency_hz_column(&record)?;
            frequency_hz.push(frequency);
            axis.push(Complex64::new(0.0, 2.0 * std::f64::consts::PI * frequency));

            for channel_idx in 0..channels {
                let value = match format {
                    CsvFormat::MagnitudePhase => parse_mag_phase_column(&record, channel_idx)?,
                    CsvFormat::RealImag => parse_real_imag_column(&record, channel_idx)?,
                };
                sample_values.push(value);
            }
        }

        let shape = if channels == 1 {
            Shape::scalar()
        } else {
            Shape::vector(channels)?
        };

        Ok(Self {
            inner: ParsedSamples {
                frequency_hz,
                samples: SampleMatrix::new(sample_values, axis.len(), channels)?,
                axis,
                shape,
                layout: Layout::RowMajor,
            },
        })
    }

    /// Override the inferred output shape after parsing.
    pub fn with_shape(mut self, shape: Shape) -> Result<Self> {
        self.inner = self.inner.with_shape(shape)?;
        Ok(self)
    }

    /// Interpret the parsed samples as scalars.
    pub fn scalar(self) -> Result<Self> {
        self.with_shape(Shape::scalar())
    }

    /// Interpret the parsed samples as a vector of the given length.
    pub fn vector(self, len: usize) -> Result<Self> {
        self.with_shape(Shape::vector(len)?)
    }

    /// Interpret the parsed samples as a matrix with the given dimensions.
    pub fn matrix(self, rows: usize, cols: usize) -> Result<Self> {
        self.with_shape(Shape::matrix(rows, cols)?)
    }

    /// Interpret the parsed samples as a tensor with the given dimensions.
    pub fn tensor<I>(self, dims: I) -> Result<Self>
    where
        I: IntoIterator<Item = usize>,
    {
        self.with_shape(Shape::tensor(dims)?)
    }

    /// Override the flattened layout used when reconstructing samples.
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.inner.layout = layout;
        self
    }
}

impl Model {
    /// Deserialize a fitted model from the default complex JSON interchange format.
    pub fn from_json(json_text: &str) -> Result<Self> {
        let json: ComplexModelJson = serde_json::from_str(json_text)?;
        Self::try_from(json)
    }

    /// Deserialize a fitted model from complex JSON stored at a filesystem path.
    pub fn from_json_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json_text = std::fs::read_to_string(path)?;
        Self::from_json(&json_text)
    }

    /// Serialize a fitted model into the default complex JSON interchange format.
    pub fn to_json(&self) -> Result<String> {
        let json = ComplexModelJson::try_from(self)?;
        serde_json::to_string_pretty(&json).map_err(Into::into)
    }

    /// Deserialize a fitted model from the real-kernel JSON interchange format.
    pub fn from_real_json(json_text: &str) -> Result<Self> {
        let json: RealKernelJsonModel = serde_json::from_str(json_text)?;
        Self::try_from(json)
    }

    /// Deserialize a fitted model from real-kernel JSON stored at a filesystem path.
    pub fn from_real_json_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json_text = std::fs::read_to_string(path)?;
        Self::from_real_json(&json_text)
    }

    /// Serialize a fitted model into the real-kernel JSON interchange format.
    pub fn to_real_json(&self, description: Option<String>) -> Result<String> {
        let json = RealKernelJsonModel::try_from((self, description))?;
        serde_json::to_string_pretty(&json).map_err(Into::into)
    }
}

impl FromStr for Csv {
    type Err = VecfitError;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_csv(s)
    }
}

impl TryFrom<ComplexModelJson> for Model {
    type Error = VecfitError;

    fn try_from(json: ComplexModelJson) -> Result<Self> {
        let pole_count = json.poles.len();
        let layout = json.layout.unwrap_or(Layout::RowMajor);
        let has_residues = json.residues.is_some();
        let has_c = json.c.is_some();
        let has_b = json.b.is_some();
        if has_residues && (has_c || has_b) {
            return Err(VecfitError::Serialization(
                "complex JSON must contain either residues or C/B state-space matrices, not both"
                    .to_string(),
            ));
        }
        if has_c != has_b {
            return Err(VecfitError::Serialization(
                "complex JSON state-space form requires both C and B".to_string(),
            ));
        }

        let poles = json
            .poles
            .iter()
            .copied()
            .map(complex_from_pair)
            .collect::<Vec<_>>();

        let channels = if let Some(residue_rows) = &json.residues {
            if residue_rows.len() != pole_count {
                return Err(VecfitError::Serialization(
                    "complex JSON residue rows must match pole count".to_string(),
                ));
            }
            json.direct_terms.len()
        } else if let (Some(c_rows), Some(b_rows)) = (&json.c, &json.b) {
            if pole_count == 0 {
                return Err(VecfitError::Serialization(
                    "complex JSON state-space form requires at least one pole".to_string(),
                ));
            }
            if c_rows.is_empty() {
                return Err(VecfitError::Serialization(
                    "complex JSON C matrix must have at least one row".to_string(),
                ));
            }
            if b_rows.len() != pole_count {
                return Err(VecfitError::Serialization(
                    "complex JSON B matrix row count must match pole count".to_string(),
                ));
            }
            let rows = c_rows.len();
            let cols = b_rows.first().map_or(0, Vec::len);
            rows.checked_mul(cols).ok_or_else(|| {
                VecfitError::Serialization(
                    "complex JSON state-space dimensions are too large".to_string(),
                )
            })?
        } else {
            return Err(VecfitError::Serialization(
                "complex JSON must contain residues or C/B state-space matrices".to_string(),
            ));
        };

        if json.direct_terms.len() != channels {
            return Err(VecfitError::Serialization(
                "complex JSON direct term count must match channel count".to_string(),
            ));
        }
        if json.proportional_terms.len() != channels {
            return Err(VecfitError::Serialization(
                "complex JSON proportional term count must match channel count".to_string(),
            ));
        }

        let direct_terms = json
            .direct_terms
            .iter()
            .copied()
            .map(complex_from_pair)
            .collect::<Vec<_>>();
        let proportional_terms = json
            .proportional_terms
            .iter()
            .copied()
            .map(complex_from_pair)
            .collect::<Vec<_>>();

        let (shape, residues, output, modal_state_space) = if let Some(residue_rows) = json.residues
        {
            if residue_rows.iter().any(|row| row.len() != channels) {
                return Err(VecfitError::Serialization(
                    "complex JSON residue row length must match channel count".to_string(),
                ));
            }
            (
                resolve_json_shape(json.shape, channels)?,
                residue_rows
                    .iter()
                    .flat_map(|row| row.iter().copied().map(complex_from_pair))
                    .collect::<Vec<_>>(),
                OutputRepresentation::Residues,
                None,
            )
        } else if let (Some(c_rows), Some(b_rows)) = (json.c, json.b) {
            let rows = c_rows.len();
            let cols = b_rows.first().map_or(0, Vec::len);
            let shape = resolve_state_space_json_shape(json.shape, rows, cols)?;
            let state_space = ModalStateSpace::new(ModalStateSpaceParts {
                poles: poles.clone(),
                c: parse_complex_matrix(c_rows, rows, pole_count, "c")?,
                b: parse_complex_matrix(b_rows, pole_count, cols, "b")?,
                d: matrix_terms_from_flat(&direct_terms, rows, cols, layout),
                e: matrix_terms_from_flat(&proportional_terms, rows, cols, layout),
                rows,
                cols,
                shape: shape.clone(),
                layout,
            })?;
            let residues = state_space.reconstruct_residues()?;
            (
                shape,
                residues,
                OutputRepresentation::StateSpace,
                Some(state_space),
            )
        } else {
            unreachable!("residue/c/b presence was validated above");
        };

        let model = Model {
            poles,
            residues,
            channels,
            constant_terms: direct_terms,
            proportional_terms,
            shape,
            layout,
            report: Report {
                abs_rmse: json.abs_rmse,
                iterations: json.iterations,
                ..Report::default()
            },
            output,
            modal_state_space,
        };
        model.validate()?;
        Ok(model)
    }
}

impl TryFrom<&Model> for ComplexModelJson {
    type Error = VecfitError;

    fn try_from(model: &Model) -> Result<Self> {
        model.validate()?;
        let (residues, c, b) = match model.output {
            OutputRepresentation::Residues => (
                Some(
                    (0..model.poles.len())
                        .map(|pole_idx| {
                            (0..model.channels)
                                .map(|channel_idx| {
                                    let value =
                                        model.residues[pole_idx * model.channels + channel_idx];
                                    [value.re, value.im]
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                ),
                None,
                None,
            ),
            OutputRepresentation::StateSpace => {
                let state_space = model.modal_state_space()?;
                (
                    None,
                    Some(complex_matrix_to_pairs(
                        &state_space.c,
                        state_space.rows,
                        state_space.states(),
                    )),
                    Some(complex_matrix_to_pairs(
                        &state_space.b,
                        state_space.states(),
                        state_space.cols,
                    )),
                )
            }
        };
        Ok(Self {
            poles: model
                .poles
                .iter()
                .map(|value| [value.re, value.im])
                .collect(),
            residues,
            c,
            b,
            shape: Some(model.shape.clone()),
            layout: Some(model.layout),
            direct_terms: model
                .constant_terms
                .iter()
                .map(|value| [value.re, value.im])
                .collect(),
            proportional_terms: model
                .proportional_terms
                .iter()
                .map(|value| [value.re, value.im])
                .collect(),
            abs_rmse: model.report.abs_rmse,
            iterations: model.report.iterations,
        })
    }
}

impl TryFrom<RealKernelJsonModel> for Model {
    type Error = VecfitError;

    fn try_from(json: RealKernelJsonModel) -> Result<Self> {
        if json.poles.len() != json.n_poles {
            return Err(VecfitError::Serialization(
                "real-kernel JSON pole list length does not match n_poles".to_string(),
            ));
        }
        let channels = json.n_functions;
        if json.direct_terms.len() != channels {
            return Err(VecfitError::Serialization(
                "real-kernel JSON direct term count must match n_functions".to_string(),
            ));
        }
        if let Some(proportional_terms) = &json.proportional_terms {
            if proportional_terms.len() != channels {
                return Err(VecfitError::Serialization(
                    "real-kernel JSON proportional term count must match n_functions".to_string(),
                ));
            }
        }

        let mut residues = Vec::with_capacity(json.n_poles * channels);
        for pole in &json.poles {
            if pole.residues.len() != channels {
                return Err(VecfitError::Serialization(
                    "real-kernel JSON residue row length must match n_functions".to_string(),
                ));
            }
            residues.extend(
                pole.residues
                    .iter()
                    .copied()
                    .map(|value| Complex64::new(value, 0.0)),
            );
        }

        let model = Model {
            poles: json
                .poles
                .iter()
                .map(|pole| Complex64::new(pole.pole, 0.0))
                .collect(),
            residues,
            channels,
            constant_terms: json
                .direct_terms
                .iter()
                .copied()
                .map(|value| Complex64::new(value, 0.0))
                .collect(),
            proportional_terms: json
                .proportional_terms
                .unwrap_or_else(|| vec![0.0; channels])
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect(),
            shape: resolve_json_shape(json.shape, channels)?,
            layout: json.layout.unwrap_or(Layout::RowMajor),
            report: Report::default(),
            output: OutputRepresentation::Residues,
            modal_state_space: None,
        };
        model.validate()?;
        Ok(model)
    }
}

impl TryFrom<(&Model, Option<String>)> for RealKernelJsonModel {
    type Error = VecfitError;

    fn try_from((model, description): (&Model, Option<String>)) -> Result<Self> {
        model.validate()?;
        if model
            .poles
            .iter()
            .any(|pole| pole.im.abs() > REAL_COEFFICIENT_TOLERANCE)
        {
            return Err(VecfitError::InvalidInput(
                "real-kernel export requires real poles".to_string(),
            ));
        }
        if model
            .residues
            .iter()
            .chain(model.constant_terms.iter())
            .chain(model.proportional_terms.iter())
            .any(|value| value.im.abs() > REAL_COEFFICIENT_TOLERANCE)
        {
            return Err(VecfitError::InvalidInput(
                "real-kernel export requires real coefficients".to_string(),
            ));
        }

        let poles = model
            .poles
            .iter()
            .enumerate()
            .map(|(pole_idx, pole)| RealKernelPoleJson {
                pole: pole.re,
                residues: (0..model.channels)
                    .map(|channel_idx| model.residues[pole_idx * model.channels + channel_idx].re)
                    .collect(),
            })
            .collect::<Vec<_>>();
        Ok(Self {
            description,
            n_functions: model.channels,
            n_poles: model.poles.len(),
            shape: Some(model.shape.clone()),
            layout: Some(model.layout),
            direct_terms: model.constant_terms.iter().map(|value| value.re).collect(),
            proportional_terms: Some(
                model
                    .proportional_terms
                    .iter()
                    .map(|value| value.re)
                    .collect(),
            ),
            poles,
        })
    }
}

/// Detect CSV format from headers. Returns the format and channel count.
fn detect_csv_format(headers: &StringRecord) -> Result<(CsvFormat, usize)> {
    if headers.len() < 3 {
        return Err(VecfitError::Csv(
            "expected at least a frequency column and one column pair".to_string(),
        ));
    }
    if (headers.len() - 1) % 2 != 0 {
        return Err(VecfitError::Csv(
            "data columns must appear in complete pairs".to_string(),
        ));
    }
    let channels = (headers.len() - 1) / 2;

    // Check the first data column to determine format
    let first_col = headers.get(1).unwrap_or("").trim().to_lowercase();
    let format = if first_col.starts_with("re") {
        CsvFormat::RealImag
    } else {
        // Default to magnitude/phase (covers |Y|, mag_, etc.)
        CsvFormat::MagnitudePhase
    };

    Ok((format, channels))
}

fn parse_frequency_hz_column(record: &StringRecord) -> Result<f64> {
    record
        .get(0)
        .ok_or_else(|| VecfitError::Csv("missing frequency column".to_string()))?
        .trim()
        .parse::<f64>()
        .map_err(Into::into)
}

fn parse_mag_phase_column(record: &StringRecord, channel_idx: usize) -> Result<Complex64> {
    let magnitude = record
        .get(1 + channel_idx * 2)
        .ok_or_else(|| VecfitError::Csv("missing magnitude column".to_string()))?
        .trim()
        .parse::<f64>()?;
    let phase_deg = record
        .get(2 + channel_idx * 2)
        .ok_or_else(|| VecfitError::Csv("missing phase column".to_string()))?
        .trim()
        .parse::<f64>()?;
    let phase_rad = phase_deg.to_radians();
    Ok(Complex64::new(
        magnitude * phase_rad.cos(),
        magnitude * phase_rad.sin(),
    ))
}

fn parse_real_imag_column(record: &StringRecord, channel_idx: usize) -> Result<Complex64> {
    let re = record
        .get(1 + channel_idx * 2)
        .ok_or_else(|| VecfitError::Csv("missing real column".to_string()))?
        .trim()
        .parse::<f64>()?;
    let im = record
        .get(2 + channel_idx * 2)
        .ok_or_else(|| VecfitError::Csv("missing imaginary column".to_string()))?
        .trim()
        .parse::<f64>()?;
    Ok(Complex64::new(re, im))
}

fn complex_from_pair(value: [f64; 2]) -> Complex64 {
    Complex64::new(value[0], value[1])
}

fn complex_matrix_to_pairs(values: &[Complex64], rows: usize, cols: usize) -> Vec<Vec<[f64; 2]>> {
    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| {
                    let value = values[row * cols + col];
                    [value.re, value.im]
                })
                .collect()
        })
        .collect()
}

fn parse_complex_matrix(
    rows: Vec<Vec<[f64; 2]>>,
    expected_rows: usize,
    expected_cols: usize,
    name: &str,
) -> Result<Vec<Complex64>> {
    if rows.len() != expected_rows {
        return Err(VecfitError::Serialization(format!(
            "complex JSON {name} matrix row count must be {expected_rows}"
        )));
    }
    if rows.iter().any(|row| row.len() != expected_cols) {
        return Err(VecfitError::Serialization(format!(
            "complex JSON {name} matrix column count must be {expected_cols}"
        )));
    }
    Ok(rows
        .into_iter()
        .flat_map(|row| row.into_iter().map(complex_from_pair))
        .collect())
}

fn resolve_json_shape(shape: Option<Shape>, channels: usize) -> Result<Shape> {
    let shape = match shape {
        Some(shape) => shape,
        None if channels == 1 => Shape::scalar(),
        None => Shape::vector(channels)?,
    };
    if shape.channels() != channels {
        return Err(VecfitError::Serialization(format!(
            "JSON shape {:?} expects {} channels but model has {}",
            shape.dims(),
            shape.channels(),
            channels
        )));
    }
    Ok(shape)
}

fn resolve_state_space_json_shape(shape: Option<Shape>, rows: usize, cols: usize) -> Result<Shape> {
    if rows == 0 || cols == 0 {
        return Err(VecfitError::Serialization(
            "complex JSON state-space matrices must have positive dimensions".to_string(),
        ));
    }
    let shape = match shape {
        Some(shape) => shape,
        None if rows == 1 && cols == 1 => Shape::scalar(),
        None if cols == 1 => Shape::vector(rows)?,
        None => Shape::matrix(rows, cols)?,
    };
    let (shape_rows, shape_cols) = residue_matrix_dims(&shape)?;
    if shape_rows != rows || shape_cols != cols {
        return Err(VecfitError::Serialization(format!(
            "complex JSON state-space dimensions {}x{} do not match shape {:?}",
            rows,
            cols,
            shape.dims()
        )));
    }
    Ok(shape)
}
