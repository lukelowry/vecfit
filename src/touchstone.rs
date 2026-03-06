//! Touchstone v1 (`.sNp`) file parser.
//!
//! Parses Touchstone files into [`ParsedSamples`] for fitting with the vecfit
//! solver. Supports `.s1p` through arbitrary `.sNp` files, all parameter types
//! (S, Y, Z, G, H, T), and all data formats (RI, MA, DB).

use std::ops::Deref;
use std::path::Path;
use std::str::FromStr;

use num_complex::Complex64;

use crate::error::{Result, VecfitError};
use crate::fit::SampleMatrix;
use crate::io::ParsedSamples;
use crate::shape::{Layout, Shape};

// ---------------------------------------------------------------------------
// Supporting enums
// ---------------------------------------------------------------------------

/// Frequency unit used in a Touchstone option line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyUnit {
    Hz,
    KHz,
    MHz,
    GHz,
}

impl FrequencyUnit {
    /// Multiplier to convert the file's frequency values to hertz.
    pub fn to_hz_factor(self) -> f64 {
        match self {
            Self::Hz => 1.0,
            Self::KHz => 1e3,
            Self::MHz => 1e6,
            Self::GHz => 1e9,
        }
    }
}

impl FromStr for FrequencyUnit {
    type Err = VecfitError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_uppercase().as_str() {
            "HZ" => Ok(Self::Hz),
            "KHZ" => Ok(Self::KHz),
            "MHZ" => Ok(Self::MHz),
            "GHZ" => Ok(Self::GHz),
            _ => Err(VecfitError::Touchstone(format!(
                "unknown frequency unit: {s}"
            ))),
        }
    }
}

/// Network parameter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterType {
    S,
    Y,
    Z,
    G,
    H,
    T,
}

impl FromStr for ParameterType {
    type Err = VecfitError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_uppercase().as_str() {
            "S" => Ok(Self::S),
            "Y" => Ok(Self::Y),
            "Z" => Ok(Self::Z),
            "G" => Ok(Self::G),
            "H" => Ok(Self::H),
            "T" => Ok(Self::T),
            _ => Err(VecfitError::Touchstone(format!(
                "unknown parameter type: {s}"
            ))),
        }
    }
}

/// Data format for complex value pairs in a Touchstone file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// Real / Imaginary
    RI,
    /// Magnitude / Angle (degrees)
    MA,
    /// Decibels / Angle (degrees)
    DB,
}

impl FromStr for DataFormat {
    type Err = VecfitError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_uppercase().as_str() {
            "RI" => Ok(Self::RI),
            "MA" => Ok(Self::MA),
            "DB" => Ok(Self::DB),
            _ => Err(VecfitError::Touchstone(format!(
                "unknown data format: {s}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// TouchstoneOptions
// ---------------------------------------------------------------------------

/// Parsed option line from a Touchstone file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TouchstoneOptions {
    pub frequency_unit: FrequencyUnit,
    pub parameter_type: ParameterType,
    pub data_format: DataFormat,
    pub reference_impedance: f64,
}

impl Default for TouchstoneOptions {
    fn default() -> Self {
        Self {
            frequency_unit: FrequencyUnit::GHz,
            parameter_type: ParameterType::S,
            data_format: DataFormat::MA,
            reference_impedance: 50.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Touchstone
// ---------------------------------------------------------------------------

/// A parsed Touchstone (`.sNp`) file.
///
/// Dereferences to [`ParsedSamples`] so that all sample-inspection and fitting
/// methods are available directly.
#[derive(Debug, Clone)]
pub struct Touchstone {
    inner: ParsedSamples,
    options: TouchstoneOptions,
    ports: usize,
}

impl Deref for Touchstone {
    type Target = ParsedSamples;

    fn deref(&self) -> &ParsedSamples {
        &self.inner
    }
}

impl Touchstone {
    /// Convert into the format-agnostic parsed samples.
    pub fn into_parsed(self) -> ParsedSamples {
        self.inner
    }

    /// Return the parsed Touchstone option line.
    pub fn touchstone_options(&self) -> &TouchstoneOptions {
        &self.options
    }

    /// Return the number of ports.
    pub fn ports(&self) -> usize {
        self.ports
    }

    /// Return the network parameter type (S, Y, Z, etc.).
    pub fn parameter_type(&self) -> ParameterType {
        self.options.parameter_type
    }

    /// Return the reference impedance in ohms.
    pub fn reference_impedance(&self) -> f64 {
        self.options.reference_impedance
    }

    /// Parse a Touchstone file from a filesystem path.
    ///
    /// The port count is inferred from the file extension (`.s1p` → 1 port,
    /// `.s2p` → 2 ports, etc.).
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let ports = infer_ports_from_extension(path)?;
        let text = std::fs::read_to_string(path).map_err(|e| {
            VecfitError::Touchstone(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_str_with_ports(&text, ports)
    }

    /// Parse Touchstone text with an explicit port count.
    pub fn from_str_with_ports(text: &str, ports: usize) -> Result<Self> {
        if ports == 0 {
            return Err(VecfitError::Touchstone(
                "port count must be at least 1".into(),
            ));
        }
        let (options, data_lines) = preprocess(text)?;
        let n_params = ports * ports;
        let points = parse_data_lines(&data_lines, n_params, &options)?;
        build_touchstone(points, options, ports)
    }
}

impl FromStr for Touchstone {
    type Err = VecfitError;

    /// Parse Touchstone text, inferring the port count from the number of data
    /// columns on the first frequency line.
    fn from_str(text: &str) -> Result<Self> {
        let (options, data_lines) = preprocess(text)?;
        let ports = infer_ports_from_data(&data_lines)?;
        let n_params = ports * ports;
        let points = parse_data_lines(&data_lines, n_params, &options)?;
        build_touchstone(points, options, ports)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A single frequency point with its N² complex parameters.
struct FrequencyPoint {
    freq_hz: f64,
    params: Vec<Complex64>,
}

/// Strip comments, find the option line, and collect data lines.
fn preprocess(text: &str) -> Result<(TouchstoneOptions, Vec<String>) > {
    let mut options = TouchstoneOptions::default();
    let mut data_lines = Vec::new();
    let mut found_option_line = false;

    for raw_line in text.lines() {
        // Strip inline comments (everything after '!')
        let line = match raw_line.find('!') {
            Some(idx) => &raw_line[..idx],
            None => raw_line,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with('#') {
            if found_option_line {
                return Err(VecfitError::Touchstone(
                    "multiple option lines found".into(),
                ));
            }
            options = parse_option_line(trimmed)?;
            found_option_line = true;
            continue;
        }

        data_lines.push(trimmed.to_string());
    }

    Ok((options, data_lines))
}

/// Parse the `# <freq_unit> <param_type> <data_format> R <impedance>` option line.
fn parse_option_line(line: &str) -> Result<TouchstoneOptions> {
    // Remove the leading '#' and split on whitespace.
    let tokens: Vec<&str> = line[1..].split_whitespace().collect();
    if tokens.is_empty() {
        return Err(VecfitError::Touchstone(
            "option line '#' contains no parameters; expected at least a frequency unit".into(),
        ));
    }

    let mut freq_unit = None;
    let mut param_type = None;
    let mut data_format = None;
    let mut ref_impedance = None;

    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i].to_ascii_uppercase();
        if tok == "R" {
            // Next token is the reference impedance
            i += 1;
            if i >= tokens.len() {
                return Err(VecfitError::Touchstone(
                    "missing reference impedance value after R".into(),
                ));
            }
            ref_impedance = Some(tokens[i].parse::<f64>().map_err(|e| {
                VecfitError::Touchstone(format!("invalid reference impedance: {e}"))
            })?);
        } else if let Ok(fu) = tok.parse::<FrequencyUnit>() {
            freq_unit = Some(fu);
        } else if let Ok(pt) = tok.parse::<ParameterType>() {
            param_type = Some(pt);
        } else if let Ok(df) = tok.parse::<DataFormat>() {
            data_format = Some(df);
        } else {
            return Err(VecfitError::Touchstone(format!(
                "unrecognized token in option line: {tok}"
            )));
        }
        i += 1;
    }

    let defaults = TouchstoneOptions::default();
    Ok(TouchstoneOptions {
        frequency_unit: freq_unit.unwrap_or(defaults.frequency_unit),
        parameter_type: param_type.unwrap_or(defaults.parameter_type),
        data_format: data_format.unwrap_or(defaults.data_format),
        reference_impedance: ref_impedance.unwrap_or(defaults.reference_impedance),
    })
}

/// Convert a value pair to `Complex64` according to the data format.
fn pair_to_complex(v1: f64, v2: f64, format: DataFormat) -> Complex64 {
    match format {
        DataFormat::RI => Complex64::new(v1, v2),
        DataFormat::MA => {
            let angle = v2.to_radians();
            Complex64::new(v1 * angle.cos(), v1 * angle.sin())
        }
        DataFormat::DB => {
            let mag = 10f64.powf(v1 / 20.0);
            let angle = v2.to_radians();
            Complex64::new(mag * angle.cos(), mag * angle.sin())
        }
    }
}

/// Parse all data lines into frequency points, handling multi-line continuation
/// for N >= 3 port files.
fn parse_data_lines(
    data_lines: &[String],
    n_params: usize,
    options: &TouchstoneOptions,
) -> Result<Vec<FrequencyPoint>> {
    if data_lines.is_empty() {
        return Err(VecfitError::Touchstone("no data lines found".into()));
    }

    let hz_factor = options.frequency_unit.to_hz_factor();
    // Number of value pairs that fit on one line (Touchstone v1 allows at most
    // 4 pairs per line).
    let mut points = Vec::new();
    let mut line_idx = 0;

    while line_idx < data_lines.len() {
        let tokens: Vec<f64> = parse_tokens(&data_lines[line_idx])?;
        if tokens.is_empty() {
            line_idx += 1;
            continue;
        }

        // First token is frequency.
        let freq_hz = tokens[0] * hz_factor;
        let mut values: Vec<f64> = tokens[1..].to_vec();
        line_idx += 1;

        // For n_params > max_pairs_per_line, we need continuation lines.
        let needed_values = n_params * 2;
        while values.len() < needed_values && line_idx < data_lines.len() {
            // Continuation lines have no frequency token.
            let cont_tokens: Vec<f64> = parse_tokens(&data_lines[line_idx])?;
            values.extend_from_slice(&cont_tokens);
            line_idx += 1;
        }

        if values.len() != needed_values {
            return Err(VecfitError::Touchstone(format!(
                "expected {} data values for frequency {freq_hz} Hz, found {}",
                needed_values,
                values.len()
            )));
        }

        let params: Vec<Complex64> = values
            .chunks_exact(2)
            .map(|pair| pair_to_complex(pair[0], pair[1], options.data_format))
            .collect();

        points.push(FrequencyPoint { freq_hz, params });
    }

    Ok(points)
}

/// Parse a single line into a vector of `f64` tokens.
fn parse_tokens(line: &str) -> Result<Vec<f64>> {
    line.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>().map_err(|e| {
                VecfitError::Touchstone(format!("failed to parse numeric token '{tok}': {e}"))
            })
        })
        .collect()
}

/// Build the final `Touchstone` from parsed frequency points.
fn build_touchstone(
    points: Vec<FrequencyPoint>,
    options: TouchstoneOptions,
    ports: usize,
) -> Result<Touchstone> {
    let n_samples = points.len();
    let n_params = ports * ports;

    let mut frequency_hz = Vec::with_capacity(n_samples);
    let mut axis = Vec::with_capacity(n_samples);
    let mut values = Vec::with_capacity(n_samples * n_params);

    for pt in points {
        frequency_hz.push(pt.freq_hz);
        axis.push(Complex64::new(
            0.0,
            2.0 * std::f64::consts::PI * pt.freq_hz,
        ));
        values.extend_from_slice(&pt.params);
    }

    let shape = if ports == 1 {
        Shape::scalar()
    } else {
        Shape::matrix(ports, ports)?
    };

    let samples = SampleMatrix::new(values, n_samples, n_params)?;
    let inner = ParsedSamples::new(frequency_hz, axis, samples, shape, Layout::RowMajor)?;

    Ok(Touchstone {
        inner,
        options,
        ports,
    })
}

/// Infer port count from a file extension like `.s2p`.
fn infer_ports_from_extension(path: &Path) -> Result<usize> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    // Expected format: s<N>p
    if ext.starts_with('s') && ext.ends_with('p') && ext.len() >= 3 {
        let middle = &ext[1..ext.len() - 1];
        if let Ok(n) = middle.parse::<usize>() {
            if n >= 1 {
                return Ok(n);
            }
        }
    }

    Err(VecfitError::Touchstone(format!(
        "cannot infer port count from extension '.{ext}'; expected .sNp"
    )))
}

/// Infer port count from the number of data values on the first data line(s).
fn infer_ports_from_data(data_lines: &[String]) -> Result<usize> {
    if data_lines.is_empty() {
        return Err(VecfitError::Touchstone("no data lines found".into()));
    }

    let first_tokens = parse_tokens(&data_lines[0])?;
    if first_tokens.len() < 3 {
        return Err(VecfitError::Touchstone(
            "first data line has too few tokens to infer port count".into(),
        ));
    }

    // First token is frequency; remaining tokens are value pairs.
    let n_values = first_tokens.len() - 1;
    if n_values % 2 != 0 {
        return Err(VecfitError::Touchstone(format!(
            "odd number of data values ({n_values}) on first data line"
        )));
    }
    let n_pairs = n_values / 2;

    // Check if n_pairs is a perfect square (1-port: 1, 2-port: 4).
    let sqrt = isqrt(n_pairs);
    if sqrt * sqrt == n_pairs {
        return Ok(sqrt);
    }

    // For N >= 3 ports, the first line has at most 4 pairs, then continuation
    // lines follow. Accumulate tokens from subsequent lines to figure it out.
    let max_first_line_pairs = 4;
    if n_pairs <= max_first_line_pairs {
        let mut total_pairs = n_pairs;
        let mut li = 1;
        while li < data_lines.len() {
            let tokens = parse_tokens(&data_lines[li])?;
            if tokens.is_empty() {
                li += 1;
                continue;
            }
            // If the line would start a new frequency point (we can't easily
            // tell), we check whether the accumulated pairs form a perfect
            // square after adding these tokens.
            if tokens.len() % 2 != 0 {
                // Odd token count means this is a new frequency line (has freq token).
                break;
            }
            total_pairs += tokens.len() / 2;
            li += 1;

            let s = isqrt(total_pairs);
            if s * s == total_pairs && s >= 2 {
                return Ok(s);
            }
        }
    }

    Err(VecfitError::Touchstone(format!(
        "cannot infer port count: {n_pairs} parameter pairs on first line is not a perfect square"
    )))
}

/// Integer square root (largest n where n*n <= val).
fn isqrt(val: usize) -> usize {
    if val == 0 {
        return 0;
    }
    let mut s = (val as f64).sqrt() as usize;
    while s * s > val {
        s -= 1;
    }
    while (s + 1) * (s + 1) <= val {
        s += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_option_line_defaults() {
        let opts = parse_option_line("# GHz S MA R 50").unwrap();
        assert_eq!(opts.frequency_unit, FrequencyUnit::GHz);
        assert_eq!(opts.parameter_type, ParameterType::S);
        assert_eq!(opts.data_format, DataFormat::MA);
        assert_eq!(opts.reference_impedance, 50.0);
    }

    #[test]
    fn parse_option_line_case_insensitive() {
        let opts = parse_option_line("# mhz y ri r 75").unwrap();
        assert_eq!(opts.frequency_unit, FrequencyUnit::MHz);
        assert_eq!(opts.parameter_type, ParameterType::Y);
        assert_eq!(opts.data_format, DataFormat::RI);
        assert_eq!(opts.reference_impedance, 75.0);
    }

    #[test]
    fn data_format_ri() {
        let c = pair_to_complex(1.0, 2.0, DataFormat::RI);
        assert!((c.re - 1.0).abs() < 1e-15);
        assert!((c.im - 2.0).abs() < 1e-15);
    }

    #[test]
    fn data_format_ma() {
        let c = pair_to_complex(2.0, 90.0, DataFormat::MA);
        assert!(c.re.abs() < 1e-10);
        assert!((c.im - 2.0).abs() < 1e-10);
    }

    #[test]
    fn data_format_db() {
        // 0 dB, 0 degrees → magnitude 1, angle 0
        let c = pair_to_complex(0.0, 0.0, DataFormat::DB);
        assert!((c.re - 1.0).abs() < 1e-10);
        assert!(c.im.abs() < 1e-10);
    }

    #[test]
    fn parse_s1p() {
        let text = "\
! 1-port S-parameter file
# MHz S RI R 50
100 0.5 0.1
200 0.3 -0.2
500 0.1 -0.5
";
        let ts: Touchstone = Touchstone::from_str_with_ports(text, 1).unwrap();
        assert_eq!(ts.ports(), 1);
        assert_eq!(ts.len(), 3);
        assert!(ts.shape().is_scalar());
        assert!((ts.frequency_hz()[0] - 100e6).abs() < 1e-3);
    }

    #[test]
    fn parse_s2p() {
        let text = "\
! 2-port S-parameter file
# GHz S MA R 50
1.0  0.9 -10  0.1 80  0.1 80  0.9 -10
2.0  0.8 -20  0.2 70  0.2 70  0.8 -20
";
        let ts = Touchstone::from_str_with_ports(text, 2).unwrap();
        assert_eq!(ts.ports(), 2);
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.channels(), 4);
        assert!((ts.frequency_hz()[0] - 1e9).abs() < 1e-3);
    }

    #[test]
    fn parse_s4p_multiline() {
        // 4-port: 16 params = first line: freq + 4 pairs, then 3 continuation
        // lines of 4 pairs each.
        let text = "\
# GHz S RI R 50
1.0  0.1 0.0  0.2 0.0  0.3 0.0  0.4 0.0
     0.5 0.0  0.6 0.0  0.7 0.0  0.8 0.0
     0.9 0.0  1.0 0.0  1.1 0.0  1.2 0.0
     1.3 0.0  1.4 0.0  1.5 0.0  1.6 0.0
";
        let ts = Touchstone::from_str_with_ports(text, 4).unwrap();
        assert_eq!(ts.ports(), 4);
        assert_eq!(ts.len(), 1);
        assert_eq!(ts.channels(), 16);
    }

    #[test]
    fn from_str_infer_1port() {
        let text = "\
# MHz S RI R 50
100 0.5 0.1
200 0.3 -0.2
";
        let ts: Touchstone = text.parse().unwrap();
        assert_eq!(ts.ports(), 1);
    }

    #[test]
    fn from_str_infer_2port() {
        let text = "\
# GHz S RI R 50
1.0  0.1 0.0  0.2 0.0  0.3 0.0  0.4 0.0
2.0  0.5 0.0  0.6 0.0  0.7 0.0  0.8 0.0
";
        let ts: Touchstone = text.parse().unwrap();
        assert_eq!(ts.ports(), 2);
    }

    #[test]
    fn frequency_unit_factors() {
        assert!((FrequencyUnit::Hz.to_hz_factor() - 1.0).abs() < f64::EPSILON);
        assert!((FrequencyUnit::KHz.to_hz_factor() - 1e3).abs() < f64::EPSILON);
        assert!((FrequencyUnit::MHz.to_hz_factor() - 1e6).abs() < f64::EPSILON);
        assert!((FrequencyUnit::GHz.to_hz_factor() - 1e9).abs() < f64::EPSILON);
    }

    #[test]
    fn default_options_no_option_line() {
        let text = "\
1.0  0.5 0.1
2.0  0.3 -0.2
";
        let ts = Touchstone::from_str_with_ports(text, 1).unwrap();
        let opts = ts.touchstone_options();
        assert_eq!(opts.frequency_unit, FrequencyUnit::GHz);
        assert_eq!(opts.parameter_type, ParameterType::S);
        assert_eq!(opts.data_format, DataFormat::MA);
        assert_eq!(opts.reference_impedance, 50.0);
    }

    #[test]
    fn inline_comments_stripped() {
        let text = "\
! header comment
# MHz S RI R 50
100 0.5 0.1 ! inline comment
200 0.3 -0.2
";
        let ts = Touchstone::from_str_with_ports(text, 1).unwrap();
        assert_eq!(ts.len(), 2);
    }
}
