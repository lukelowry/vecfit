use std::num::ParseFloatError;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VecfitError {
    #[error("io error: {0}")]
    Io(String),
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    #[error("invalid shape: {0}")]
    Shape(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("numerical failure: {0}")]
    Numerical(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("csv error: {0}")]
    Csv(String),
}

pub type Result<T> = std::result::Result<T, VecfitError>;

impl From<serde_json::Error> for VecfitError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

impl From<csv::Error> for VecfitError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value.to_string())
    }
}

impl From<ParseFloatError> for VecfitError {
    fn from(value: ParseFloatError) -> Self {
        Self::InvalidInput(value.to_string())
    }
}

impl From<std::io::Error> for VecfitError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<faer::linalg::solvers::SvdError> for VecfitError {
    fn from(value: faer::linalg::solvers::SvdError) -> Self {
        Self::Numerical(format!("{value:?}"))
    }
}

impl From<faer::linalg::solvers::EvdError> for VecfitError {
    fn from(value: faer::linalg::solvers::EvdError) -> Self {
        Self::Numerical(format!("{value:?}"))
    }
}
