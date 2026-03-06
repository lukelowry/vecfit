use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::error::{Result, VecfitError};

/// Memory layout used when flattening multi-dimensional responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Layout {
    #[default]
    RowMajor,
    ColumnMajor,
}

/// Recorded output shape for each fitted sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn scalar() -> Self {
        Self(Vec::new())
    }

    pub fn vector(len: usize) -> Result<Self> {
        Self::tensor([len])
    }

    pub fn matrix(rows: usize, cols: usize) -> Result<Self> {
        Self::tensor([rows, cols])
    }

    pub fn tensor<I>(dims: I) -> Result<Self>
    where
        I: IntoIterator<Item = usize>,
    {
        let dims = dims.into_iter().collect::<Vec<_>>();
        if dims.contains(&0) {
            return Err(VecfitError::Shape(
                "shape dimensions must be positive".to_string(),
            ));
        }
        Ok(Self(dims))
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    pub fn channels(&self) -> usize {
        if self.is_scalar() {
            1
        } else {
            self.0.iter().product()
        }
    }

    pub fn expect_vector(&self) -> Result<usize> {
        match self.0.as_slice() {
            [len] => Ok(*len),
            _ => Err(VecfitError::Shape(format!(
                "expected vector shape, found {:?}",
                self.0
            ))),
        }
    }

    pub fn expect_matrix(&self) -> Result<(usize, usize)> {
        match self.0.as_slice() {
            [rows, cols] => Ok((*rows, *cols)),
            _ => Err(VecfitError::Shape(format!(
                "expected matrix shape, found {:?}",
                self.0
            ))),
        }
    }

    /// Infer shape from channel count: scalar for 1, square matrix if
    /// `channels` is a perfect square, otherwise vector.
    pub fn infer_square(channels: usize) -> Result<Self> {
        if channels == 1 {
            return Ok(Self::scalar());
        }
        let sqrt = (channels as f64).sqrt() as usize;
        if sqrt * sqrt == channels {
            Self::matrix(sqrt, sqrt)
        } else {
            Self::vector(channels)
        }
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self::scalar()
    }
}

/// Flattened response payload paired with shape metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatResponse<T> {
    pub values: Vec<T>,
    pub shape: Shape,
    pub layout: Layout,
}

impl<T> FlatResponse<T> {
    pub fn new(values: Vec<T>, shape: Shape, layout: Layout) -> Result<Self> {
        let expected = shape.channels();
        if values.len() != expected {
            return Err(VecfitError::Shape(format!(
                "flattened data length {} does not match shape {:?} ({} channels)",
                values.len(),
                shape.dims(),
                expected
            )));
        }
        Ok(Self {
            values,
            shape,
            layout,
        })
    }

    pub fn scalar(value: T) -> Self {
        Self {
            values: vec![value],
            shape: Shape::scalar(),
            layout: Layout::RowMajor,
        }
    }
}

/// Owned response payload that can be reconstructed into scalar, vector, or matrix form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseSample<T> {
    pub values: Vec<T>,
    pub shape: Shape,
    pub layout: Layout,
}

impl<T> ResponseSample<T> {
    pub fn new(values: Vec<T>, shape: Shape, layout: Layout) -> Result<Self> {
        FlatResponse::new(values, shape, layout).map(Into::into)
    }

    pub fn into_scalar(self) -> Result<T> {
        if !self.shape.is_scalar() {
            return Err(VecfitError::Shape(format!(
                "expected scalar shape, found {:?}",
                self.shape.dims()
            )));
        }
        self.values
            .into_iter()
            .next()
            .ok_or_else(|| VecfitError::Shape("scalar value is missing".to_string()))
    }

    pub fn into_vector(self) -> Result<Vec<T>> {
        self.shape.expect_vector()?;
        Ok(self.values)
    }

    pub fn into_matrix(self) -> Result<Vec<Vec<T>>>
    where
        T: Clone,
    {
        let (rows, cols) = self.shape.expect_matrix()?;
        let mut out = vec![Vec::with_capacity(cols); rows];
        match self.layout {
            Layout::RowMajor => {
                for (row_idx, row) in out.iter_mut().enumerate().take(rows) {
                    for col_idx in 0..cols {
                        row.push(self.values[row_idx * cols + col_idx].clone());
                    }
                }
            }
            Layout::ColumnMajor => {
                for col_idx in 0..cols {
                    for (row_idx, row) in out.iter_mut().enumerate().take(rows) {
                        row.push(self.values[col_idx * rows + row_idx].clone());
                    }
                }
            }
        }
        Ok(out)
    }
}

impl<T> From<FlatResponse<T>> for ResponseSample<T> {
    fn from(value: FlatResponse<T>) -> Self {
        Self {
            values: value.values,
            shape: value.shape,
            layout: value.layout,
        }
    }
}

/// Scalar types that the fitting API accepts directly.
pub trait ResponseScalar: Clone {
    fn to_complex(&self) -> Complex64;
}

impl ResponseScalar for f64 {
    fn to_complex(&self) -> Complex64 {
        Complex64::new(*self, 0.0)
    }
}

impl ResponseScalar for Complex64 {
    fn to_complex(&self) -> Complex64 {
        *self
    }
}

/// Trait implemented by scalar, vector, matrix, and wrapper response types accepted by fitting.
pub trait IntoResponse {
    fn into_response(self) -> Result<FlatResponse<Complex64>>;
}

impl<T> IntoResponse for FlatResponse<T>
where
    T: ResponseScalar,
{
    fn into_response(self) -> Result<FlatResponse<Complex64>> {
        FlatResponse::new(
            self.values
                .into_iter()
                .map(|value| value.to_complex())
                .collect(),
            self.shape,
            self.layout,
        )
    }
}

impl<T> IntoResponse for ResponseSample<T>
where
    T: ResponseScalar,
{
    fn into_response(self) -> Result<FlatResponse<Complex64>> {
        FlatResponse::new(
            self.values
                .into_iter()
                .map(|value| value.to_complex())
                .collect(),
            self.shape,
            self.layout,
        )
    }
}

impl IntoResponse for f64 {
    fn into_response(self) -> Result<FlatResponse<Complex64>> {
        Ok(FlatResponse::scalar(self.to_complex()))
    }
}

impl IntoResponse for Complex64 {
    fn into_response(self) -> Result<FlatResponse<Complex64>> {
        Ok(FlatResponse::scalar(self))
    }
}

impl<T, const N: usize> IntoResponse for [T; N]
where
    T: ResponseScalar,
{
    fn into_response(self) -> Result<FlatResponse<Complex64>> {
        FlatResponse::new(
            self.into_iter().map(|value| value.to_complex()).collect(),
            Shape::vector(N)?,
            Layout::RowMajor,
        )
    }
}

impl<T, const R: usize, const C: usize> IntoResponse for [[T; C]; R]
where
    T: ResponseScalar,
{
    fn into_response(self) -> Result<FlatResponse<Complex64>> {
        let mut values = Vec::with_capacity(R * C);
        for row in self {
            for value in row {
                values.push(value.to_complex());
            }
        }
        FlatResponse::new(values, Shape::matrix(R, C)?, Layout::RowMajor)
    }
}

impl<T> IntoResponse for Vec<T>
where
    T: ResponseScalar,
{
    fn into_response(self) -> Result<FlatResponse<Complex64>> {
        let len = self.len();
        FlatResponse::new(
            self.into_iter().map(|value| value.to_complex()).collect(),
            Shape::vector(len)?,
            Layout::RowMajor,
        )
    }
}

impl<T> IntoResponse for Vec<Vec<T>>
where
    T: ResponseScalar,
{
    fn into_response(self) -> Result<FlatResponse<Complex64>> {
        let rows = self.len();
        let cols = self.first().map_or(0, Vec::len);
        if rows == 0 || cols == 0 {
            return Err(VecfitError::Shape(
                "matrix response cannot be empty".to_string(),
            ));
        }
        if self.iter().any(|row| row.len() != cols) {
            return Err(VecfitError::Shape(
                "matrix response rows must have constant length".to_string(),
            ));
        }
        let mut values = Vec::with_capacity(rows * cols);
        for row in self {
            for value in row {
                values.push(value.to_complex());
            }
        }
        FlatResponse::new(values, Shape::matrix(rows, cols)?, Layout::RowMajor)
    }
}
