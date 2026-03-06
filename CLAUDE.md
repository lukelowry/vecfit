# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build                          # build the library
cargo test                           # run all tests (unit + integration in tests/basic.rs)
cargo test scalar_fit                # run a single test by name substring
cargo bench --bench fit              # run Criterion benchmarks
cargo run --example fit_scalar       # fastest end-to-end check
cargo run --example plot_scalar_report  # plotted examples write to examples/out/
```

Rust edition 2024, MSRV 1.85. Dual-licensed MIT/Apache-2.0.

## Architecture

This is a pure-Rust implementation of relaxed vector fitting for rational approximation of frequency-domain data. The crate supports scalar, vector, matrix, and arbitrary-tensor-valued responses.

### Module responsibilities

- **`src/fit.rs`** - Core solver: initial pole generation, least-squares pole relocation loop, residue identification. Key types: `FitOptions` (solver config), `ProblemRef` (borrowed problem description), `SampleMatrix`/`SampleMatrixRef` (flat row-major response data), `Report` (fit metrics). Uses `faer` for dense linear algebra (SVD, least-squares).
- **`src/model.rs`** - `Model` is the primary public type. Owns fitted poles, residues, constant/proportional terms, shape, and layout. Provides multiple `fit_*` entry points (`fit`, `fit_hz`, `fit_rad`, `fit_real`, `fit_mapped`, `fit_samples`) and shape-aware evaluation (`evaluate_scalar`, `evaluate_vector`, `evaluate_matrix`, `evaluate`).
- **`src/shape.rs`** - `Shape` (dimension descriptor) and `Layout` (row-major/column-major). The `IntoResponse` trait lets users return scalars, arrays, vecs, or nested vecs from response closures; the shape is inferred automatically.
- **`src/io.rs`** - `CsvSamples` for EMT-style magnitude/phase CSV ingestion (parses `|Y|`/`ang_Y` column pairs). `ComplexModelJson` and `RealKernelJsonModel` for JSON interchange. `CsvSamples` implements `FromStr` so CSV text can be `.parse()`'d directly.
- **`src/emt.rs`** - EMT (electromagnetic transients) export: `RealSectionModel` (first/second-order real sections), `StateSpaceModel` (continuous A/B/C/D), `DiscreteStateSpaceModel` (discretized via backward Euler or Tustin).
- **`src/error.rs`** - `VecfitError` enum with `thiserror` derives. `From` impls for `serde_json`, `csv`, `io`, `faer` SVD/EVD errors.

### Data flow

1. User provides a complex sample axis and a response closure (or CSV/JSON data).
2. `Model::fit` collects responses, infers shape via `IntoResponse`, flattens to a `SampleMatrix`.
3. `fit.rs` generates initial poles (log-spaced with imaginary offsets), then iterates: builds a pole-basis matrix, solves a weighted least-squares system, extracts new poles via eigendecomposition, flips unstable poles, and repeats until convergence.
4. Final residue identification solves one more least-squares problem.
5. The resulting `Model` can evaluate on new axes or export to real sections / state-space.

### Key dependencies

- **`faer`** - Dense linear algebra (SVD solve, eigendecomposition)
- **`num-complex`** - `Complex64` throughout
- **`plotters`** (dev) - Used by `examples/support/mod.rs` for shared plotting layout across all plot examples

### Conventions

- Response data is stored flat in row-major order: `values[sample_idx * channels + channel_idx]`.
- Conjugate pole pairs are always stored together (real pole, or complex + conjugate).
- No legacy aliases remain in the public API.
