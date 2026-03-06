# vecfit

`vecfit` is a pure-Rust crate for relaxed vector fitting of scalar, vector, matrix, and tensor-valued frequency responses.

## Start Here

If you want the fastest end-to-end check in this repository, run:

```bash
cargo run --example fit_scalar
```

That example fits a scalar transfer function on a complex sample axis and prints the fitted pole count plus RMSE.

```rust
use num_complex::Complex64;
use vecfit::{FitOptions, Model};

fn main() -> Result<(), vecfit::VecfitError> {
    let sample_axis = (0..200)
        .map(|k| Complex64::new(0.0, 0.5 + k as f64))
        .collect::<Vec<_>>();

    let model = Model::fit(
        &sample_axis,
        |sample| {
            1.0 / (sample + Complex64::new(3.0, 0.0))
                + 0.4 / (sample + Complex64::new(8.0, 0.0))
                + 0.05
        },
        FitOptions::new().poles(4),
    )?;

    println!(
        "poles={}, abs_rmse={:.3e}",
        model.pole_count(),
        model.abs_rmse()
    );
    Ok(())
}
```

## Core Types

- `Model` owns a fitted rational approximation.
- `FitOptions` configures the solve.
- `Shape` and `Layout` describe flattened outputs when you need explicit control.
- `CsvSamples` parses EMT magnitude/phase CSV and can fit directly.

## What It Covers

- Scalar, vector, matrix, and tensor-valued fits.
- Complex axes plus `fit_hz(...)` and `fit_rad(...)` helpers.
- Shape-aware evaluation with `evaluate_scalar(...)`, `evaluate_vector(...)`, `evaluate_matrix(...)`, and `evaluate(...)`.
- EMT-oriented CSV ingestion with `CsvSamples::from_csv(...)`, `CsvSamples::from_path(...)`, or `csv_text.parse::<CsvSamples>()`.
- JSON import/export with `to_json(...)`, `from_json(...)`, `from_json_path(...)`, and the real-kernel variants.
- Real-section and state-space export for real-only fits.

## Examples

Quick starts:

- `cargo run --example fit_scalar`
- `cargo run --example fit_vector`
- `cargo run --example fit_csv`
- `cargo run --example json_roundtrip`
- `cargo run --example export_matrix_emt`

Report and plotting examples:

- `cargo run --example plot_scalar_report`
- `cargo run --example plot_vector_report`
- `cargo run --example plot_matrix_emt_report`
- `cargo run --example plot_scalar_csv`
- `cargo run --example plot_matrix_csv`
- `cargo run --example plot_complex_json`
- `cargo run --example plot_real_kernel_json`

Plotted examples write PNG and Markdown summaries to `examples/out/`.

See `examples/README.md` for the full example catalog.

## Benchmarks

Run the Criterion benchmark suite with:

```bash
cargo bench --bench fit
```

## Scope

This crate covers the core relaxed vector fitting workflow, shape-aware evaluation, EMT exports, JSON/CSV interchange, and runnable examples.

Out of scope for this release:

- passivity enforcement
- a full recursive-convolution engine

## License

Licensed under either:

- Apache License, Version 2.0
- MIT license

at your option.
