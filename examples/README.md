# Example Catalog

The example set is split into minimal examples and richer plotted reports.

## Minimal Examples
- `fit_scalar`: shortest scalar fit and scalar evaluation path.
- `fit_vector`: shortest multi-channel fit on a `jw` axis in hertz.
- `fit_csv`: parse CSV text (magnitude/phase format) and fit it via `ParsedSamples::fit`.
- `json_roundtrip`: serialize a fitted model to complex JSON and load it back.
- `export_matrix_emt`: fit a small real-only matrix model and export real sections plus a discrete state-space model.

These are the examples to start with if you want copyable integration snippets for another Rust crate or application.

## Plotted Report Examples
- `plot_scalar_report`: generated scalar response with the shared comparison layout.
- `plot_vector_report`: generated multi-channel response with all channels overlaid on shared magnitude, phase, and relative-error panels.
- `plot_matrix_emt_report`: generated 2x2 matrix response with resonances and cross-coupling, all matrix elements overlaid on shared panels.
- `plot_scalar_csv`: read scalar magnitude/phase data from CSV and render the shared comparison layout.
- `plot_matrix_csv`: read 2x2 magnitude/phase matrix data from CSV, declare its shape, and render the shared comparison layout.
- `plot_complex_json`: read a complex model JSON file and visualize it with the shared response theme.
- `plot_real_kernel_json`: read a real-kernel JSON file and visualize it with the shared response theme.

All plotted examples reuse `examples/support/mod.rs` for common plotting layout, styling, and file-output helpers.

## Typical Commands
```bash
cargo run --example fit_scalar
cargo run --example fit_csv
cargo run --example export_matrix_emt
cargo run --example plot_vector_report
```
