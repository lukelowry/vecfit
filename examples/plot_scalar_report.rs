#[path = "support/mod.rs"]
mod support;

use num_complex::Complex64;
use support::{
    ComparisonSeries, draw_comparison_report, example_output_path, logspace, write_summary_markdown,
};
use vecfit::{FitOptions, Model};

/// Characteristic admittance-like transfer function with multiple resonances.
fn target_response(s: Complex64) -> Complex64 {
    // DC / high-frequency asymptote
    let d = Complex64::new(0.005, 0.0);
    // Slow real pole (low-frequency roll-off)
    let r0 = Complex64::new(0.06, 0.0);
    let p0 = Complex64::new(-8.0, 0.0);
    // Resonance at ~30 Hz (β ≈ 188)
    let r1 = Complex64::new(0.12, -0.08);
    let p1 = Complex64::new(-15.0, 188.0);
    // Resonance at ~300 Hz (β ≈ 1885)
    let r2 = Complex64::new(0.06, -0.10);
    let p2 = Complex64::new(-60.0, 1885.0);
    // Resonance at ~2500 Hz (β ≈ 15708)
    let r3 = Complex64::new(0.03, -0.04);
    let p3 = Complex64::new(-400.0, 15708.0);

    d + r0 / (s - p0)
        + r1 / (s - p1)
        + r1.conj() / (s - p1.conj())
        + r2 / (s - p2)
        + r2.conj() / (s - p2.conj())
        + r3 / (s - p3)
        + r3.conj() / (s - p3.conj())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency_hz = logspace(1.0, 10_000.0, 400);
    let sample_axis = frequency_hz
        .iter()
        .map(|hz| Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz))
        .collect::<Vec<_>>();

    let model = Model::fit_hz(
        &frequency_hz,
        |hz| target_response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz)),
        FitOptions::new().poles(10),
    )?;
    let reference_response = sample_axis
        .iter()
        .map(|&sample| target_response(sample))
        .collect::<Vec<_>>();
    let fitted_response = model.evaluate_scalar(&sample_axis)?;

    let plot_title = format!("Scalar Fit ({} poles)", model.pole_count());
    let plot_path = example_output_path("scalar_fit.png")?;
    draw_comparison_report(
        &plot_path,
        &frequency_hz,
        "Frequency (Hz)",
        &plot_title,
        &[ComparisonSeries {
            label: "Y",
            reference: &reference_response,
            fitted: &fitted_response,
        }],
    )?;

    let formatted_poles = model
        .poles()
        .iter()
        .map(|pole| format!("- `{:.6} + j{:.6}`", pole.re, pole.im))
        .collect::<Vec<_>>()
        .join("\n");
    let summary_path = example_output_path("scalar_fit.md")?;
    write_summary_markdown(
        &summary_path,
        &format!(
            "# Scalar Fit\n\n- Poles: `{}`\n- Channels: `{}`\n- Absolute RMSE: `{:.6e}`\n- Relative RMSE: `{:.6e}`\n\n## Poles\n{}\n",
            model.pole_count(),
            model.channels(),
            model.abs_rmse(),
            model.rel_rmse(),
            formatted_poles
        ),
    )?;

    println!(
        "saved {} and {}",
        plot_path.display(),
        summary_path.display()
    );
    Ok(())
}
