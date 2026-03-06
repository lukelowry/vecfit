#[path = "support/mod.rs"]
mod support;

use num_complex::Complex64;
use support::{
    ComparisonSeries, draw_comparison_report, example_output_path, extract_channel, logspace,
    write_summary_markdown,
};
use vecfit::{FitOptions, Model};

/// 3-phase characteristic admittance with different resonance patterns per phase.
fn vector_response(s: Complex64) -> Vec<Complex64> {
    // Phase A: dominant resonance at ~50 Hz, secondary at ~500 Hz
    let ya = Complex64::new(0.004, 0.0)
        + Complex64::new(0.05, 0.0) / (s - Complex64::new(-10.0, 0.0))
        + Complex64::new(0.15, -0.10) / (s - Complex64::new(-20.0, 314.0))
        + Complex64::new(0.15, 0.10) / (s - Complex64::new(-20.0, -314.0))
        + Complex64::new(0.05, -0.08) / (s - Complex64::new(-80.0, 3142.0))
        + Complex64::new(0.05, 0.08) / (s - Complex64::new(-80.0, -3142.0));

    // Phase B: dominant resonance at ~120 Hz, secondary at ~1500 Hz
    let yb = Complex64::new(0.003, 0.0)
        + Complex64::new(0.04, 0.0) / (s - Complex64::new(-12.0, 0.0))
        + Complex64::new(0.10, -0.12) / (s - Complex64::new(-35.0, 754.0))
        + Complex64::new(0.10, 0.12) / (s - Complex64::new(-35.0, -754.0))
        + Complex64::new(0.04, -0.05) / (s - Complex64::new(-150.0, 9425.0))
        + Complex64::new(0.04, 0.05) / (s - Complex64::new(-150.0, -9425.0));

    // Phase C: broad resonance at ~80 Hz, sharp resonance at ~800 Hz
    let yc = Complex64::new(0.005, 0.0)
        + Complex64::new(0.06, 0.0) / (s - Complex64::new(-6.0, 0.0))
        + Complex64::new(0.08, -0.06) / (s - Complex64::new(-50.0, 503.0))
        + Complex64::new(0.08, 0.06) / (s - Complex64::new(-50.0, -503.0))
        + Complex64::new(0.07, -0.09) / (s - Complex64::new(-30.0, 5027.0))
        + Complex64::new(0.07, 0.09) / (s - Complex64::new(-30.0, -5027.0));

    vec![ya, yb, yc]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency_hz = logspace(1.0, 10_000.0, 400);
    let sample_axis = frequency_hz
        .iter()
        .map(|hz| Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz))
        .collect::<Vec<_>>();
    let model = Model::fit_hz(
        &frequency_hz,
        |hz| vector_response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz)),
        FitOptions::new().poles(18),
    )?;
    let reference_response = sample_axis
        .iter()
        .map(|&sample| vector_response(sample))
        .collect::<Vec<_>>();
    let fitted_response = model.evaluate_vector(&sample_axis)?;

    let labels = ["Ch 1", "Ch 2", "Ch 3"];
    let reference_channels = (0..labels.len())
        .map(|channel_idx| extract_channel(&reference_response, channel_idx))
        .collect::<Vec<_>>();
    let fitted_channels = (0..labels.len())
        .map(|channel_idx| extract_channel(&fitted_response, channel_idx))
        .collect::<Vec<_>>();
    let traces = labels
        .iter()
        .enumerate()
        .map(|(channel_idx, label)| ComparisonSeries {
            label,
            reference: &reference_channels[channel_idx],
            fitted: &fitted_channels[channel_idx],
        })
        .collect::<Vec<_>>();

    let plot_title = format!("Vector Fit ({} poles)", model.pole_count());
    let plot_path = example_output_path("vector_fit.png")?;
    draw_comparison_report(
        &plot_path,
        &frequency_hz,
        "Frequency (Hz)",
        &plot_title,
        &traces,
    )?;

    let summary_path = example_output_path("vector_fit.md")?;
    write_summary_markdown(
        &summary_path,
        &format!(
            "# Vector Fit\n\n- Poles: `{}`\n- Channels: `{}`\n- Absolute RMSE: `{:.6e}`\n- Relative RMSE: `{:.6e}`\n",
            model.pole_count(),
            model.channels(),
            model.abs_rmse(),
            model.rel_rmse()
        ),
    )?;

    println!(
        "saved {} and {}",
        plot_path.display(),
        summary_path.display()
    );
    Ok(())
}
