#[path = "support/mod.rs"]
mod support;

use num_complex::Complex64;
use support::{
    ComparisonSeries, draw_comparison_report, example_output_path, extract_matrix_entry, logspace,
    write_summary_markdown,
};
use vecfit::{FitOptions, Model};

/// 2-port network admittance with resonances and cross-coupling.
fn matrix_response(s: Complex64) -> [[Complex64; 2]; 2] {
    // Shared poles for coupling
    let p_real = Complex64::new(-25.0, 0.0);
    // Resonance at ~60 Hz
    let p1 = Complex64::new(-18.0, 377.0);
    // Resonance at ~400 Hz
    let p2 = Complex64::new(-50.0, 2513.0);
    // Resonance at ~2000 Hz
    let p3 = Complex64::new(-200.0, 12566.0);

    // Y11: strong self-admittance with all resonances
    let y11 = Complex64::new(0.010, 0.0)
        + Complex64::new(0.15, 0.0) / (s - p_real)
        + Complex64::new(0.20, -0.12) / (s - p1)
        + Complex64::new(0.20, 0.12) / (s - p1.conj())
        + Complex64::new(0.08, -0.10) / (s - p2)
        + Complex64::new(0.08, 0.10) / (s - p2.conj())
        + Complex64::new(0.03, -0.02) / (s - p3)
        + Complex64::new(0.03, 0.02) / (s - p3.conj());

    // Y12 = Y21: weaker mutual admittance (coupling), opposite sign at resonances
    let y12 = Complex64::new(-0.002, 0.0)
        + Complex64::new(-0.04, 0.0) / (s - p_real)
        + Complex64::new(-0.05, 0.03) / (s - p1)
        + Complex64::new(-0.05, -0.03) / (s - p1.conj())
        + Complex64::new(-0.02, 0.025) / (s - p2)
        + Complex64::new(-0.02, -0.025) / (s - p2.conj())
        + Complex64::new(-0.008, 0.005) / (s - p3)
        + Complex64::new(-0.008, -0.005) / (s - p3.conj());

    // Y22: different amplitude profile but same resonance structure
    let y22 = Complex64::new(0.008, 0.0)
        + Complex64::new(0.12, 0.0) / (s - p_real)
        + Complex64::new(0.14, -0.09) / (s - p2)
        + Complex64::new(0.14, 0.09) / (s - p2.conj())
        + Complex64::new(0.10, -0.07) / (s - p1)
        + Complex64::new(0.10, 0.07) / (s - p1.conj())
        + Complex64::new(0.04, -0.03) / (s - p3)
        + Complex64::new(0.04, 0.03) / (s - p3.conj());

    [[y11, y12], [y12, y22]]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency_hz = logspace(1.0, 10_000.0, 400);
    let sample_axis = frequency_hz
        .iter()
        .map(|hz| Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz))
        .collect::<Vec<_>>();
    let options = FitOptions::new().poles(10);
    let model = Model::fit_hz(
        &frequency_hz,
        |hz| matrix_response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz)),
        options,
    )?;

    let reference_response = sample_axis
        .iter()
        .map(|&sample| matrix_response(sample).map(|row| row.to_vec()).to_vec())
        .collect::<Vec<_>>();
    let fitted_response = model.evaluate_matrix(&sample_axis)?;

    let entries = [
        ("Y11", 0usize, 0usize),
        ("Y12", 0usize, 1usize),
        ("Y21", 1usize, 0usize),
        ("Y22", 1usize, 1usize),
    ];
    let reference_entries = entries
        .iter()
        .map(|(_, row_idx, col_idx)| extract_matrix_entry(&reference_response, *row_idx, *col_idx))
        .collect::<Vec<_>>();
    let fitted_entries = entries
        .iter()
        .map(|(_, row_idx, col_idx)| extract_matrix_entry(&fitted_response, *row_idx, *col_idx))
        .collect::<Vec<_>>();
    let traces = entries
        .iter()
        .enumerate()
        .map(|(entry_idx, (label, _, _))| ComparisonSeries {
            label,
            reference: &reference_entries[entry_idx],
            fitted: &fitted_entries[entry_idx],
        })
        .collect::<Vec<_>>();

    let plot_title = format!("Matrix EMT Fit ({} poles)", model.pole_count());
    let plot_path = example_output_path("matrix_emt_fit.png")?;
    draw_comparison_report(
        &plot_path,
        &frequency_hz,
        "Frequency (Hz)",
        &plot_title,
        &traces,
    )?;

    let summary_path = example_output_path("matrix_emt_fit.md")?;
    write_summary_markdown(
        &summary_path,
        &format!(
            "# Matrix EMT Fit\n\n- Poles: `{}`\n- Channels: `{}`\n- Shape: `{:?}`\n- Absolute RMSE: `{:.6e}`\n- Relative RMSE: `{:.6e}`\n",
            model.pole_count(),
            model.channels(),
            model.shape().dims(),
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
