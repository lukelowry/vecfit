#[path = "support/mod.rs"]
mod support;

use support::{
    ComparisonSeries, draw_comparison_report, example_data_path, example_output_path,
    extract_matrix_entry, write_summary_markdown,
};
use vecfit::{CsvSamples, FitOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let csv_path = example_data_path("matrix_admittance_2x2.csv");
    let csv_samples = CsvSamples::from_path(&csv_path)?.matrix(2, 2)?;
    let model = csv_samples.fit(FitOptions::new().poles(10))?;
    let reference_response = csv_samples.matrices()?;
    let fitted_response = model.evaluate_matrix(csv_samples.axis())?;

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

    let plot_title = format!("Matrix CSV Fit ({} poles)", model.pole_count());
    let plot_path = example_output_path("matrix_csv_fit.png")?;
    draw_comparison_report(
        &plot_path,
        csv_samples.frequency_hz(),
        "Frequency (Hz)",
        &plot_title,
        &traces,
    )?;

    let summary_path = example_output_path("matrix_csv_fit.md")?;
    write_summary_markdown(
        &summary_path,
        &format!(
            "# Matrix CSV Fit\n\n- Poles: `{}`\n- Channels: `{}`\n- Shape: `{:?}`\n- Absolute RMSE: `{:.6e}`\n- Relative RMSE: `{:.6e}`\n",
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
