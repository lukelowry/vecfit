#[path = "support/mod.rs"]
mod support;

use support::{
    ComparisonSeries, draw_comparison_report, example_data_path, example_output_path,
    write_summary_markdown,
};
use vecfit::{Csv, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let csv_path = example_data_path("scalar_transfer.csv");
    let csv_samples = Csv::from_path(&csv_path)?;
    let model = csv_samples.fit(Options::new().poles(8))?;
    let reference_response = csv_samples.scalars()?;
    let fitted_response = model.eval_scalar(csv_samples.axis())?;

    let plot_title = format!("Scalar CSV Fit ({} poles)", model.pole_count());
    let plot_path = example_output_path("scalar_csv_fit.png")?;
    draw_comparison_report(
        &plot_path,
        csv_samples.frequency_hz(),
        "Frequency (Hz)",
        &plot_title,
        &[ComparisonSeries {
            label: "Y",
            reference: &reference_response,
            fitted: &fitted_response,
        }],
    )?;

    let summary_path = example_output_path("scalar_csv_fit.md")?;
    write_summary_markdown(
        &summary_path,
        &format!(
            "# Scalar CSV Fit\n\n- Poles: `{}`\n- Channels: `{}`\n- Absolute RMSE: `{:.6e}`\n- Relative RMSE: `{:.6e}`\n",
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
