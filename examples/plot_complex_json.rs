#[path = "support/mod.rs"]
mod support;

use num_complex::Complex64;
use support::{
    ResponseSeries, draw_response_report, example_data_path, example_output_path, extract_channel,
    logspace, write_summary_markdown,
};
use vecfit::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json_path = example_data_path("complex_model.json");
    let model = Model::from_json_path(&json_path)?;
    let frequency_hz = logspace(1.0, 10_000.0, 400);
    let sample_axis = frequency_hz
        .iter()
        .map(|hz| Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz))
        .collect::<Vec<_>>();
    let evaluated_response = model.eval_vector(&sample_axis)?;
    let labels = ["Ch 1", "Ch 2"];
    let channel_values = (0..labels.len())
        .map(|channel_idx| extract_channel(&evaluated_response, channel_idx))
        .collect::<Vec<_>>();
    let traces = labels
        .iter()
        .enumerate()
        .map(|(channel_idx, label)| ResponseSeries {
            label,
            values: &channel_values[channel_idx],
        })
        .collect::<Vec<_>>();

    let plot_title = format!("Complex Response ({} poles)", model.pole_count());
    let plot_path = example_output_path("complex_json_response.png")?;
    draw_response_report(
        &plot_path,
        &frequency_hz,
        "Frequency (Hz)",
        &plot_title,
        &traces,
    )?;

    let summary_path = example_output_path("complex_json_response.md")?;
    write_summary_markdown(
        &summary_path,
        &format!(
            "# Complex JSON Response\n\n- Poles: `{}`\n- Channels: `{}`\n",
            model.pole_count(),
            model.channels()
        ),
    )?;
    println!(
        "saved {} and {}",
        plot_path.display(),
        summary_path.display()
    );
    Ok(())
}
