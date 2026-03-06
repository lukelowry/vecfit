#[path = "support/mod.rs"]
mod support;

use num_complex::Complex64;
use support::{
    ResponseSeries, draw_response_report, example_data_path, example_output_path, extract_channel,
    logspace, write_summary_markdown,
};
use vecfit::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json_path = example_data_path("real_kernel_model.json");
    let model = Model::from_real_json_path(&json_path)?;
    let lambda_values = logspace(0.01, 1000.0, 400);
    let sample_axis = lambda_values
        .iter()
        .map(|value| Complex64::new(*value, 0.0))
        .collect::<Vec<_>>();
    let evaluated_response = model.eval_vector(&sample_axis)?;
    let labels = ["K1", "K2"];
    let kernel_values = (0..labels.len())
        .map(|channel_idx| extract_channel(&evaluated_response, channel_idx))
        .collect::<Vec<_>>();
    let traces = labels
        .iter()
        .enumerate()
        .map(|(channel_idx, label)| ResponseSeries {
            label,
            values: &kernel_values[channel_idx],
        })
        .collect::<Vec<_>>();

    let plot_title = format!("Real-Kernel Response ({} poles)", model.pole_count());
    let plot_path = example_output_path("real_kernel_json_response.png")?;
    draw_response_report(&plot_path, &lambda_values, "Lambda", &plot_title, &traces)?;

    let summary_path = example_output_path("real_kernel_json_response.md")?;
    write_summary_markdown(
        &summary_path,
        &format!(
            "# Real-Kernel JSON Response\n\n- Poles: `{}`\n- Channels: `{}`\n",
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
