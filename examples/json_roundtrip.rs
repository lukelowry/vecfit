use num_complex::Complex64;
use vecfit::{Model, Options, complex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_axis = (0..80)
        .map(|k| Complex64::new(0.0, 1.0 + k as f64))
        .collect::<Vec<_>>();
    let model = Model::fit(
        complex(&sample_axis),
        |s| vec![1.0 / (s + 3.0), 0.5 / (s + 7.0)],
        Options::new().poles(3),
    )?;
    let json = model.to_json()?;
    let loaded_model = Model::from_json(&json)?;
    println!(
        "poles={}, channels={}, json_bytes={}",
        loaded_model.pole_count(),
        loaded_model.channels(),
        json.len()
    );
    Ok(())
}
