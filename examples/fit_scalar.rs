use num_complex::Complex64;
use vecfit::{Model, Options, c, complex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_axis = (0..300)
        .map(|k| Complex64::new(0.0, 1.0 + k as f64 * 0.5))
        .collect::<Vec<_>>();
    let model = Model::fit(
        complex(&sample_axis),
        |s| {
            // Resonant transfer function with a conjugate pair near ω ≈ 50
            0.05 + 1.0 / (s + 3.0)
                + c(0.8, -0.5) / (s + c(5.0, -50.0))
                + c(0.8, 0.5) / (s + c(5.0, 50.0))
        },
        Options::new().poles(6),
    )?;
    println!(
        "poles={}, channels={}, abs_rmse={:.3e}, rel_rmse={:.3e}",
        model.pole_count(),
        model.channels(),
        model.abs_rmse(),
        model.rel_rmse()
    );
    Ok(())
}
