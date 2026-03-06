use num_complex::Complex64;
use vecfit::{Model, Options, c, hz};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency_hz = (1..200).map(|k| k as f64 * 5.0).collect::<Vec<_>>();
    let model = Model::fit(
        hz(&frequency_hz),
        |hz| {
            let s = Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz);
            let p1 = c(-20.0, 300.0);
            let p2 = c(-50.0, 1500.0);
            vec![
                // Channel 1: resonance near 48 Hz
                0.01 + c(0.1, -0.06) / (s - p1) + c(0.1, 0.06) / (s - p1.conj()),
                // Channel 2: resonance near 239 Hz
                0.008 + c(0.06, -0.08) / (s - p2) + c(0.06, 0.08) / (s - p2.conj()),
            ]
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
