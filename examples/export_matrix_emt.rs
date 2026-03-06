use num_complex::Complex64;
use vecfit::{DiscretizationMethod, FitOptions, Model};

fn matrix_response(s: Complex64) -> [[Complex64; 2]; 2] {
    [
        [
            Complex64::new(0.12, 0.0)
                + Complex64::new(2.0, 0.0) / (s + Complex64::new(40.0, 0.0))
                + Complex64::new(0.4, 0.0) / (s + Complex64::new(500.0, 0.0)),
            Complex64::new(-0.03, 0.0) + Complex64::new(0.7, 0.0) / (s + Complex64::new(80.0, 0.0)),
        ],
        [
            Complex64::new(-0.03, 0.0) + Complex64::new(0.7, 0.0) / (s + Complex64::new(80.0, 0.0)),
            Complex64::new(0.08, 0.0)
                + Complex64::new(1.6, 0.0) / (s + Complex64::new(30.0, 0.0))
                + Complex64::new(0.3, 0.0) / (s + Complex64::new(300.0, 0.0)),
        ],
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency_hz = (0..160)
        .map(|idx| {
            let t = idx as f64 / 159.0;
            10f64.powf(t * 4.0)
        })
        .collect::<Vec<_>>();
    let options = FitOptions::new().poles(5).real_only(true);
    let model = Model::fit_hz(
        &frequency_hz,
        |hz| matrix_response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz)),
        options,
    )?;
    let sections = model.real_sections()?;
    let discrete = model
        .state_space()?
        .discretize(1.0e-4, DiscretizationMethod::Tustin)?;
    println!(
        "poles={}, channels={}, sections={}, discrete_channels={}",
        model.pole_count(),
        model.channels(),
        sections.channels[0].sections.len(),
        discrete.channels.len()
    );
    Ok(())
}
