use num_complex::Complex64;
use vecfit::{DiscretizationMethod, Model, Options, hz};

fn matrix_response(s: Complex64) -> [[Complex64; 2]; 2] {
    [
        [
            0.12 + 2.0 / (s + 40.0) + 0.4 / (s + 500.0),
            -0.03 + 0.7 / (s + 80.0),
        ],
        [
            -0.03 + 0.7 / (s + 80.0),
            0.08 + 1.6 / (s + 30.0) + 0.3 / (s + 300.0),
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
    let options = Options::new().poles(5).real_only(true);
    let model = Model::fit(
        hz(&frequency_hz),
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
