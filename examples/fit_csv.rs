use vecfit::{Csv, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let csv_text =
        "freq_Hz,|Y1|,ang_Y1\n1,10,0\n3,8,-10\n10,4,-45\n30,2,-60\n100,1,-80\n300,0.5,-85\n";
    let csv_samples = Csv::from_csv(csv_text)?;
    let model = csv_samples.fit(Options::new().poles(3))?;
    println!(
        "poles={}, channels={}, abs_rmse={:.3e}, rel_rmse={:.3e}",
        model.pole_count(),
        model.channels(),
        model.abs_rmse(),
        model.rel_rmse()
    );
    Ok(())
}
