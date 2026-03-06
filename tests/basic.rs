use approx::assert_relative_eq;
use num_complex::Complex64;
use vecfit::{
    ChannelStateSpace, Csv, DiscretizationMethod, FlatResponse, IntoResponse, Layout, Model,
    ModelParts, Options, Shape, StateSpaceModel, VecfitError, complex, hz, rad, real,
};

fn build_samples(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|k| Complex64::new(0.0, 0.5 + k as f64))
        .collect()
}

/// Known transfer function used across multiple tests.
fn reference_scalar(s: Complex64) -> Complex64 {
    Complex64::from(0.05) + 1.2 / (s + 3.0) + 0.4 / (s + 15.0)
}

// ============================================================
// Axis wrapper tests
// ============================================================

#[test]
fn hz_axis_maps_to_j_2pi_f() {
    let freq = vec![1.0, 10.0, 100.0];
    let axis = hz(&freq);
    let mapped = axis.to_complex();
    for (f, s) in freq.iter().zip(mapped.iter()) {
        assert_relative_eq!(s.re, 0.0, epsilon = 1e-15);
        assert_relative_eq!(s.im, 2.0 * std::f64::consts::PI * f, epsilon = 1e-12);
    }
}

#[test]
fn rad_axis_maps_to_j_omega() {
    let omega = vec![1.0, 10.0, 100.0];
    let axis = rad(&omega);
    let mapped = axis.to_complex();
    for (w, s) in omega.iter().zip(mapped.iter()) {
        assert_relative_eq!(s.re, 0.0, epsilon = 1e-15);
        assert_relative_eq!(s.im, *w, epsilon = 1e-15);
    }
}

#[test]
fn real_axis_maps_to_real_line() {
    let x = vec![0.5, 1.0, 5.0, 20.0];
    let axis = real(&x);
    let mapped = axis.to_complex();
    for (xv, s) in x.iter().zip(mapped.iter()) {
        assert_relative_eq!(s.re, *xv, epsilon = 1e-15);
        assert_relative_eq!(s.im, 0.0, epsilon = 1e-15);
    }
}

#[test]
fn complex_axis_is_passthrough() {
    let pts = vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.5)];
    let axis = complex(&pts);
    let mapped = axis.to_complex();
    for (orig, mapped) in pts.iter().zip(mapped.iter()) {
        assert_eq!(orig, mapped);
    }
}

// ============================================================
// Fit accuracy regression tests
// ============================================================

#[test]
fn scalar_fit_accuracy_regression() {
    let sample_axis = build_samples(200);
    let model = Model::fit(
        complex(&sample_axis),
        reference_scalar,
        Options::new().poles(4),
    )
    .expect("fit should succeed");

    assert!(
        model.abs_rmse() < 1e-6,
        "scalar RMSE should be tight for a known rational function, got {:.3e}",
        model.abs_rmse()
    );
    assert!(model.is_stable(), "all poles should be stable");

    // Verify actual values match reference at every sample point
    let fitted = model
        .eval_scalar(&sample_axis)
        .expect("scalar eval should succeed");
    for (i, (y, s)) in fitted.iter().zip(sample_axis.iter()).enumerate() {
        let expected = reference_scalar(*s);
        let err = (y - expected).norm();
        assert!(
            err < 1e-4,
            "sample {i}: fitted {y} vs reference {expected}, error {err:.3e}"
        );
    }
}

#[test]
fn vector_fit_accuracy_regression() {
    let sample_axis = build_samples(150);
    let model = Model::fit(
        complex(&sample_axis),
        |s| vec![1.0 / (s + 2.0), 0.5 / (s + 8.0), 0.3 / (s + 20.0)],
        Options::new().poles(4),
    )
    .expect("vector fit should succeed");

    assert_eq!(model.channels(), 3);
    assert!(
        model.abs_rmse() < 1e-3,
        "vector RMSE should be tight, got {:.3e}",
        model.abs_rmse()
    );

    let vectors = model
        .eval_vector(&sample_axis)
        .expect("vector eval should succeed");
    for (i, (vec_val, s)) in vectors.iter().zip(sample_axis.iter()).enumerate() {
        let expected = [1.0 / (s + 2.0), 0.5 / (s + 8.0), 0.3 / (s + 20.0)];
        for (ch, (y, e)) in vec_val.iter().zip(expected.iter()).enumerate() {
            let err = (y - e).norm();
            assert!(
                err < 1e-2,
                "sample {i} ch {ch}: fitted {y} vs reference {e}, error {err:.3e}"
            );
        }
    }
}

#[test]
fn rad_axis_fit_matches_hz_axis() {
    let freq: Vec<f64> = (1..=60).map(|k| k as f64).collect();
    let omega: Vec<f64> = freq
        .iter()
        .map(|f| 2.0 * std::f64::consts::PI * f)
        .collect();

    let hz_model = Model::fit(
        hz(&freq),
        |f| {
            let w = 2.0 * std::f64::consts::PI * f;
            1.0 / (1.0 + w)
        },
        Options::new().poles(2),
    )
    .expect("hz fit should succeed");

    let rad_model = Model::fit(rad(&omega), |w| 1.0 / (1.0 + w), Options::new().poles(2))
        .expect("rad fit should succeed");

    // Both should achieve good accuracy
    assert!(
        hz_model.abs_rmse() < 0.1,
        "hz RMSE = {:.3e}",
        hz_model.abs_rmse()
    );
    assert!(
        rad_model.abs_rmse() < 0.1,
        "rad RMSE = {:.3e}",
        rad_model.abs_rmse()
    );
}

#[test]
fn real_axis_fits_decay_kernel() {
    // Fit f(x) = 2/(x+1) + 0.5/(x+10) on the real line
    let x: Vec<f64> = (1..=100).map(|k| k as f64 * 0.5).collect();
    let model = Model::fit(
        real(&x),
        |x| 2.0 / (x + 1.0) + 0.5 / (x + 10.0),
        Options::new().poles(3),
    )
    .expect("real-axis fit should succeed");

    assert!(
        model.abs_rmse() < 1e-3,
        "real-axis RMSE should be tight, got {:.3e}",
        model.abs_rmse()
    );
}

// ============================================================
// fit_samples (raw buffer API)
// ============================================================

#[test]
fn fit_samples_matches_closure_fit() {
    let sample_axis = build_samples(100);

    // Build reference values manually (flat row-major, 2 channels)
    let flat_values: Vec<Complex64> = sample_axis
        .iter()
        .flat_map(|s| vec![1.0 / (s + 3.0), 0.5 / (s + 8.0)])
        .collect();

    let model = Model::fit_samples(
        complex(&sample_axis),
        &flat_values,
        Shape::vector(2).expect("shape"),
        Options::new().poles(3),
    )
    .expect("fit_samples should succeed");

    assert_eq!(model.channels(), 2);
    assert!(
        model.abs_rmse() < 0.01,
        "fit_samples RMSE = {:.3e}",
        model.abs_rmse()
    );
}

// ============================================================
// Shape inference
// ============================================================

#[test]
fn shape_inference_scalars_arrays_vecs() {
    // Scalar
    let s = (1.0f64).into_response().expect("scalar");
    assert!(s.shape.is_scalar());

    let s = Complex64::new(1.0, 2.0)
        .into_response()
        .expect("complex scalar");
    assert!(s.shape.is_scalar());

    // Fixed-size array → vector
    let v = [1.0, 2.0, 3.0].into_response().expect("array vector");
    assert_eq!(v.shape.expect_vector().unwrap(), 3);

    // Vec → vector
    let v = vec![1.0, 2.0].into_response().expect("vec vector");
    assert_eq!(v.shape.expect_vector().unwrap(), 2);

    // Nested array → matrix
    let m = [[1.0, 2.0], [3.0, 4.0]]
        .into_response()
        .expect("array matrix");
    assert_eq!(m.shape.expect_matrix().unwrap(), (2, 2));

    // Vec<Vec> → matrix
    let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        .into_response()
        .expect("vec matrix");
    assert_eq!(m.shape.expect_matrix().unwrap(), (2, 2));
}

#[test]
fn shape_infer_square() {
    let scalar = Shape::infer_square(1).expect("infer_square(1)");
    assert!(scalar.is_scalar());

    let matrix = Shape::infer_square(4).expect("infer_square(4)");
    assert_eq!(matrix.expect_matrix().unwrap(), (2, 2));

    let vector = Shape::infer_square(3).expect("infer_square(3)");
    assert_eq!(vector.expect_vector().unwrap(), 3);
}

// ============================================================
// CSV / TSV / SSV / custom delimiter parsing
// ============================================================

#[test]
fn csv_rectangular_format() {
    let csv = "freq_Hz,re_Y1,im_Y1\n1,3.0,4.0\n10,1.0,-2.0\n";
    let parsed = Csv::from_csv(csv).expect("rectangular csv");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed.channels(), 1);
    let scalars = parsed.scalars().expect("scalars");
    assert_relative_eq!(scalars[0].re, 3.0, epsilon = 1e-12);
    assert_relative_eq!(scalars[0].im, 4.0, epsilon = 1e-12);
    assert_relative_eq!(scalars[1].re, 1.0, epsilon = 1e-12);
    assert_relative_eq!(scalars[1].im, -2.0, epsilon = 1e-12);
}

#[test]
fn csv_magnitude_phase_format() {
    let csv = "freq_Hz,|Y1|,ang_Y1\n1,10,45\n10,5,-90\n";
    let parsed = Csv::from_csv(csv).expect("mag/phase csv");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed.channels(), 1);
    // First row: magnitude 10, angle 45 degrees
    let scalars = parsed.scalars().expect("scalars");
    let expected_re = 10.0 * (45.0f64.to_radians()).cos();
    let expected_im = 10.0 * (45.0f64.to_radians()).sin();
    assert_relative_eq!(scalars[0].re, expected_re, epsilon = 1e-10);
    assert_relative_eq!(scalars[0].im, expected_im, epsilon = 1e-10);
    // Frequency maps to j*2*pi*f
    assert_relative_eq!(
        parsed.axis()[0].im,
        2.0 * std::f64::consts::PI,
        epsilon = 1e-12
    );
}

#[test]
fn csv_fit_produces_valid_model() {
    let csv = "freq_Hz,re_f1,im_f1\n1,0.95,-0.31\n5,0.35,-0.72\n10,0.10,-0.98\n50,0.02,-1.0\n100,0.01,-1.0\n";
    let model = Csv::from_csv(csv)
        .expect("csv parse")
        .fit(Options::new().poles(2))
        .expect("fit");
    assert!(model.shape().is_scalar());
    assert!(model.abs_rmse().is_finite());
}

#[test]
fn csv_rejects_incomplete_column_pairs() {
    let csv = "freq_Hz,|Y1|,ang_Y1,|Y2|\n1,10,45,3\n";
    let err = Csv::from_csv(csv).expect_err("should reject incomplete pairs");
    assert!(matches!(err, VecfitError::Csv(_)));
}

#[test]
fn csv_rejects_empty_data() {
    let csv = "freq_Hz,|Y1|,ang_Y1\n";
    assert!(Csv::from_csv(csv).is_err());
}

#[test]
fn tsv_parsing_works() {
    let tsv = "freq_Hz\t|Y1|\tang_Y1\n1\t10\t45\n10\t5\t-90\n";
    let parsed = Csv::from_tsv(tsv).expect("tsv parse");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed.channels(), 1);
}

#[test]
fn ssv_parsing_works() {
    let ssv = "freq_Hz;re_Y1;im_Y1\n1;3.0;4.0\n10;1.0;-2.0\n";
    let parsed = Csv::from_ssv(ssv).expect("ssv parse");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed.channels(), 1);
    let scalars = parsed.scalars().expect("scalars");
    assert_relative_eq!(scalars[0].re, 3.0, epsilon = 1e-12);
    assert_relative_eq!(scalars[0].im, 4.0, epsilon = 1e-12);
}

#[test]
fn custom_delimiter_parsing_works() {
    let pipe = "freq_Hz|re_Y1|im_Y1\n1|3.0|4.0\n10|1.0|-2.0\n";
    let parsed = Csv::from_delimited(pipe, b'|').expect("pipe-delimited parse");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed.channels(), 1);
    let scalars = parsed.scalars().expect("scalars");
    assert_relative_eq!(scalars[0].re, 3.0, epsilon = 1e-12);
}

// ============================================================
// Touchstone
// ============================================================

// (Touchstone has 13 unit tests in src/touchstone.rs covering parsing,
//  option lines, data formats, frequency units, and multi-port files.)

// ============================================================
// JSON round-trip
// ============================================================

#[test]
fn complex_json_roundtrip() {
    let sample_axis = build_samples(80);
    let model = Model::fit(
        complex(&sample_axis),
        |sk| vec![1.0 / (sk + 3.0), 2.0 / (sk + 6.0)],
        Options::new().poles(3),
    )
    .expect("fit should succeed");
    let json = model.to_json().expect("JSON export");
    let loaded = Model::from_json(&json).expect("JSON import");
    assert_eq!(loaded.channels(), model.channels());
    assert_eq!(loaded.pole_count(), model.pole_count());

    // Verify loaded model evaluates identically
    let orig = model.eval_flat(&sample_axis).expect("orig eval");
    let reloaded = loaded.eval_flat(&sample_axis).expect("loaded eval");
    for (a, b) in orig.values.iter().zip(reloaded.values.iter()) {
        assert_relative_eq!(a.re, b.re, epsilon = 1e-12);
        assert_relative_eq!(a.im, b.im, epsilon = 1e-12);
    }
}

#[test]
fn complex_json_roundtrip_preserves_shape_and_layout() {
    let model = Model::from_parts(ModelParts {
        poles: vec![Complex64::new(-2.0, 0.0)],
        residues: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
        channels: 4,
        constant_terms: vec![Complex64::new(0.1, 0.0); 4],
        proportional_terms: vec![Complex64::new(0.0, 0.0); 4],
        shape: Shape::matrix(2, 2).expect("shape"),
        layout: Layout::ColumnMajor,
        report: Default::default(),
    })
    .expect("model parts");
    let json = model.to_json().expect("export");
    let loaded = Model::from_json(&json).expect("import");
    assert_eq!(loaded.shape(), model.shape());
    assert_eq!(loaded.layout(), model.layout());
}

#[test]
fn real_kernel_json_roundtrip() {
    let model = Model::from_parts(ModelParts {
        poles: vec![Complex64::new(-1.0, 0.0), Complex64::new(-5.0, 0.0)],
        residues: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
        channels: 1,
        constant_terms: vec![Complex64::new(0.1, 0.0)],
        proportional_terms: vec![Complex64::new(0.0, 0.0)],
        shape: Shape::scalar(),
        layout: Layout::RowMajor,
        report: Default::default(),
    })
    .expect("model parts");
    let json = model
        .to_real_json(Some("test".to_string()))
        .expect("real JSON export");
    let loaded = Model::from_real_json(&json).expect("real JSON import");
    assert_eq!(loaded.pole_count(), 2);
}

#[test]
fn real_kernel_json_preserves_shape_and_layout() {
    let model = Model::from_parts(ModelParts {
        poles: vec![Complex64::new(-1.0, 0.0), Complex64::new(-5.0, 0.0)],
        residues: vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.5, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.5, 0.0),
        ],
        channels: 4,
        constant_terms: vec![Complex64::new(0.1, 0.0); 4],
        proportional_terms: vec![Complex64::new(0.0, 0.0); 4],
        shape: Shape::matrix(2, 2).expect("shape"),
        layout: Layout::ColumnMajor,
        report: Default::default(),
    })
    .expect("model parts");
    let json = model
        .to_real_json(Some("matrix".to_string()))
        .expect("export");
    let loaded = Model::from_real_json(&json).expect("import");
    assert_eq!(loaded.shape(), model.shape());
    assert_eq!(loaded.layout(), model.layout());
}

#[test]
fn from_json_rejects_missing_poles_field() {
    let json = r#"{"residues":[],"d":[],"e":[],"rmse":0,"iters":0}"#;
    assert!(Model::from_json(json).is_err());
}

// ============================================================
// Evaluation helpers
// ============================================================

#[test]
fn magnitude_db_and_phase_deg_correct() {
    // Build a known model and verify magnitude_db / phase_deg against hand-computed values
    let model = Model::from_parts(ModelParts {
        poles: vec![Complex64::new(-1.0, 0.0)],
        residues: vec![Complex64::new(1.0, 0.0)],
        channels: 1,
        constant_terms: vec![Complex64::new(0.0, 0.0)],
        proportional_terms: vec![Complex64::new(0.0, 0.0)],
        shape: Shape::scalar(),
        layout: Layout::RowMajor,
        report: Default::default(),
    })
    .expect("model parts");

    // Evaluate at s = j*1: H(s) = 1/(j*1 + 1) = (1 - j) / 2
    let s = vec![Complex64::new(0.0, 1.0)];
    let expected = Complex64::new(1.0, 0.0) / Complex64::new(1.0, 1.0);

    let mag_db = model.magnitude_db(&s).expect("magnitude_db");
    let phase = model.phase_deg(&s).expect("phase_deg");

    let expected_mag_db = 20.0 * expected.norm().log10();
    let expected_phase = expected.arg().to_degrees();

    assert_relative_eq!(mag_db[0][0], expected_mag_db, epsilon = 1e-10);
    assert_relative_eq!(phase[0][0], expected_phase, epsilon = 1e-10);
}

#[test]
fn channel_errors_against_known_reference() {
    let sample_axis = build_samples(100);
    let model = Model::fit(
        complex(&sample_axis),
        reference_scalar,
        Options::new().poles(4),
    )
    .expect("fit");

    // Build reference values (flat, 1 channel)
    let reference: Vec<Complex64> = sample_axis.iter().map(|s| reference_scalar(*s)).collect();
    let errors = model
        .channel_errors(&sample_axis, &reference)
        .expect("channel_errors");

    assert_eq!(errors.abs_rmse.len(), 1);
    assert_eq!(errors.rel_rmse.len(), 1);
    assert!(
        errors.abs_rmse[0] < 1e-5,
        "abs_rmse = {:.3e}",
        errors.abs_rmse[0]
    );
    assert!(
        errors.rel_rmse[0] < 1e-4,
        "rel_rmse = {:.3e}",
        errors.rel_rmse[0]
    );
}

// ============================================================
// Matrix fit + evaluation
// ============================================================

#[test]
fn matrix_fit_evaluates_correctly() {
    let freqs = (1..120).map(|k| k as f64).collect::<Vec<_>>();
    let model = Model::fit(
        hz(&freqs),
        |f| {
            let w = 2.0 * std::f64::consts::PI * f;
            [
                [1.0 / (1.0 + w), 0.5 / (2.0 + w)],
                [0.5 / (2.0 + w), 1.2 / (3.0 + w)],
            ]
        },
        Options::new().poles(4),
    )
    .expect("matrix fit");

    assert_eq!(model.shape().expect_matrix().unwrap(), (2, 2));
    assert_eq!(model.channels(), 4);

    let eval = model
        .eval_matrix(
            &freqs
                .iter()
                .map(|hz| Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz))
                .collect::<Vec<_>>(),
        )
        .expect("matrix eval");

    assert_eq!(eval.len(), freqs.len());
    assert_eq!(eval[0].len(), 2);
    assert_eq!(eval[0][0].len(), 2);

    assert!(
        model.abs_rmse() < 1e-4,
        "matrix RMSE = {:.3e}",
        model.abs_rmse()
    );
}

// ============================================================
// EMT export
// ============================================================

#[test]
fn emt_real_sections_and_discretization() {
    let model = Model::from_parts(ModelParts {
        poles: vec![Complex64::new(-1.0, 0.0), Complex64::new(-4.0, 0.0)],
        residues: vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
        channels: 1,
        constant_terms: vec![Complex64::new(0.1, 0.0)],
        proportional_terms: vec![Complex64::new(0.0, 0.0)],
        shape: Shape::scalar(),
        layout: Layout::RowMajor,
        report: Default::default(),
    })
    .expect("model parts");

    let sections = model.real_sections().expect("real sections");
    assert_eq!(sections.channels.len(), 1);

    let ss = model.state_space().expect("state space");
    let backward = ss
        .discretize(1e-4, DiscretizationMethod::BackwardEuler)
        .expect("backward euler");
    assert_eq!(backward.channels.len(), 1);

    let tustin = ss
        .discretize(1e-4, DiscretizationMethod::Tustin)
        .expect("tustin");
    assert_eq!(tustin.channels.len(), 1);
}

#[test]
fn real_only_matrix_emt_exports() {
    fn response(s: Complex64) -> [[Complex64; 2]; 2] {
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

    let freqs = (0..160)
        .map(|idx| {
            let t = idx as f64 / 159.0;
            10f64.powf(t * 4.0)
        })
        .collect::<Vec<_>>();

    let model = Model::fit(
        hz(&freqs),
        |f| response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * f)),
        Options::new().poles(5).real_only(true),
    )
    .expect("real-only matrix fit");

    let sections = model.real_sections().expect("sections");
    let discrete = model
        .state_space()
        .expect("state space")
        .discretize(1e-4, DiscretizationMethod::Tustin)
        .expect("discretize");

    assert_eq!(sections.channels.len(), 4);
    assert_eq!(discrete.channels.len(), 4);
}

// ============================================================
// Column-major layout
// ============================================================

#[test]
fn column_major_evaluate_and_json_roundtrip() {
    let model = Model::from_parts(ModelParts {
        poles: vec![Complex64::new(-2.0, 0.0), Complex64::new(-8.0, 0.0)],
        residues: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.7, 0.0),
        ],
        channels: 2,
        constant_terms: vec![Complex64::new(0.1, 0.0); 2],
        proportional_terms: vec![Complex64::new(0.0, 0.0); 2],
        shape: Shape::vector(2).expect("shape"),
        layout: Layout::ColumnMajor,
        report: Default::default(),
    })
    .expect("model parts");

    let axis = build_samples(50);
    let flat = model.eval_flat(&axis).expect("eval");
    assert_eq!(flat.samples, 50);

    let json = model.to_json().expect("export");
    let loaded = Model::from_json(&json).expect("import");
    assert_eq!(loaded.layout(), Layout::ColumnMajor);
    assert_eq!(loaded.channels(), 2);
}

// ============================================================
// Error handling
// ============================================================

#[test]
fn fit_rejects_zero_poles() {
    let axis = build_samples(20);
    let err = Model::fit(complex(&axis), |s| 1.0 / (s + 3.0), Options::new().poles(0))
        .expect_err("zero poles should be rejected");
    assert!(matches!(err, VecfitError::InvalidInput(_)));
}

#[test]
fn underdetermined_fit_returns_error() {
    let axis = build_samples(3);
    let err = Model::fit(
        complex(&axis),
        |s| 1.0 / (s + 3.0) + 0.1,
        Options::new().poles(3),
    )
    .expect_err("underdetermined fit should fail");
    assert!(matches!(err, VecfitError::InvalidInput(_)));
}

#[test]
fn fit_rejects_invalid_weight_length() {
    let axis = build_samples(20);
    let err = Model::fit(
        complex(&axis),
        |s| 1.0 / (s + 3.0) + 0.1,
        Options::new().poles(2).weights(vec![1.0; 3]),
    )
    .expect_err("wrong-length weights should be rejected");
    assert!(matches!(err, VecfitError::Dimension(_)));
}

#[test]
fn fit_rejects_negative_weights() {
    let axis = build_samples(20);
    let err = Model::fit(
        complex(&axis),
        |s| 1.0 / (s + 3.0) + 0.1,
        Options::new().poles(2).weights(vec![-1.0; axis.len()]),
    )
    .expect_err("negative weights should be rejected");
    assert!(matches!(err, VecfitError::InvalidInput(_)));
}

#[test]
fn fit_rejects_layout_mismatch() {
    let axis = vec![Complex64::new(0.0, 1.0), Complex64::new(0.0, 2.0)];
    let err = Model::fit(
        complex(&axis),
        |sk| {
            let idx = if sk.im < 1.5 { 0usize } else { 1usize };
            FlatResponse::new(
                vec![
                    Complex64::new(idx as f64, 0.0),
                    Complex64::new(idx as f64 + 1.0, 0.0),
                ],
                Shape::vector(2).expect("shape"),
                if idx % 2 == 0 {
                    Layout::RowMajor
                } else {
                    Layout::ColumnMajor
                },
            )
            .expect("response")
        },
        Options::new().poles(1),
    )
    .expect_err("mixed layouts should be rejected");
    assert!(matches!(err, VecfitError::Shape(_)));
}

#[test]
fn invalid_model_parts_are_rejected() {
    let err = Model::from_parts(ModelParts {
        poles: vec![Complex64::new(-1.0, 0.0)],
        residues: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
        channels: 2,
        constant_terms: vec![Complex64::new(0.1, 0.0)], // wrong count
        proportional_terms: vec![Complex64::new(0.0, 0.0); 2],
        shape: Shape::vector(2).expect("shape"),
        layout: Layout::RowMajor,
        report: Default::default(),
    })
    .expect_err("invalid constant_terms count should fail");
    assert!(err.to_string().contains("constant term count"));
}

#[test]
fn invalid_state_space_returns_error() {
    let state_space = StateSpaceModel {
        shape: Shape::scalar(),
        layout: Layout::RowMajor,
        channels: vec![ChannelStateSpace {
            a: vec![1.0], // 1 element but n_states=2 needs 4
            n_states: 2,
            b: vec![1.0, 1.0],
            c: vec![1.0, 1.0],
            d: 0.0,
            proportional: 0.0,
        }],
    };
    let err = state_space
        .discretize(1e-3, DiscretizationMethod::BackwardEuler)
        .expect_err("invalid state-space dimensions should fail");
    assert!(matches!(err, VecfitError::Dimension(_)));
}

// ============================================================
// Pole history
// ============================================================

#[test]
fn pole_history_tracking() {
    let axis = build_samples(80);

    // Disabled by default
    let model = Model::fit(
        complex(&axis),
        |s| 1.0 / (s + 3.0) + 0.1,
        Options::new().poles(2),
    )
    .expect("fit");
    assert!(model.pole_history().is_none());

    // Enabled
    let model = Model::fit(
        complex(&axis),
        |s| 1.0 / (s + 3.0) + 0.1,
        Options::new().poles(2).track_pole_history(true),
    )
    .expect("fit");
    let history = model.pole_history().expect("should have history");
    assert_eq!(history.len(), model.report().iterations);
    assert!(!history.is_empty());
}
