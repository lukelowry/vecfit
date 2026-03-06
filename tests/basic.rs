use approx::assert_relative_eq;
use num_complex::Complex64;
use vecfit::{
    ChannelStateSpace, CsvSamples, DiscretizationMethod, FitOptions, FlatResponse, IntoResponse,
    Layout, Model, ModelParts, ResponseSample, Shape, StateSpaceModel, VecfitError,
};

fn build_samples(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|k| Complex64::new(0.0, 0.5 + k as f64))
        .collect()
}

#[test]
fn scalar_fit_and_evaluate() {
    let sample_axis = build_samples(200);
    let model = Model::fit(
        &sample_axis,
        |sk| {
            1.0 / (sk + Complex64::new(3.0, 0.0))
                + 2.0 / (sk + Complex64::new(20.0, 0.0))
                + Complex64::new(0.5, 0.0)
        },
        FitOptions::new().poles(2),
    )
    .expect("fit should succeed");
    assert_eq!(model.shape(), &Shape::scalar());
    let y = model
        .evaluate_scalar(&sample_axis)
        .expect("scalar evaluation should succeed");
    assert_eq!(y.len(), sample_axis.len());
    assert!(model.abs_rmse() < 1.0);
}

#[test]
fn vector_shape_roundtrip() {
    let shaped = [1.0, 2.0, 3.0]
        .into_response()
        .expect("vector flatten should work");
    assert_eq!(shaped.shape.expect_vector().unwrap(), 3);
    let restored = ResponseSample::from(shaped)
        .into_vector()
        .expect("vector reconstruction should work");
    assert_eq!(restored.len(), 3);
}

#[test]
fn matrix_shape_roundtrip() {
    let flattened = vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        .into_response()
        .expect("matrix flatten should work");
    assert_eq!(flattened.shape.expect_matrix().unwrap(), (2, 2));
    let restored = ResponseSample::from(flattened)
        .into_matrix()
        .expect("matrix reconstruction should work");
    assert_eq!(restored[1][1], Complex64::new(4.0, 0.0));
}

#[test]
fn explicit_tensor_wrapper_works() {
    let flattened = FlatResponse::new(
        vec![Complex64::new(1.0, 0.0); 6],
        Shape::tensor([2, 3]).expect("shape should be valid"),
        Layout::RowMajor,
    )
    .expect("tensor wrapper should validate");
    assert_eq!(flattened.shape.channels(), 6);
}

#[test]
fn jw_hz_matrix_fit_is_shape_aware() {
    let freqs = (1..120).map(|k| k as f64).collect::<Vec<_>>();
    let model = Model::fit_hz(
        &freqs,
        |hz| {
            let w = 2.0 * std::f64::consts::PI * hz;
            [
                [1.0 / (1.0 + w), 0.5 / (2.0 + w)],
                [0.5 / (2.0 + w), 1.2 / (3.0 + w)],
            ]
        },
        FitOptions::new().poles(4),
    )
    .expect("matrix fit should succeed");
    assert_eq!(model.shape().expect_matrix().unwrap(), (2, 2));
    let eval = model
        .evaluate_matrix(
            &freqs
                .iter()
                .map(|hz| Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz))
                .collect::<Vec<_>>(),
        )
        .expect("matrix evaluation should work");
    assert_eq!(eval.len(), freqs.len());
    assert_eq!(eval[0].len(), 2);
    assert_eq!(eval[0][0].len(), 2);
}

#[test]
fn real_only_matrix_fit_supports_emt_exports() {
    fn response(s: Complex64) -> [[Complex64; 2]; 2] {
        [
            [
                Complex64::new(0.12, 0.0)
                    + Complex64::new(2.0, 0.0) / (s + Complex64::new(40.0, 0.0))
                    + Complex64::new(0.4, 0.0) / (s + Complex64::new(500.0, 0.0)),
                Complex64::new(-0.03, 0.0)
                    + Complex64::new(0.7, 0.0) / (s + Complex64::new(80.0, 0.0)),
            ],
            [
                Complex64::new(-0.03, 0.0)
                    + Complex64::new(0.7, 0.0) / (s + Complex64::new(80.0, 0.0)),
                Complex64::new(0.08, 0.0)
                    + Complex64::new(1.6, 0.0) / (s + Complex64::new(30.0, 0.0))
                    + Complex64::new(0.3, 0.0) / (s + Complex64::new(300.0, 0.0)),
            ],
        ]
    }

    let freqs = (0..160)
        .map(|idx| {
            let t = idx as f64 / 159.0;
            10f64.powf(t * 4.0)
        })
        .collect::<Vec<_>>();
    let options = FitOptions::new().poles(5).real_only(true);
    let model = Model::fit_hz(
        &freqs,
        |hz| response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz)),
        options,
    )
    .expect("real-only matrix fit should succeed");
    let sections = model
        .real_sections()
        .expect("real-only matrix fit should export sections");
    let discrete = model
        .state_space()
        .expect("state-space export should succeed")
        .discretize(1.0e-4, vecfit::DiscretizationMethod::Tustin)
        .expect("discretization should succeed");
    assert_eq!(sections.channels.len(), 4);
    assert_eq!(discrete.channels.len(), 4);
}

#[test]
fn underdetermined_fit_returns_error_instead_of_panicking() {
    let sample_axis = build_samples(3);
    let err = Model::fit(
        &sample_axis,
        |sk| 1.0 / (sk + Complex64::new(3.0, 0.0)) + 0.1,
        FitOptions::new().poles(3),
    )
    .expect_err("underdetermined fit should return a structured error");
    assert!(matches!(err, vecfit::VecfitError::InvalidInput(_)));
}

#[test]
fn complex_json_roundtrip() {
    let sample_axis = build_samples(80);
    let model = Model::fit(
        &sample_axis,
        |sk| {
            vec![
                1.0 / (sk + Complex64::new(3.0, 0.0)),
                2.0 / (sk + Complex64::new(6.0, 0.0)),
            ]
        },
        FitOptions::new().poles(3),
    )
    .expect("fit should succeed");
    let json = model.to_json().expect("complex JSON export should work");
    let loaded = Model::from_json(&json).expect("complex JSON import should work");
    assert_eq!(loaded.channels(), model.channels());
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
        shape: Shape::matrix(2, 2).expect("matrix shape should be valid"),
        layout: Layout::ColumnMajor,
        report: Default::default(),
    })
    .expect("model parts should validate");
    let json = model.to_json().expect("complex JSON export should work");
    let loaded = Model::from_json(&json).expect("complex JSON import should work");
    assert_eq!(loaded.shape(), model.shape());
    assert_eq!(loaded.layout(), model.layout());
}

#[test]
fn real_kernel_json_roundtrip_requires_real_model() {
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
    .expect("model parts should validate");
    let json = model
        .to_real_json(Some("test".to_string()))
        .expect("real-kernel JSON export should work");
    let loaded = Model::from_real_json(&json).expect("real-kernel JSON import should work");
    assert_eq!(loaded.pole_count(), 2);
}

#[test]
fn real_kernel_json_roundtrip_preserves_shape_and_layout() {
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
        shape: Shape::matrix(2, 2).expect("matrix shape should be valid"),
        layout: Layout::ColumnMajor,
        report: Default::default(),
    })
    .expect("model parts should validate");
    let json = model
        .to_real_json(Some("matrix".to_string()))
        .expect("real-kernel JSON export should work");
    let loaded = Model::from_real_json(&json).expect("real-kernel JSON import should work");
    assert_eq!(loaded.shape(), model.shape());
    assert_eq!(loaded.layout(), model.layout());
}

#[test]
fn fit_rejects_invalid_option_weights() {
    let sample_axis = build_samples(20);
    let err = Model::fit(
        &sample_axis,
        |sk| 1.0 / (sk + Complex64::new(3.0, 0.0)) + 0.1,
        FitOptions::new().poles(2).weights(vec![1.0; 3]),
    )
    .expect_err("short option weight vector should be rejected");
    assert!(matches!(err, VecfitError::Dimension(_)));

    let err = Model::fit(
        &sample_axis,
        |sk| 1.0 / (sk + Complex64::new(3.0, 0.0)) + 0.1,
        FitOptions::new()
            .poles(2)
            .weights(vec![-1.0; sample_axis.len()]),
    )
    .expect_err("negative option weights should be rejected");
    assert!(matches!(err, VecfitError::InvalidInput(_)));
}

#[test]
fn fit_rejects_layout_changes_between_samples() {
    let samples = [0usize, 1usize];
    let err = Model::fit_mapped(
        &samples,
        |idx| Complex64::new(0.0, *idx as f64 + 1.0),
        |idx| {
            FlatResponse::new(
                vec![
                    Complex64::new(*idx as f64, 0.0),
                    Complex64::new(*idx as f64 + 1.0, 0.0),
                ],
                Shape::vector(2).expect("vector shape should be valid"),
                if *idx % 2 == 0 {
                    Layout::RowMajor
                } else {
                    Layout::ColumnMajor
                },
            )
            .expect("response should be valid")
        },
        FitOptions::new().poles(1),
    )
    .expect_err("mixed response layouts should be rejected");
    assert!(matches!(err, VecfitError::Shape(_)));
}

#[test]
fn invalid_model_parts_are_rejected() {
    let model = Model::from_parts(ModelParts {
        poles: vec![Complex64::new(-1.0, 0.0)],
        residues: vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
        channels: 2,
        constant_terms: vec![Complex64::new(0.1, 0.0)],
        proportional_terms: vec![Complex64::new(0.0, 0.0); 2],
        shape: Shape::vector(2).expect("vector shape should be valid"),
        layout: Layout::RowMajor,
        report: Default::default(),
    })
    .expect_err("invalid model data should fail validation");
    let err = model.to_string();
    assert!(err.contains("constant term count"));
}

#[test]
fn invalid_state_space_returns_error_instead_of_panicking() {
    let state_space = StateSpaceModel {
        shape: Shape::scalar(),
        layout: Layout::RowMajor,
        channels: vec![ChannelStateSpace {
            a: vec![1.0],
            n_states: 2,
            b: vec![1.0, 1.0],
            c: vec![1.0, 1.0],
            d: 0.0,
            proportional: 0.0,
        }],
    };
    let err = state_space
        .discretize(1.0e-3, DiscretizationMethod::BackwardEuler)
        .expect_err("invalid state-space data should return an error");
    assert!(matches!(err, VecfitError::Dimension(_)));
}

#[test]
fn csv_parser_builds_jw_samples() {
    let csv = "freq_Hz,|Y1|,ang_Y1\n1,10,45\n10,5,-90\n";
    let parsed = CsvSamples::from_csv(csv).expect("csv parse should work");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed.frequency_hz().len(), parsed.len());
    assert_eq!(parsed.channels(), 1);
    assert_relative_eq!(
        parsed.axis()[0].im,
        2.0 * std::f64::consts::PI,
        epsilon = 1e-12
    );
    assert_eq!(
        parsed
            .scalars()
            .expect("scalar reconstruction should work")
            .len(),
        parsed.len()
    );
}

#[test]
fn emt_exports_work_for_real_model() {
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
    .expect("model parts should validate");
    let sections = model
        .real_sections()
        .expect("real section export should work");
    assert_eq!(sections.channels.len(), 1);
    let state_space = model.state_space().expect("state-space export should work");
    let discrete = state_space
        .discretize(1.0e-4, vecfit::DiscretizationMethod::BackwardEuler)
        .expect("discretization should work");
    assert_eq!(discrete.channels.len(), 1);
}

#[test]
fn csv_parser_rejects_incomplete_magnitude_phase_pairs() {
    let csv = "freq_Hz,|Y1|,ang_Y1,|Y2|\n1,10,45,3\n";
    let err = CsvSamples::from_csv(csv)
        .expect_err("csv parse should reject incomplete magnitude/phase pairs");
    assert!(matches!(err, vecfit::VecfitError::Csv(_)));
}

#[test]
fn column_major_fit_evaluate_json_roundtrip() {
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
    .expect("model parts should validate");
    let axis = build_samples(50);
    let flat = model.evaluate_flat(&axis).expect("evaluate should work");
    assert_eq!(flat.samples, 50);
    let json = model.to_json().expect("JSON export should work");
    let loaded = Model::from_json(&json).expect("JSON import should work");
    assert_eq!(loaded.layout(), Layout::ColumnMajor);
    assert_eq!(loaded.channels(), 2);
}

#[test]
fn csv_parser_handles_nan_frequency() {
    let csv = "freq_Hz,|Y1|,ang_Y1\nNaN,10,45\n";
    let result = CsvSamples::from_csv(csv);
    if let Ok(parsed) = result {
        assert!(parsed.frequency_hz()[0].is_nan());
    }
}

#[test]
fn csv_parser_handles_empty_data() {
    let csv = "freq_Hz,|Y1|,ang_Y1\n";
    let result = CsvSamples::from_csv(csv);
    // Either errors or returns empty - both are acceptable
    if let Ok(parsed) = &result {
        assert_eq!(parsed.len(), 0);
    }
}

#[test]
fn from_json_rejects_missing_poles() {
    let json = r#"{"residues":[],"d":[],"e":[],"rmse":0,"iters":0}"#;
    assert!(Model::from_json(json).is_err());
}

#[test]
fn fit_rejects_zero_poles() {
    let axis = build_samples(20);
    let err = Model::fit(
        &axis,
        |sk| 1.0 / (sk + Complex64::new(3.0, 0.0)),
        FitOptions::new().poles(0),
    )
    .expect_err("zero poles should be rejected");
    assert!(matches!(err, VecfitError::InvalidInput(_)));
}
