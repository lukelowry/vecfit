use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use num_complex::Complex64;
use vecfit::{CsvSamples, FitOptions, Model};

fn scalar_response(s: Complex64) -> Complex64 {
    Complex64::new(0.03, 0.0)
        + Complex64::new(3.2, 0.0) / (s + Complex64::new(25.0, 0.0))
        + Complex64::new(1.0, 0.0) / (s + Complex64::new(250.0, 0.0))
}

fn vector_response(s: Complex64, channels: usize) -> Vec<Complex64> {
    (0..channels)
        .map(|idx| {
            let base = 15.0 + 12.0 * idx as f64;
            let slow = 0.03 + 0.005 * idx as f64;
            let fast = 0.4 + 0.03 * idx as f64;
            Complex64::new(slow, 0.0)
                + Complex64::new(1.6 + 0.1 * idx as f64, 0.0) / (s + Complex64::new(base, 0.0))
                + Complex64::new(fast, 0.0) / (s + Complex64::new(10.0 * base, 0.0))
        })
        .collect()
}

fn matrix_response(s: Complex64, n: usize) -> Vec<Vec<Complex64>> {
    (0..n)
        .map(|row| {
            (0..n)
                .map(|col| {
                    let base = 20.0 + 25.0 * (row + col) as f64;
                    let dc = if row == col {
                        0.08
                    } else {
                        -0.01 / (1.0 + (row + col) as f64)
                    };
                    Complex64::new(dc, 0.0)
                        + Complex64::new(1.2 / (1.0 + row as f64 + col as f64), 0.0)
                            / (s + Complex64::new(base, 0.0))
                })
                .collect()
        })
        .collect()
}

fn logspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let a = start.log10();
            let b = stop.log10();
            (0..n)
                .map(|idx| {
                    let t = idx as f64 / (n as f64 - 1.0);
                    10f64.powf(a + t * (b - a))
                })
                .collect()
        }
    }
}

fn bench_fit_scalar_samples(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_scalar_samples");
    for &n_samples in &[200usize, 1_000, 3_000] {
        let freqs = logspace(1.0, 10_000.0, n_samples);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &freqs,
            |b, freqs| {
                b.iter(|| {
                    let model = Model::fit_hz(
                        black_box(freqs),
                        |hz| scalar_response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz)),
                        black_box(FitOptions::new().poles(4)),
                    )
                    .expect("scalar fit should succeed");
                    black_box(model);
                });
            },
        );
    }
    group.finish();
}

fn bench_fit_vector_channels(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_vector_channels");
    let freqs = logspace(1.0, 10_000.0, 600);
    for &channels in &[1usize, 4, 16] {
        group.throughput(Throughput::Elements((freqs.len() * channels) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(channels),
            &channels,
            |b, &channels| {
                b.iter(|| {
                    let model = Model::fit_hz(
                        black_box(&freqs),
                        |hz| {
                            vector_response(
                                Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz),
                                channels,
                            )
                        },
                        black_box(FitOptions::new().poles(6)),
                    )
                    .expect("vector fit should succeed");
                    black_box(model);
                });
            },
        );
    }
    group.finish();
}

fn bench_fit_matrix_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_matrix_sizes");
    let freqs = logspace(1.0, 10_000.0, 400);
    for &size in &[2usize, 3, 4] {
        group.throughput(Throughput::Elements((freqs.len() * size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            &size,
            |b, &size| {
                b.iter(|| {
                    let model = Model::fit_hz(
                        black_box(&freqs),
                        |hz| {
                            matrix_response(
                                Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz),
                                size,
                            )
                        },
                        black_box(FitOptions::new().poles(6)),
                    )
                    .expect("matrix fit should succeed");
                    black_box(model);
                });
            },
        );
    }
    group.finish();
}

fn bench_evaluate_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_flat");
    let freqs = logspace(1.0, 10_000.0, 1_200);
    let model = Model::fit_hz(
        &freqs,
        |hz| vector_response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz), 8),
        FitOptions::new().poles(8),
    )
    .expect("evaluation benchmark fit should succeed");
    let s = freqs
        .iter()
        .map(|hz| Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz))
        .collect::<Vec<_>>();
    for &n_samples in &[300usize, 1_200, 4_800] {
        let dense = logspace(1.0, 10_000.0, n_samples)
            .iter()
            .map(|hz| Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz))
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements((n_samples * model.channels()) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &dense,
            |b, dense| {
                b.iter(|| {
                    let values = model
                        .evaluate_flat(black_box(dense))
                        .expect("evaluation should succeed");
                    black_box(values);
                });
            },
        );
    }
    black_box(s);
    group.finish();
}

fn bench_parse_csv(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_csv");
    let scalar_csv = include_str!("../examples/data/scalar_transfer.csv");
    let matrix_csv = include_str!("../examples/data/matrix_admittance_2x2.csv");
    for (label, text) in [("scalar_csv", scalar_csv), ("matrix_csv", matrix_csv)] {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &text, |b, text| {
            b.iter(|| {
                let parsed =
                    CsvSamples::from_csv(black_box(text)).expect("csv parse should succeed");
                black_box(parsed);
            });
        });
    }
    group.finish();
}

fn bench_fit_pole_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_pole_count");
    let freqs = logspace(1.0, 10_000.0, 600);
    for &poles in &[4usize, 8, 16, 32] {
        group.throughput(Throughput::Elements(freqs.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(poles), &poles, |b, &poles| {
            b.iter(|| {
                let model = Model::fit_hz(
                    black_box(&freqs),
                    |hz| scalar_response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz)),
                    black_box(FitOptions::new().poles(poles)),
                )
                .expect("fit should succeed");
                black_box(model);
            });
        });
    }
    group.finish();
}

fn bench_json_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_roundtrip");
    let freqs = logspace(1.0, 10_000.0, 400);
    let model = Model::fit_hz(
        &freqs,
        |hz| vector_response(Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz), 4),
        FitOptions::new().poles(6),
    )
    .expect("fit should succeed");
    let json = model.to_json().expect("export should work");
    group.throughput(Throughput::Bytes(json.len() as u64));
    group.bench_function("serialize", |b| {
        b.iter(|| black_box(model.to_json().unwrap()));
    });
    group.bench_function("deserialize", |b| {
        b.iter(|| black_box(Model::from_json(black_box(&json)).unwrap()));
    });
    group.finish();
}

fn bench_discretize(c: &mut Criterion) {
    use vecfit::DiscretizationMethod;
    let mut group = c.benchmark_group("discretize");
    let freqs = logspace(1.0, 10_000.0, 400);
    let model = Model::fit_hz(
        &freqs,
        |hz| {
            let s = Complex64::new(0.0, 2.0 * std::f64::consts::PI * hz);
            let h11 = Complex64::new(0.08, 0.0)
                + Complex64::new(2.0, 0.0) / (s + Complex64::new(40.0, 0.0))
                + Complex64::new(0.4, 0.0) / (s + Complex64::new(500.0, 0.0));
            let h12 = Complex64::new(-0.03, 0.0)
                + Complex64::new(0.7, 0.0) / (s + Complex64::new(80.0, 0.0));
            let h22 = Complex64::new(0.06, 0.0)
                + Complex64::new(1.6, 0.0) / (s + Complex64::new(30.0, 0.0))
                + Complex64::new(0.3, 0.0) / (s + Complex64::new(300.0, 0.0));
            [[h11, h12], [h12, h22]]
        },
        FitOptions::new().poles(6).real_only(true),
    )
    .expect("fit should succeed");
    let ss = model.state_space().expect("state-space should work");
    for method in [
        DiscretizationMethod::BackwardEuler,
        DiscretizationMethod::Tustin,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{method:?}")),
            &method,
            |b, method| {
                b.iter(|| black_box(ss.discretize(1e-4, *method).unwrap()));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fit_scalar_samples,
    bench_fit_vector_channels,
    bench_fit_matrix_sizes,
    bench_fit_pole_count,
    bench_evaluate_flat,
    bench_parse_csv,
    bench_json_roundtrip,
    bench_discretize,
);
criterion_main!(benches);
