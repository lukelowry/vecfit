use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use vecfit::{Model, OutputRepresentation};

fn vecfit() -> Command {
    Command::new(env!("CARGO_BIN_EXE_vecfit"))
}

fn temp_json_path(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("vecfit-{name}-{}-{nanos}.json", std::process::id()))
}

#[test]
fn help_succeeds() {
    let output = vecfit().arg("--help").output().expect("run vecfit --help");
    assert!(output.status.success(), "status: {}", output.status);
}

#[test]
fn fit_help_succeeds() {
    let output = vecfit()
        .args(["fit", "--help"])
        .output()
        .expect("run vecfit fit --help");
    assert!(output.status.success(), "status: {}", output.status);
}

#[test]
fn version_contains_package_version() {
    let output = vecfit()
        .arg("--version")
        .output()
        .expect("run vecfit --version");
    assert!(output.status.success(), "status: {}", output.status);
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "stdout: {stdout}"
    );
}

#[test]
fn fit_matrix_csv_writes_loadable_json() {
    let output_path = temp_json_path("matrix-fit");
    let _ = std::fs::remove_file(&output_path);

    let output = vecfit()
        .args([
            "fit",
            "examples/data/matrix_admittance_2x2.csv",
            "--matrix",
            "2",
            "2",
            "--poles",
            "4",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .expect("run vecfit fit");

    assert!(
        output.status.success(),
        "status: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let model = Model::from_json_path(&output_path).expect("load output model");
    assert_eq!(model.shape().expect_matrix().expect("matrix shape"), (2, 2));

    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn fit_matrix_csv_can_export_modal_state_space() {
    let output_path = temp_json_path("matrix-fit-state-space");
    let _ = std::fs::remove_file(&output_path);

    let output = vecfit()
        .args([
            "fit",
            "examples/data/matrix_admittance_2x2.csv",
            "--matrix",
            "2",
            "2",
            "--poles",
            "4",
            "--state-space",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .expect("run vecfit fit");

    assert!(
        output.status.success(),
        "status: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let json_text = std::fs::read_to_string(&output_path).expect("read output model");
    let value: serde_json::Value = serde_json::from_str(&json_text).expect("json value");
    assert!(value.get("residues").is_none(), "json: {json_text}");
    assert!(value.get("C").is_some(), "json: {json_text}");
    assert!(value.get("B").is_some(), "json: {json_text}");

    let model = Model::from_json_path(&output_path).expect("load output model");
    assert_eq!(model.output(), OutputRepresentation::StateSpace);
    assert_eq!(model.shape().expect_matrix().expect("matrix shape"), (2, 2));

    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn fit_cli_terms_configures_d_and_e() {
    let output_path = temp_json_path("terms-none");
    let _ = std::fs::remove_file(&output_path);

    let output = vecfit()
        .args([
            "fit",
            "examples/data/matrix_admittance_2x2.csv",
            "--matrix",
            "2",
            "2",
            "--poles",
            "4",
            "--terms",
            "none",
            "--output",
        ])
        .arg(&output_path)
        .output()
        .expect("run vecfit fit");

    assert!(
        output.status.success(),
        "status: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let model = Model::from_json_path(&output_path).expect("load output model");
    assert!(
        model
            .constant_terms()
            .iter()
            .all(|value| value.norm() == 0.0)
    );
    assert!(
        model
            .proportional_terms()
            .iter()
            .all(|value| value.norm() == 0.0)
    );

    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn missing_input_is_nonzero() {
    let output = vecfit().arg("fit").output().expect("run vecfit fit");
    assert!(!output.status.success(), "status: {}", output.status);
}
