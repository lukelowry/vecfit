use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use vecfit::{Csv, Options, OutputRepresentation, WeightStrategy};

type CliResult<T> = Result<T, String>;

fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("--help" | "-h") => {
            print_top_level_help();
            Ok(())
        }
        Some("--version" | "-V") => {
            println!("vecfit {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("fit") => run_fit(args),
        Some(flag) if flag.starts_with('-') => Err(format!("unknown flag: {flag}")),
        Some(command) => Err(format!("unknown command: {command}")),
        None => Err("missing command; try `vecfit --help`".to_string()),
    }
}

#[derive(Debug)]
struct FitConfig {
    input: PathBuf,
    matrix: Option<(usize, usize)>,
    output: Option<PathBuf>,
    options: Options,
}

fn run_fit(args: impl IntoIterator<Item = String>) -> CliResult<()> {
    let Some(config) = parse_fit_args(args)? else {
        return Ok(());
    };

    let samples = Csv::from_path(&config.input)
        .map_err(|err| format!("failed to load {}: {err}", config.input.display()))?;
    let samples = if let Some((rows, cols)) = config.matrix {
        samples
            .matrix(rows, cols)
            .map_err(|err| format!("invalid matrix dimensions: {err}"))?
    } else {
        samples
    };

    let model = samples
        .fit(config.options)
        .map_err(|err| format!("fit failed: {err}"))?;
    let json = model
        .to_json()
        .map_err(|err| format!("failed to serialize model: {err}"))?;

    if let Some(output) = config.output {
        fs::write(&output, json)
            .map_err(|err| format!("failed to write {}: {err}", output.display()))?;
    } else {
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(json.as_bytes())
            .and_then(|_| stdout.write_all(b"\n"))
            .map_err(|err| format!("failed to write stdout: {err}"))?;
    }

    Ok(())
}

fn parse_fit_args(args: impl IntoIterator<Item = String>) -> CliResult<Option<FitConfig>> {
    let mut args = args.into_iter();
    let mut input = None;
    let mut matrix = None;
    let mut output = None;
    let mut options = Options::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_fit_help();
                return Ok(None);
            }
            "--poles" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--poles requires a value".to_string())?;
                options.poles = parse_positive_usize(&value, "pole count")?;
            }
            "--matrix" => {
                if matrix.is_some() {
                    return Err("--matrix may only be provided once".to_string());
                }
                let rows = args
                    .next()
                    .ok_or_else(|| "--matrix requires ROWS COLS".to_string())?;
                let cols = args
                    .next()
                    .ok_or_else(|| "--matrix requires ROWS COLS".to_string())?;
                matrix = Some((
                    parse_positive_usize(&rows, "matrix row count")?,
                    parse_positive_usize(&cols, "matrix column count")?,
                ));
            }
            "--real-only" => options.real_only = true,
            "--weighted" => options.weight_strategy = WeightStrategy::InverseMagnitude,
            "--terms" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--terms requires a value".to_string())?;
                let (fit_constant, fit_proportional) = parse_fit_terms(&value)?;
                options.fit_constant = fit_constant;
                options.fit_proportional = fit_proportional;
            }
            "--state-space" => options.output = OutputRepresentation::StateSpace,
            "--output" => {
                if output.is_some() {
                    return Err("--output may only be provided once".to_string());
                }
                let path = args
                    .next()
                    .ok_or_else(|| "--output requires a value".to_string())?;
                output = Some(PathBuf::from(path));
            }
            flag if flag.starts_with('-') => return Err(format!("unknown flag: {flag}")),
            positional => {
                if input.is_some() {
                    return Err(format!("unexpected argument: {positional}"));
                }
                input = Some(PathBuf::from(positional));
            }
        }
    }

    let input = input.ok_or_else(|| "missing input file".to_string())?;
    Ok(Some(FitConfig {
        input,
        matrix,
        output,
        options,
    }))
}

fn parse_positive_usize(value: &str, name: &str) -> CliResult<usize> {
    match value.parse::<usize>() {
        Ok(parsed) if parsed > 0 => Ok(parsed),
        _ => Err(format!("invalid {name}: {value}")),
    }
}

fn parse_fit_terms(value: &str) -> CliResult<(bool, bool)> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "none" | "0" => return Ok((false, false)),
        "d" => return Ok((true, false)),
        "e" => return Ok((false, true)),
        "de" | "ed" => return Ok((true, true)),
        _ => {}
    }

    let mut fit_d = false;
    let mut fit_e = false;
    let mut saw_term = false;
    for term in normalized.split(',').map(str::trim) {
        match term {
            "d" => {
                fit_d = true;
                saw_term = true;
            }
            "e" => {
                fit_e = true;
                saw_term = true;
            }
            _ => {
                return Err(format!(
                    "invalid --terms value: {value}; use one of none, d, e, de, d,e"
                ));
            }
        }
    }
    if saw_term {
        Ok((fit_d, fit_e))
    } else {
        Err(format!(
            "invalid --terms value: {value}; use one of none, d, e, de, d,e"
        ))
    }
}

fn print_top_level_help() {
    println!(
        "vecfit {version}

Usage:
  vecfit fit <input.csv> [--poles N] [--matrix ROWS COLS] [--terms TERMS] [--real-only] [--weighted] [--state-space] [--output model.json]
  vecfit --help
  vecfit --version",
        version = env!("CARGO_PKG_VERSION")
    );
}

fn print_fit_help() {
    println!(
        "Usage:
  vecfit fit <input.csv> [--poles N] [--matrix ROWS COLS] [--terms TERMS] [--real-only] [--weighted] [--state-space] [--output model.json]

Options:
  --poles N               Number of poles to fit (default: 6)
  --matrix ROWS COLS      Interpret CSV channels as a matrix
  --terms TERMS           Fit polynomial terms: none, d, e, de, or d,e (default: d)
  --real-only             Constrain fitted poles to the real axis
  --weighted              Use inverse-magnitude sample weighting
  --state-space           Export a complex modal state-space realization
  --output model.json     Write complex model JSON to a file instead of stdout"
    );
}
