#![allow(dead_code)]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use num_complex::Complex64;
use plotters::coord::ranged1d::ValueFormatter;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

const CANVAS_BACKGROUND: RGBColor = RGBColor(244, 246, 248);
const PANEL_BACKGROUND: RGBColor = RGBColor(255, 255, 255);
const GRID_COLOR: RGBColor = RGBColor(205, 211, 217);
const TEXT_COLOR: RGBColor = RGBColor(47, 58, 70);
const TABLEAU_BLUE: RGBColor = RGBColor(78, 121, 167);
const TABLEAU_ORANGE: RGBColor = RGBColor(242, 142, 43);
const TABLEAU_RED: RGBColor = RGBColor(225, 87, 89);
const TABLEAU_TEAL: RGBColor = RGBColor(118, 183, 178);
const TABLEAU_GREEN: RGBColor = RGBColor(89, 161, 79);
const TABLEAU_GOLD: RGBColor = RGBColor(237, 201, 72);

#[derive(Clone, Copy)]
enum PlotSeriesStyle {
    ReferenceMarkers,
    FittedLine,
    ResponseLine,
    ErrorLine,
}

struct PlotSeries {
    label: String,
    values: Vec<f64>,
    color: RGBColor,
    style: PlotSeriesStyle,
}

struct ReportPanel {
    title: String,
    y_desc: &'static str,
    series: Vec<PlotSeries>,
}

pub struct ComparisonSeries<'a> {
    pub label: &'a str,
    pub reference: &'a [Complex64],
    pub fitted: &'a [Complex64],
}

pub struct ResponseSeries<'a> {
    pub label: &'a str,
    pub values: &'a [Complex64],
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn example_data_path(name: &str) -> PathBuf {
    project_root().join("examples").join("data").join(name)
}

pub fn example_output_dir() -> Result<PathBuf, Box<dyn Error>> {
    let output_dir = project_root().join("examples").join("out");
    fs::create_dir_all(&output_dir)?;
    Ok(output_dir)
}

pub fn example_output_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    Ok(example_output_dir()?.join(name))
}

pub fn write_summary_markdown(path: &Path, body: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, body)?;
    Ok(())
}

pub fn logspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
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

fn magnitude(values: &[Complex64]) -> Vec<f64> {
    values.iter().map(|value| value.norm()).collect()
}

fn phase_deg(values: &[Complex64]) -> Vec<f64> {
    let mut phases: Vec<f64> = values
        .iter()
        .map(|value| value.arg().to_degrees())
        .collect();
    unwrap_phase_deg(&mut phases);
    phases
}

fn unwrap_phase_deg(phases: &mut [f64]) {
    for i in 1..phases.len() {
        let mut delta = phases[i] - phases[i - 1];
        while delta > 180.0 {
            delta -= 360.0;
        }
        while delta < -180.0 {
            delta += 360.0;
        }
        phases[i] = phases[i - 1] + delta;
    }
}

fn relative_error(target: &[Complex64], fitted: &[Complex64]) -> Vec<f64> {
    target
        .iter()
        .zip(fitted.iter())
        .map(|(lhs, rhs)| (*lhs - *rhs).norm() / lhs.norm().max(1e-14))
        .collect()
}

pub fn extract_channel(samples: &[Vec<Complex64>], channel_idx: usize) -> Vec<Complex64> {
    samples.iter().map(|row| row[channel_idx]).collect()
}

pub fn extract_matrix_entry(
    samples: &[Vec<Vec<Complex64>>],
    row_idx: usize,
    col_idx: usize,
) -> Vec<Complex64> {
    samples
        .iter()
        .map(|matrix| matrix[row_idx][col_idx])
        .collect()
}

pub fn draw_comparison_report(
    filename: &Path,
    x_values: &[f64],
    x_desc: &str,
    report_name: &str,
    traces: &[ComparisonSeries<'_>],
) -> Result<(), Box<dyn Error>> {
    if traces.is_empty() {
        return Err("comparison plot requires at least one trace".into());
    }

    let mut magnitude_series = Vec::with_capacity(traces.len() * 2);
    let mut phase_series = Vec::with_capacity(traces.len() * 2);
    let mut error_series = Vec::with_capacity(traces.len());

    for (trace_idx, trace) in traces.iter().enumerate() {
        validate_trace_lengths(trace.reference, trace.fitted, x_values.len(), trace.label)?;
        let color = palette_color(trace_idx);
        magnitude_series.push(PlotSeries {
            label: format!("Ref {}", trace.label),
            values: magnitude(trace.reference),
            color,
            style: PlotSeriesStyle::ReferenceMarkers,
        });
        magnitude_series.push(PlotSeries {
            label: format!("Fit {}", trace.label),
            values: magnitude(trace.fitted),
            color,
            style: PlotSeriesStyle::FittedLine,
        });
        phase_series.push(PlotSeries {
            label: format!("Ref {}", trace.label),
            values: phase_deg(trace.reference),
            color,
            style: PlotSeriesStyle::ReferenceMarkers,
        });
        phase_series.push(PlotSeries {
            label: format!("Fit {}", trace.label),
            values: phase_deg(trace.fitted),
            color,
            style: PlotSeriesStyle::FittedLine,
        });
        error_series.push(PlotSeries {
            label: trace.label.to_string(),
            values: relative_error(trace.reference, trace.fitted),
            color,
            style: PlotSeriesStyle::ErrorLine,
        });
    }

    draw_panels(
        filename,
        x_values,
        x_desc,
        &[
            ReportPanel {
                title: format!("{report_name} Magnitude"),
                y_desc: "Magnitude",
                series: magnitude_series,
            },
            ReportPanel {
                title: format!("{report_name} Phase"),
                y_desc: "Phase (deg)",
                series: phase_series,
            },
            ReportPanel {
                title: format!("{report_name} Relative Error"),
                y_desc: "Relative error",
                series: error_series,
            },
        ],
    )
}

pub fn draw_response_report(
    filename: &Path,
    x_values: &[f64],
    x_desc: &str,
    report_name: &str,
    traces: &[ResponseSeries<'_>],
) -> Result<(), Box<dyn Error>> {
    if traces.is_empty() {
        return Err("response plot requires at least one trace".into());
    }

    let mut magnitude_series = Vec::with_capacity(traces.len());
    let mut phase_series = Vec::with_capacity(traces.len());

    for (trace_idx, trace) in traces.iter().enumerate() {
        validate_series_length(trace.values, x_values.len(), trace.label)?;
        let color = palette_color(trace_idx);
        magnitude_series.push(PlotSeries {
            label: trace.label.to_string(),
            values: magnitude(trace.values),
            color,
            style: PlotSeriesStyle::ResponseLine,
        });
        phase_series.push(PlotSeries {
            label: trace.label.to_string(),
            values: phase_deg(trace.values),
            color,
            style: PlotSeriesStyle::ResponseLine,
        });
    }

    draw_panels(
        filename,
        x_values,
        x_desc,
        &[
            ReportPanel {
                title: format!("{report_name} Magnitude"),
                y_desc: "Magnitude",
                series: magnitude_series,
            },
            ReportPanel {
                title: format!("{report_name} Phase"),
                y_desc: "Phase (deg)",
                series: phase_series,
            },
        ],
    )
}

fn palette_color(index: usize) -> RGBColor {
    const PALETTE: [RGBColor; 6] = [
        TABLEAU_BLUE,
        TABLEAU_ORANGE,
        TABLEAU_RED,
        TABLEAU_TEAL,
        TABLEAU_GREEN,
        TABLEAU_GOLD,
    ];
    PALETTE[index % PALETTE.len()]
}

fn validate_series_length(
    values: &[Complex64],
    x_len: usize,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if values.len() != x_len {
        return Err(format!(
            "trace '{label}' has {} points but x-axis has {x_len}",
            values.len()
        )
        .into());
    }
    Ok(())
}

fn validate_trace_lengths(
    reference: &[Complex64],
    fitted: &[Complex64],
    x_len: usize,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    validate_series_length(reference, x_len, label)?;
    validate_series_length(fitted, x_len, label)?;
    Ok(())
}

/// Choose the legend corner with the least data density.
fn choose_legend_position(x_values: &[f64], series: &[PlotSeries]) -> SeriesLabelPosition {
    if series.is_empty() || x_values.len() < 2 {
        return SeriesLabelPosition::UpperLeft;
    }
    let x_mid = (x_values[0] + x_values[x_values.len() - 1]) / 2.0;
    let all_y: Vec<f64> = series
        .iter()
        .flat_map(|s| s.values.iter().copied())
        .collect();
    let y_mid = {
        let sum: f64 = all_y.iter().sum();
        sum / all_y.len().max(1) as f64
    };

    let (mut ul, mut ur, mut ll, mut lr) = (0u32, 0u32, 0u32, 0u32);
    for s in series {
        for (x, y) in x_values.iter().zip(s.values.iter()) {
            match (*x < x_mid, *y > y_mid) {
                (true, true) => ul += 1,
                (false, true) => ur += 1,
                (true, false) => ll += 1,
                (false, false) => lr += 1,
            }
        }
    }
    let min = ul.min(ur).min(ll).min(lr);
    if min == ur {
        SeriesLabelPosition::UpperRight
    } else if min == ll {
        SeriesLabelPosition::LowerLeft
    } else if min == lr {
        SeriesLabelPosition::LowerRight
    } else {
        SeriesLabelPosition::UpperLeft
    }
}

fn draw_panel_mesh<'a, DB, XT>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<XT, RangedCoordf64>>,
    x_desc: &str,
    panel: &ReportPanel,
) -> Result<(), Box<dyn Error>>
where
    DB: DrawingBackend + 'a,
    DB::ErrorType: 'static,
    XT: Ranged + ValueFormatter<<XT as Ranged>::ValueType>,
{
    chart
        .configure_mesh()
        .x_labels(8)
        .y_labels(8)
        .x_desc(x_desc)
        .y_desc(panel.y_desc)
        .axis_desc_style(("sans-serif", 22).into_font().color(&TEXT_COLOR))
        .label_style(("sans-serif", 16).into_font().color(&TEXT_COLOR))
        .bold_line_style(GRID_COLOR.mix(0.45))
        .light_line_style(GRID_COLOR.mix(0.22))
        .draw()?;
    Ok(())
}

fn draw_panel_series<'a, DB, XT>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<XT, RangedCoordf64>>,
    x_values: &[f64],
    panel: &ReportPanel,
) -> Result<(), Box<dyn Error>>
where
    DB: DrawingBackend + 'a,
    DB::ErrorType: 'static,
    XT: Ranged<ValueType = f64>,
{
    const SAMPLE_RADIUS: i32 = 7;
    let marker_stride = (x_values.len() / 28).max(1);

    for series in &panel.series {
        match series.style {
            PlotSeriesStyle::ReferenceMarkers => {
                chart
                    .draw_series(
                        x_values
                            .iter()
                            .zip(series.values.iter())
                            .enumerate()
                            .filter(|(idx, _)| idx % marker_stride == 0)
                            .map(|(_, (&xv, &yv))| {
                                Circle::new(
                                    (xv, yv),
                                    SAMPLE_RADIUS,
                                    series.color.mix(0.72).filled(),
                                )
                            }),
                    )?
                    .label(series.label.as_str())
                    .legend({
                        let color = series.color;
                        move |(x, y)| {
                            Circle::new((x + 10, y), SAMPLE_RADIUS, color.mix(0.72).filled())
                        }
                    });
            }
            PlotSeriesStyle::ResponseLine => {
                chart
                    .draw_series(LineSeries::new(
                        x_values
                            .iter()
                            .zip(series.values.iter())
                            .map(|(&xv, &yv)| (xv, yv)),
                        series.color.stroke_width(3),
                    ))?
                    .label(series.label.as_str())
                    .legend({
                        let color = series.color;
                        move |(x, y)| {
                            PathElement::new(vec![(x, y), (x + 22, y)], color.stroke_width(3))
                        }
                    });
                chart.draw_series(
                    x_values
                        .iter()
                        .zip(series.values.iter())
                        .enumerate()
                        .filter(|(idx, _)| idx % marker_stride == 0)
                        .map(|(_, (&xv, &yv))| {
                            Circle::new((xv, yv), SAMPLE_RADIUS, series.color.mix(0.72).filled())
                        }),
                )?;
            }
            PlotSeriesStyle::FittedLine | PlotSeriesStyle::ErrorLine => {
                chart
                    .draw_series(LineSeries::new(
                        x_values
                            .iter()
                            .zip(series.values.iter())
                            .map(|(&xv, &yv)| (xv, yv)),
                        series.color.stroke_width(3),
                    ))?
                    .label(series.label.as_str())
                    .legend({
                        let color = series.color;
                        move |(x, y)| {
                            PathElement::new(vec![(x, y), (x + 22, y)], color.stroke_width(3))
                        }
                    });
            }
        }
    }

    chart
        .configure_series_labels()
        .position(choose_legend_position(x_values, &panel.series))
        .background_style(WHITE.mix(0.95))
        .label_font(("sans-serif", 15).into_font().color(&TEXT_COLOR))
        .border_style(TEXT_COLOR.mix(0.2))
        .draw()?;
    Ok(())
}

fn draw_panels(
    filename: &Path,
    x_values: &[f64],
    x_desc: &str,
    panels: &[ReportPanel],
) -> Result<(), Box<dyn Error>> {
    if x_values.len() < 2 {
        return Err("plot x-axis requires at least two samples".into());
    }
    if panels.is_empty() {
        return Err("plot requires at least one panel".into());
    }
    let root = BitMapBackend::new(filename, (1600, 360 * panels.len() as u32)).into_drawing_area();
    root.fill(&CANVAS_BACKGROUND)?;
    let areas = root.split_evenly((panels.len(), 1));
    let x0 = *x_values.first().ok_or("missing x data")?;
    let x1 = *x_values.last().ok_or("missing x data")?;
    let use_log_x = x_values.iter().all(|value| *value > 0.0);

    for (area, panel) in areas.into_iter().zip(panels.iter()) {
        area.fill(&PANEL_BACKGROUND)?;
        let (mut y_min, mut y_max) = panel
            .series
            .iter()
            .flat_map(|series| series.values.iter().copied())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
                (lo.min(value), hi.max(value))
            });
        if !y_min.is_finite() || !y_max.is_finite() {
            y_min = -1.0;
            y_max = 1.0;
        }
        if (y_max - y_min).abs() < 1e-12 {
            let pad = if y_max.abs() < 1e-9 {
                1.0
            } else {
                y_max.abs() * 0.1
            };
            y_min -= pad;
            y_max += pad;
        } else {
            let pad = (y_max - y_min) * 0.08;
            y_min -= pad;
            y_max += pad;
        }

        if use_log_x {
            let mut chart = ChartBuilder::on(&area)
                .margin(20)
                .caption(
                    panel.title.as_str(),
                    ("sans-serif", 26).into_font().color(&TEXT_COLOR),
                )
                .x_label_area_size(44)
                .y_label_area_size(62)
                .build_cartesian_2d((x0..x1).log_scale(), y_min..y_max)?;
            draw_panel_mesh(&mut chart, x_desc, panel)?;
            draw_panel_series(&mut chart, x_values, panel)?;
        } else {
            let mut chart = ChartBuilder::on(&area)
                .margin(20)
                .caption(
                    panel.title.as_str(),
                    ("sans-serif", 26).into_font().color(&TEXT_COLOR),
                )
                .x_label_area_size(44)
                .y_label_area_size(62)
                .build_cartesian_2d(x0..x1, y_min..y_max)?;
            draw_panel_mesh(&mut chart, x_desc, panel)?;
            draw_panel_series(&mut chart, x_values, panel)?;
        }
    }

    root.present()?;
    Ok(())
}

