import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2, linregress, t


APP_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = APP_DIR / "Experimentos"
AUX_DIR = APP_DIR / "arquivos_auxiliares"
SENSOR_MAP_FILE = AUX_DIR / "mapeamento_sensor_output.csv"
EXPERIMENT_MAP_FILE = AUX_DIR / "mapeamento_experimentos_parametros.csv"
UNCERTAINTY_FILE = AUX_DIR / "incertezas_instrumentais.csv"
PRESSURE_CALIBRATION_FILE = AUX_DIR / "relacao_corrente_pressao.csv"

# criando classes decoradas para representar os dados de forma estruturada

@dataclass
class DomainInterval:
    experiment_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    points: int


@dataclass
class WorkspaceData:
    agilent: pd.DataFrame
    balance: pd.DataFrame
    sensor_uncertainties: Dict[str, float]
    balance_uncertainty: float
    experiment_metadata: pd.DataFrame
    sensor_titles: Dict[str, str]


@dataclass
class PressureCalibration:
    coefficient_a_ma: Dict[str, float]
    coefficient_b_kpa: Dict[str, float]
    adjustment_uncertainty_kpa: Dict[str, float]


def _read_reference_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sensor_map = pd.read_csv(SENSOR_MAP_FILE)
    experiment_map = pd.read_csv(EXPERIMENT_MAP_FILE)
    uncertainty_map = pd.read_csv(UNCERTAINTY_FILE)
    pressure_calibration = pd.read_csv(PRESSURE_CALIBRATION_FILE)
    return sensor_map, experiment_map, uncertainty_map, pressure_calibration


def _extract_experiment_number(path: Path) -> str:
    match = re.search(r"experimento_(\d+)", path.as_posix())
    if match is None:
        raise ValueError(f"Nao foi possivel extrair o numero do experimento de {path}")
    return str(int(match.group(1)))


def _normalize_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace("+", "", regex=False)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9eE\-\+.]+", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_agilent_timestamp(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r":(\d{3})$", r".\1", regex=True)
    return pd.to_datetime(cleaned, format="%d/%m/%Y %H:%M:%S.%f", errors="coerce")


def _load_experiment_metadata(experiment_map: pd.DataFrame) -> pd.DataFrame:
    metadata = experiment_map.copy()
    metadata["experimento"] = metadata["experimento"].astype(str)
    metadata["replicate_group"] = (
        metadata["valor da vazao"].astype(str) + " | " + metadata["valor da Temperatura de entrada lado quente"].astype(str)
    )
    metadata["flow_ml_min"] = pd.to_numeric(metadata["valor da vazao"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    metadata["hot_temp_c"] = pd.to_numeric(
        metadata["valor da Temperatura de entrada lado quente"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )
    metadata = metadata.rename(
        columns={
            "valor da vazao": "flow_setting",
            "valor da Temperatura de entrada lado quente": "hot_side_inlet_setting",
        }
    )
    return metadata


def _load_sensor_reference(
    sensor_map: pd.DataFrame,
    uncertainty_map: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float], float, PressureCalibration]:
    sensor_code_to_name: Dict[str, str] = {}
    sensor_titles: Dict[str, str] = {}
    sensor_uncertainties: Dict[str, float] = {}

    uncertainty_lookup = dict(zip(uncertainty_map["instrumento"].astype(str), uncertainty_map["incerteza"]))
    balance_uncertainty = float(uncertainty_lookup.get("balanca", 0.0))
    pressure_calibration = PressureCalibration(coefficient_a_ma={}, coefficient_b_kpa={}, adjustment_uncertainty_kpa={})

    if PRESSURE_CALIBRATION_FILE.exists():
        pressure_df = pd.read_csv(PRESSURE_CALIBRATION_FILE)
        pressure_df["instrumento"] = pressure_df["instrumento"].astype(str)
        pressure_df["A (mA)"] = pd.to_numeric(pressure_df["A (mA)"], errors="coerce")
        pressure_df["B"] = pd.to_numeric(pressure_df["B"], errors="coerce")
        pressure_df["incerteza_ajuste (AX+B=Y em KPa)"] = pd.to_numeric(
            pressure_df["incerteza_ajuste (AX+B=Y em KPa)"],
            errors="coerce",
        )
        for _, row in pressure_df.iterrows():
            pressure_calibration.coefficient_a_ma[str(row["instrumento"])] = float(row["A (mA)"])
            pressure_calibration.coefficient_b_kpa[str(row["instrumento"])] = float(row["B"])
            pressure_calibration.adjustment_uncertainty_kpa[str(row["instrumento"])] = float(row["incerteza_ajuste (AX+B=Y em KPa)"])

    for _, row in sensor_map.iterrows():
        code = str(row["referencia_output_agilent"])
        canonical = str(row["referencia_tratamento_dados"])
        title = str(row["titulo_tratamento_dados"])
        instrument = str(row["instrumento"])
        sensor_code_to_name[code] = canonical
        sensor_titles[canonical] = title
        if instrument in uncertainty_lookup:
            base_uncertainty = float(uncertainty_lookup[instrument])
            if canonical.startswith("P") and instrument in pressure_calibration.coefficient_a_ma:
                coeff_a = pressure_calibration.coefficient_a_ma[instrument]
                adjustment_uncertainty = pressure_calibration.adjustment_uncertainty_kpa.get(instrument, 0.0)
                sensor_uncertainties[canonical] = float(
                    np.sqrt((coeff_a * base_uncertainty) ** 2 + adjustment_uncertainty**2) * 1000.0
                )
            else:
                sensor_uncertainties[canonical] = base_uncertainty

    return sensor_code_to_name, sensor_titles, sensor_uncertainties, balance_uncertainty, pressure_calibration


def _load_agilent_file(
    path: Path,
    experiment_id: str,
    replicate_group: str,
    sensor_code_to_name: Dict[str, str],
    pressure_calibration: PressureCalibration,
) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", skiprows=14)
    measurement_columns = {}
    for column in df.columns:
        match = re.match(r"^(\d+)\s+<.*>", str(column))
        if match and match.group(1) in sensor_code_to_name:
            measurement_columns[column] = sensor_code_to_name[match.group(1)]

    df = df.rename(columns=measurement_columns)
    if "Time" not in df.columns:
        raise ValueError(f"Coluna Time nao encontrada em {path.name}")

    df["timestamp"] = _parse_agilent_timestamp(df["Time"])
    for column in measurement_columns.values():
        df[column] = pd.to_numeric(df[column], errors="coerce")

    pressure_columns = [column for column in measurement_columns.values() if column.startswith("P")]
    for column in pressure_columns:
        instrument = "143025" if column == "P1" else "143026" if column == "P2" else None
        if instrument and instrument in pressure_calibration.coefficient_a_ma:
            coeff_a = pressure_calibration.coefficient_a_ma[instrument]
            coeff_b = pressure_calibration.coefficient_b_kpa[instrument]
            df[column] = (df[column] * 1000.0 * coeff_a + coeff_b) * 1000.0

    keep_columns = ["timestamp"] + list(measurement_columns.values())
    if "Scan" in df.columns:
        keep_columns = ["Scan"] + keep_columns

    result = df[keep_columns].copy()
    result.insert(0, "replicate_group", replicate_group)
    result.insert(0, "experiment_id", experiment_id)
    return result


def _load_balance_file(path: Path, experiment_id: str, replicate_group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["Data"].astype(str) + " " + df["Hora"].astype(str), errors="coerce")
    df["Leitura"] = _normalize_numeric(df["Leitura"])
    result = df[["timestamp", "Leitura"]].copy()
    result.insert(0, "replicate_group", replicate_group)
    result.insert(0, "experiment_id", experiment_id)
    return result


@st.cache_data(show_spinner=False)
def load_workspace_data() -> WorkspaceData:
    sensor_map, experiment_map, uncertainty_map, _ = _read_reference_tables()
    sensor_code_to_name, sensor_titles, sensor_uncertainties, balance_uncertainty, pressure_calibration = _load_sensor_reference(
        sensor_map,
        uncertainty_map,
    )
    experiment_metadata = _load_experiment_metadata(experiment_map)

    agilent_frames: List[pd.DataFrame] = []
    balance_frames: List[pd.DataFrame] = []

    for folder in sorted(EXPERIMENTS_DIR.glob("experimento_*"), key=lambda path: int(_extract_experiment_number(path))):
        experiment_id = _extract_experiment_number(folder)
        metadata_row = experiment_metadata[experiment_metadata["experimento"] == experiment_id]
        if metadata_row.empty:
            continue
        replicate_group = str(metadata_row.iloc[0]["replicate_group"])

        agilent_file = next(folder.glob("Data *.csv"), None)
        balance_file = next(folder.glob("dados_*.csv"), None)

        if agilent_file is not None:
            agilent_frames.append(
                _load_agilent_file(
                    agilent_file,
                    experiment_id,
                    replicate_group,
                    sensor_code_to_name,
                    pressure_calibration,
                )
            )
        if balance_file is not None:
            balance_frames.append(_load_balance_file(balance_file, experiment_id, replicate_group))

    agilent_df = pd.concat(agilent_frames, ignore_index=True) if agilent_frames else pd.DataFrame()
    balance_df = pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()
    metadata_for_merge = experiment_metadata[["experimento", "flow_setting", "hot_side_inlet_setting"]].copy()

    if not agilent_df.empty:
        agilent_df = agilent_df.merge(metadata_for_merge, left_on="experiment_id", right_on="experimento", how="left")
    if not balance_df.empty:
        balance_df = balance_df.merge(metadata_for_merge, left_on="experiment_id", right_on="experimento", how="left")

    return WorkspaceData(
        agilent=agilent_df,
        balance=balance_df,
        sensor_uncertainties=sensor_uncertainties,
        balance_uncertainty=balance_uncertainty,
        experiment_metadata=experiment_metadata,
        sensor_titles=sensor_titles,
    )


def summarize_balance(
    df: pd.DataFrame,
    experiment_col: str,
    group_col: str,
    value_col: str,
    instrument_uncertainty: float,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for (exp_id, grp_id), group in df.groupby([experiment_col, group_col]):
        clean = group[value_col].dropna()
        mean = float(clean.mean()) if not clean.empty else np.nan
        u_t = t_student_uncertainty(group[value_col], confidence_level)
        u_comb = float(np.sqrt(instrument_uncertainty**2 + u_t**2))
        rows.append(
            {
                "experiment_id": exp_id,
                "replicate_group": grp_id,
                "permeate_mean": mean,
                "permeate_u_inst": instrument_uncertainty,
                "permeate_u_t": u_t,
                "permeate_u_comb": u_comb,
                "n_points": int(clean.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=None, engine="python")


def find_longest_true_block(mask: pd.Series) -> Optional[Tuple[int, int]]:
    values = mask.fillna(False).to_numpy()
    best_start, best_end = -1, -1
    current_start = -1
    for i, value in enumerate(values):
        if value and current_start == -1:
            current_start = i
        if not value and current_start != -1:
            if i - current_start > best_end - best_start + 1:
                best_start, best_end = current_start, i - 1
            current_start = -1
    if current_start != -1 and len(values) - current_start > best_end - best_start + 1:
        best_start, best_end = current_start, len(values) - 1
    if best_start == -1:
        return None
    return best_start, best_end


def compute_steady_state_domains(
    df: pd.DataFrame,
    experiment_col: str,
    time_col: str,
    temperature_cols: List[str],
    temp_variation_limit: float,
    rolling_window: int,
) -> List[DomainInterval]:
    domains: List[DomainInterval] = []
    for experiment_id, group in df.groupby(experiment_col):
        ordered = group.sort_values(time_col).reset_index(drop=True)
        available_temperature_cols = [column for column in temperature_cols if column in ordered.columns]
        if len(available_temperature_cols) != len(temperature_cols):
            continue

        rolling_ranges = pd.DataFrame(index=ordered.index)
        for column in available_temperature_cols:
            rolling_ranges[column] = (
                ordered[column]
                .rolling(window=rolling_window, min_periods=rolling_window)
                .apply(lambda x: float(np.max(x) - np.min(x)), raw=True)
            )

        steady_mask = (rolling_ranges <= temp_variation_limit).all(axis=1)
        block = find_longest_true_block(steady_mask)
        if block is None:
            continue
        start_idx = max(0, block[0] - rolling_window + 1)
        end_idx = block[1]
        domains.append(
            DomainInterval(
                experiment_id=str(experiment_id),
                start_time=pd.to_datetime(ordered.loc[start_idx, time_col]),
                end_time=pd.to_datetime(ordered.loc[end_idx, time_col]),
                points=end_idx - start_idx + 1,
            )
        )
    return domains


def filter_by_domains(
    df: pd.DataFrame,
    domains: List[DomainInterval],
    experiment_col: str,
    time_col: str,
) -> pd.DataFrame:
    domain_map = {d.experiment_id: d for d in domains}
    rows = []
    for exp_id, group in df.groupby(experiment_col):
        key = str(exp_id)
        if key not in domain_map:
            continue
        d = domain_map[key]
        in_domain = group[(group[time_col] >= d.start_time) & (group[time_col] <= d.end_time)]
        rows.append(in_domain)
    if not rows:
        return df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def t_student_uncertainty(values: pd.Series, confidence_level: float = 0.95) -> float:
    clean = values.dropna().to_numpy()
    n = len(clean)
    if n <= 1:
        return 0.0
    std = float(np.std(clean, ddof=1))
    t_value = float(t.ppf((1 + confidence_level) / 2, df=n - 1))
    return t_value * std / np.sqrt(n)


def summarize_sensors(
    df: pd.DataFrame,
    experiment_col: str,
    group_col: str,
    sensor_uncertainties: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    sensor_cols = list(sensor_uncertainties.keys())
    for (exp_id, grp_id), group in df.groupby([experiment_col, group_col]):
        base = {
            "experiment_id": exp_id,
            "replicate_group": grp_id,
        }
        for sensor in sensor_cols:
            u_inst = sensor_uncertainties[sensor]
            mean = float(group[sensor].mean())
            u_t = t_student_uncertainty(group[sensor])
            u_comb = float(np.sqrt(u_inst**2 + u_t**2))
            base[f"{sensor}_mean"] = mean
            base[f"{sensor}_u_inst"] = u_inst
            base[f"{sensor}_u_t"] = u_t
            base[f"{sensor}_u_comb"] = u_comb
        rows.append(base)
    return pd.DataFrame(rows)


def grubbs_iterative(
    grouped: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    accepted_frames = []
    rejected_rows = []
    for group_name, group in grouped.groupby(group_col):
        working = group.copy().reset_index(drop=True)
        iteration = 1
        while len(working) >= 3:
            values = working[value_col].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            if std == 0:
                break
            deviations = np.abs(values - mean)
            max_idx = int(np.argmax(deviations))
            g_stat = float(deviations[max_idx] / std)
            n = len(working)
            t_crit = float(t.ppf(1 - alpha / (2 * n), df=n - 2))
            g_crit = float(((n - 1) / np.sqrt(n)) * np.sqrt((t_crit**2) / (n - 2 + t_crit**2)))
            if g_stat <= g_crit:
                break
            rejected = working.iloc[max_idx].copy()
            rejected["grubbs_iteration"] = iteration
            rejected["grubbs_stat"] = g_stat
            rejected["grubbs_critical"] = g_crit
            rejected_rows.append(rejected)
            working = working.drop(index=max_idx).reset_index(drop=True)
            iteration += 1
        accepted_frames.append(working)
    accepted_df = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else grouped.iloc[0:0]
    rejected_df = pd.DataFrame(rejected_rows) if rejected_rows else grouped.iloc[0:0]
    return accepted_df, rejected_df


def combine_replicate_uncertainty(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    value_uncertainty_col: str,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for group_name, group in df.groupby(group_col):
        n = len(group)
        group_mean = float(group[value_col].mean())
        u_exp = float(np.sqrt(np.sum(group[value_uncertainty_col].to_numpy() ** 2)) / n)
        if n > 1:
            std_rep = float(np.std(group[value_col].to_numpy(), ddof=1))
            t_value = float(t.ppf((1 + confidence_level) / 2, df=n - 1))
            u_rep = t_value * std_rep / np.sqrt(n)
        else:
            u_rep = 0.0
        u_comb = float(np.sqrt(u_exp**2 + u_rep**2))
        rows.append(
            {
                "replicate_group": group_name,
                "n_instances": n,
                "permeate_mean_group": group_mean,
                "u_from_experiments": u_exp,
                "u_from_replicates_t": u_rep,
                "u_group_combined": u_comb,
            }
        )
    return pd.DataFrame(rows)


def chi_square_propagated_uncertainty(
    accepted_df: pd.DataFrame,
    group_summary: pd.DataFrame,
    group_col: str,
    permeate_col: str,
    permeate_u_col: str,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    replicated_groups = group_summary[group_summary["n_instances"] > 1]
    if replicated_groups.empty:
        return pd.DataFrame(columns=[group_col, "u_chi_square_proxy", "u_single_final"])

    per_group_variance = accepted_df.groupby(group_col)[permeate_col].var(ddof=1).dropna()
    counts = accepted_df.groupby(group_col)[permeate_col].count()
    replicated_variances = per_group_variance[per_group_variance.index.isin(replicated_groups["replicate_group"])]
    replicated_counts = counts[counts.index.isin(replicated_groups["replicate_group"])]
    dof_total = int(np.sum(replicated_counts - 1))
    if dof_total <= 0:
        return pd.DataFrame(columns=[group_col, "u_chi_square_proxy", "u_single_final"])

    pooled_variance = float(np.sum((replicated_counts - 1) * replicated_variances) / dof_total)
    alpha = 1 - confidence_level
    chi2_low = float(chi2.ppf(alpha / 2, dof_total))
    sigma_upper = float(np.sqrt((dof_total * pooled_variance) / chi2_low))

    singles = group_summary[group_summary["n_instances"] == 1].copy()
    if singles.empty:
        return pd.DataFrame(columns=[group_col, "u_chi_square_proxy", "u_single_final"])

    singles[group_col] = singles["replicate_group"]
    single_uncertainties = (
        accepted_df[[group_col, permeate_u_col]]
        .drop_duplicates(subset=[group_col])
        .rename(columns={permeate_u_col: "u_experiment_single"})
    )
    singles = singles.merge(single_uncertainties, on=group_col, how="left")
    singles["u_chi_square_proxy"] = sigma_upper
    singles["u_single_final"] = np.sqrt(singles["u_experiment_single"] ** 2 + sigma_upper**2)
    return singles[[group_col, "u_chi_square_proxy", "u_single_final"]]


def to_datetime_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def minimalist_line_plot(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, title=title, template="plotly_white")
    fig.update_layout(
        legend_title_text="Experimento",
        margin=dict(l=10, r=10, t=40, b=10),
        title_font=dict(size=16),
    )
    return fig


def multi_sensor_line_plot(
    df: pd.DataFrame,
    x: str,
    sensor_columns: List[str],
    title: str,
    sensor_titles: Dict[str, str],
) -> go.Figure:
    available_columns = [column for column in sensor_columns if column in df.columns]
    plot_df = df[[x, "experiment_id"] + available_columns].melt(
        id_vars=[x, "experiment_id"],
        value_vars=available_columns,
        var_name="sensor",
        value_name="value",
    )
    plot_df["sensor_label"] = plot_df["sensor"].map(sensor_titles).fillna(plot_df["sensor"])
    fig = px.line(
        plot_df,
        x=x,
        y="value",
        color="sensor_label",
        line_group="experiment_id",
        title=title,
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), title_font=dict(size=16))
    return fig


def build_sensor_uncertainty_frame(sensor_uncertainties: Dict[str, float], sensor_titles: Dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sensor": sensor,
                "descricao": sensor_titles.get(sensor, sensor),
                "incerteza_instrumental": uncertainty,
            }
            for sensor, uncertainty in sensor_uncertainties.items()
        ]
    )


def build_experiment_registry(experiment_metadata: pd.DataFrame) -> pd.DataFrame:
    registry = experiment_metadata.copy()
    if "flow_ml_min" not in registry.columns:
        registry["flow_ml_min"] = pd.to_numeric(registry["flow_setting"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    if "hot_temp_c" not in registry.columns:
        registry["hot_temp_c"] = pd.to_numeric(
            registry["hot_side_inlet_setting"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
            errors="coerce",
        )
    registry = registry.sort_values(["flow_ml_min", "hot_temp_c", "experimento"], na_position="last").reset_index(drop=True)

    group_order: Dict[str, int] = {}
    group_rows = []
    for index, (replicate_group, group) in enumerate(registry.groupby("replicate_group", sort=False), start=1):
        group_order[replicate_group] = index
        experiments = group["experimento"].tolist()
        group_rows.append(
            {
                "replicate_group": replicate_group,
                "group_order": index,
                "group_call_name": f"Grupo G{index:02d}",
                "group_display_name": f"G{index:02d} · {replicate_group}",
                "is_replicate_group": len(group) > 1,
                "group_size": int(len(group)),
                "experiments": experiments,
                "experiments_label": ", ".join(experiments),
            }
        )

    group_registry = pd.DataFrame(group_rows)
    registry = registry.merge(group_registry, on="replicate_group", how="left")
    registry["experiment_display_name"] = registry.apply(
        lambda row: f"E{row['experimento']} · {row['group_display_name']}",
        axis=1,
    )
    return registry


def _group_mean_from_dataframe(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    return (
        df[[group_col, value_col]]
        .dropna(subset=[value_col])
        .drop_duplicates(subset=[group_col])
        .rename(columns={value_col: "permeate_mean"})
    )


def build_final_report(
    accepted_instances: pd.DataFrame,
    group_summary: pd.DataFrame,
    chi_single: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    replicate_registry = registry[registry["is_replicate_group"]].drop_duplicates(subset=["replicate_group"]).copy()
    single_registry = registry[~registry["is_replicate_group"]].copy()

    replicated_report = group_summary.merge(replicate_registry, on="replicate_group", how="left")
    replicated_report = replicated_report.rename(columns={"u_group_combined": "reported_uncertainty"})
    replicated_report["report_type"] = "grupo com réplicas"
    replicated_report["permeate_mean"] = replicated_report["permeate_mean_group"]
    replicated_report["report_label"] = replicated_report["group_display_name"]
    replicated_report["experiments_label"] = replicated_report["experiments_label"].fillna("")

    single_means = _group_mean_from_dataframe(accepted_instances, "replicate_group", "permeate_mean")
    single_report = chi_single.merge(single_means, on="replicate_group", how="left")
    single_report = single_report.merge(single_registry, on="replicate_group", how="left")
    single_report = single_report.rename(columns={"u_single_final": "reported_uncertainty"})
    single_report["report_type"] = "experimento único"
    single_report["report_label"] = single_report["group_display_name"]
    single_report["experiments_label"] = single_report["experiments_label"].fillna("")

    final_report = pd.concat([replicated_report, single_report], ignore_index=True, sort=False)
    if final_report.empty:
        return final_report

    keep_columns = [
        "replicate_group",
        "report_label",
        "report_type",
        "experiments_label",
        "group_size",
        "flow_setting",
        "hot_side_inlet_setting",
        "flow_ml_min",
        "hot_temp_c",
        "permeate_mean",
        "reported_uncertainty",
        "n_instances",
        "n_points",
        "u_from_experiments",
        "u_from_replicates_t",
        "u_chi_square_proxy",
        "u_single_final",
        "permeate_u_inst",
        "permeate_u_t",
        "u_group_combined",
    ]
    for column in keep_columns:
        if column not in final_report.columns:
            final_report[column] = np.nan

    final_report = final_report[keep_columns].copy()
    final_report = final_report.sort_values(["flow_ml_min", "hot_temp_c", "replicate_group"], na_position="last").reset_index(drop=True)
    return final_report


def build_grouping_tables(registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    replicate_groups = registry[registry["is_replicate_group"]].drop_duplicates(subset=["replicate_group"]).copy()
    singles = registry[~registry["is_replicate_group"]].copy()
    replicate_columns = [column for column in ["group_display_name", "experiments_label", "flow_setting", "hot_side_inlet_setting"] if column in replicate_groups.columns]
    single_columns = [column for column in ["group_display_name", "experimento", "flow_setting", "hot_side_inlet_setting"] if column in singles.columns]
    return (
        replicate_groups[replicate_columns].reset_index(drop=True),
        singles[single_columns].rename(columns={"experimento": "experiment_id"}).reset_index(drop=True),
    )


def fit_time_trend(df: pd.DataFrame, time_col: str, value_col: str) -> Dict[str, float]:
    clean = df[[time_col, value_col]].dropna().sort_values(time_col)
    if len(clean) < 2:
        return {"slope_per_min": np.nan, "intercept": np.nan, "r2": np.nan}
    x = (clean[time_col] - clean[time_col].iloc[0]).dt.total_seconds() / 60.0
    y = clean[value_col].to_numpy(dtype=float)
    result = linregress(x, y)
    return {
        "slope_per_min": float(result.slope),
        "intercept": float(result.intercept),
        "r2": float(result.rvalue**2),
    }


def summarize_experiment_detail(
    agilent_detail: pd.DataFrame,
    balance_detail: pd.DataFrame,
    experiment_id: str,
    group_row: pd.Series,
) -> pd.DataFrame:
    rows = []
    if not agilent_detail.empty:
        for sensor in ["T1", "T2", "T3", "T4", "P1", "P2"]:
            if sensor in agilent_detail.columns:
                series = agilent_detail[sensor].dropna()
                rows.append(
                    {
                        "métrica": sensor,
                        "média": float(series.mean()) if not series.empty else np.nan,
                        "mín": float(series.min()) if not series.empty else np.nan,
                        "máx": float(series.max()) if not series.empty else np.nan,
                        "amplitude": float(series.max() - series.min()) if len(series) else np.nan,
                    }
                )
    if not balance_detail.empty:
        trend = fit_time_trend(balance_detail, "timestamp", "Leitura")
        clean_balance = balance_detail["Leitura"].dropna()
        rows.append(
            {
                "métrica": "Permeado",
                "média": float(clean_balance.mean()) if not clean_balance.empty else np.nan,
                "mín": float(clean_balance.min()) if not clean_balance.empty else np.nan,
                "máx": float(clean_balance.max()) if not clean_balance.empty else np.nan,
                "amplitude": float(clean_balance.max() - clean_balance.min()) if len(clean_balance) else np.nan,
                "slope_g_min": trend["slope_per_min"],
                "r2": trend["r2"],
            }
        )
    summary = pd.DataFrame(rows)
    summary["experimento"] = experiment_id
    summary["grupo"] = group_row.get("group_display_name", group_row.get("replicate_group", ""))
    return summary


def plot_permeate_vs_conditions(report_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    plot_df = report_df.copy()
    plot_df["reported_uncertainty"] = pd.to_numeric(plot_df["reported_uncertainty"], errors="coerce")
    plot_df["permeate_mean"] = pd.to_numeric(plot_df["permeate_mean"], errors="coerce")
    plot_df = plot_df.dropna(subset=["permeate_mean"])

    fig_flow = px.scatter(
        plot_df,
        x="flow_ml_min",
        y="permeate_mean",
        color="hot_temp_c",
        size="group_size",
        error_y="reported_uncertainty",
        hover_data=["report_label", "experiments_label", "report_type"],
        title="Permeado consolidado por vazão",
        template="plotly_white",
        color_continuous_scale="Viridis",
    )
    fig_flow.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Vazão (mL/min)", yaxis_title="Permeado consolidado")

    fig_temp = px.scatter(
        plot_df,
        x="hot_temp_c",
        y="permeate_mean",
        color="flow_ml_min",
        size="group_size",
        error_y="reported_uncertainty",
        hover_data=["report_label", "experiments_label", "report_type"],
        title="Permeado consolidado por temperatura de entrada quente",
        template="plotly_white",
        color_continuous_scale="Plasma",
    )
    fig_temp.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Temperatura de entrada quente (°C)", yaxis_title="Permeado consolidado")

    heatmap = (
        plot_df.pivot_table(index="hot_temp_c", columns="flow_ml_min", values="permeate_mean", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    fig_heatmap = px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale="Cividis",
        title="Mapa de calor do permeado médio por condição AGMD",
        labels=dict(x="Vazão (mL/min)", y="Temperatura quente (°C)", color="Permeado"),
    )
    fig_heatmap.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig_flow, fig_temp, fig_heatmap


def plot_experiment_detail(agilent_detail: pd.DataFrame, balance_detail: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    temp_df = agilent_detail.melt(id_vars=["timestamp"], value_vars=[c for c in ["T1", "T2", "T3", "T4"] if c in agilent_detail.columns], var_name="sensor", value_name="value")
    temp_df = temp_df.dropna(subset=["value"])
    fig_temp = px.line(temp_df, x="timestamp", y="value", color="sensor", template="plotly_white", title="Temperaturas ao longo do tempo")
    fig_temp.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Tempo", yaxis_title="Temperatura (°C)")

    press_df = agilent_detail.melt(id_vars=["timestamp"], value_vars=[c for c in ["P1", "P2"] if c in agilent_detail.columns], var_name="sensor", value_name="value")
    press_df = press_df.dropna(subset=["value"])
    fig_press = px.line(press_df, x="timestamp", y="value", color="sensor", template="plotly_white", title="Pressões ao longo do tempo")
    fig_press.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Tempo", yaxis_title="Pressão (Pa)")

    fig_perm = px.line(balance_detail, x="timestamp", y="Leitura", template="plotly_white", title="Permeado ao longo do tempo")
    fig_perm.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Tempo", yaxis_title="Permeado")
    return fig_temp, fig_press, fig_perm


def ensure_replicate_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "replicate_group" in out.columns:
        return out
    if {"flow_setting", "hot_side_inlet_setting"}.issubset(out.columns):
        out["replicate_group"] = out["flow_setting"].astype(str) + " | " + out["hot_side_inlet_setting"].astype(str)
    else:
        out["replicate_group"] = out["experiment_id"].astype(str)
    return out


st.set_page_config(page_title="Análise AGMD - Regime Permanente", layout="wide")
st.title("Análise de Regime Permanente (AGMD)")
st.markdown(
    """
Esta interface carrega automaticamente os experimentos locais, usa o arquivo Agilent para detectar o regime permanente
pela variação de temperatura e a balança para analisar o permeado.

Fluxo executado:
1. carregamento automático dos arquivos em `Experimentos/`;
2. aplicação do mapeamento de sensores e das incertezas instrumentais;
3. filtragem temporal do domínio de regime permanente por experimento;
4. cálculo de médias e incertezas (instrumental + t-Student);
5. teste de Grubbs nas réplicas usando o permeado da balança;
6. combinação de incertezas para grupos com réplicas;
7. propagação de incerteza para experimentos sem réplicas com base em qui-quadrado.
"""
)

workspace = load_workspace_data()
experiment_registry = build_experiment_registry(workspace.experiment_metadata)
replicate_groups_table, single_experiments_table = build_grouping_tables(experiment_registry)
steady_state_temperature_cols = ["T1", "T2", "T3", "T4"]

with st.sidebar:
    st.header("Configuração")
    st.caption("O regime permanente agora é definido conjuntamente pelos sensores T1, T2, T3 e T4.")
    temp_variation_limit = st.number_input(
        "Variação máxima permitida em cada sensor de temperatura (°C)",
        min_value=0.0,
        value=0.5,
        step=0.05,
    )
    rolling_window = st.number_input("Janela (pontos) para estabilidade térmica", min_value=3, value=10, step=1)
    grubbs_alpha = st.number_input("Alpha do teste de Grubbs", min_value=0.001, max_value=0.2, value=0.05, step=0.001)
    confidence_level = st.number_input(
        "Nível de confiança para incertezas (t-Student e qui-quadrado)",
        min_value=0.80,
        max_value=0.999,
        value=0.95,
        step=0.01,
    )
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Visão geral"
    if "selected_experiment_id" not in st.session_state and not experiment_registry.empty:
        st.session_state.selected_experiment_id = str(experiment_registry.iloc[0]["experimento"])
    st.radio("Página", ["Visão geral", "Detalhe do experimento"], key="view_mode")
    st.selectbox(
        "Experimento para detalhamento",
        options=experiment_registry["experimento"].tolist(),
        index=0,
        key="selected_experiment_id",
    )


def select_experiment(experiment_id: str) -> None:
    st.session_state.selected_experiment_id = str(experiment_id)
    st.session_state.view_mode = "Detalhe do experimento"
    st.rerun()


def render_experiment_picker(registry: pd.DataFrame) -> None:
    st.subheader("Experimentos e grupos disponíveis")
    replicate_groups = registry[registry["is_replicate_group"]].drop_duplicates(subset=["replicate_group"]).copy()
    singles = registry[~registry["is_replicate_group"]].copy()

    st.markdown("**Grupos com réplicas**")
    if replicate_groups.empty:
        st.info("Nenhum grupo com réplicas foi identificado.")
    else:
        for _, row in replicate_groups.iterrows():
            with st.container(border=True):
                st.write(f"{row['group_display_name']} - {row['flow_setting']} - {row['hot_side_inlet_setting']}")
                st.caption(f"Experimentos: {row['experiments_label']}")
                button_cols = st.columns(min(4, len(row['experiments'])))
                for index, experiment_id in enumerate(row["experiments"]):
                    with button_cols[index % len(button_cols)]:
                        if st.button(f"Detalhar {experiment_id}", key=f"group_{row['replicate_group']}_{experiment_id}"):
                            select_experiment(experiment_id)

    st.markdown("**Experimentos sem réplicas**")
    if singles.empty:
        st.info("Nenhum experimento único encontrado.")
    else:
        single_cols = st.columns(4)
        for index, (_, row) in enumerate(singles.iterrows()):
            with single_cols[index % len(single_cols)]:
                if st.button(f"Detalhar {row['experimento']}", key=f"single_{row['experimento']}"):
                    select_experiment(row["experimento"])
st.subheader("1) Dados carregados automaticamente")
st.write(f"Experimentos com arquivo Agilent: {workspace.agilent['experiment_id'].nunique() if not workspace.agilent.empty else 0}")
st.write(f"Experimentos com leitura da balança: {workspace.balance['experiment_id'].nunique() if not workspace.balance.empty else 0}")

if workspace.agilent.empty or workspace.balance.empty:
    st.error("Nao foi possivel carregar todos os dados locais em Experimentos/.")
    st.stop()

st.write("Pré-visualização do arquivo Agilent consolidado")
st.dataframe(workspace.agilent.head(), use_container_width=True)
st.write("Pré-visualização da balança consolidada")
st.dataframe(workspace.balance.head(), use_container_width=True)

st.subheader("2) Incertezas e mapeamentos aplicados")
st.dataframe(build_sensor_uncertainty_frame(workspace.sensor_uncertainties, workspace.sensor_titles), use_container_width=True)
st.dataframe(workspace.experiment_metadata[["experimento", "flow_setting", "hot_side_inlet_setting", "replicate_group"]], use_container_width=True)
st.dataframe(replicate_groups_table, use_container_width=True)
st.dataframe(single_experiments_table, use_container_width=True)

render_experiment_picker(experiment_registry)

agilent_df = workspace.agilent.copy()
balance_df = workspace.balance.copy()
agilent_df = ensure_replicate_group(agilent_df)
balance_df = ensure_replicate_group(balance_df)

experiment_col = "experiment_id"
group_col = "replicate_group"
time_col = "timestamp"

st.subheader("3) Determinação do domínio de regime permanente")
domains = compute_steady_state_domains(
    agilent_df,
    experiment_col=experiment_col,
    time_col=time_col,
    temperature_cols=steady_state_temperature_cols,
    temp_variation_limit=temp_variation_limit,
    rolling_window=int(rolling_window),
)
if not domains:
    st.error("Nenhum domínio de regime permanente foi detectado com os parâmetros atuais.")
    st.stop()

domains_df = pd.DataFrame([d.__dict__ for d in domains])
st.dataframe(domains_df, use_container_width=True)

agilent_domain_df = filter_by_domains(agilent_df, domains, experiment_col, time_col)
balance_domain_df = filter_by_domains(balance_df, domains, experiment_col, time_col)

sensor_columns = [column for column in ["T1", "T2", "T3", "T4", "P1", "P2"] if column in agilent_domain_df.columns]
st.subheader("4) Estatísticas por experimento no domínio de regime permanente")
instance_summary = summarize_sensors(
    agilent_domain_df,
    experiment_col=experiment_col,
    group_col=group_col,
    sensor_uncertainties={column: workspace.sensor_uncertainties.get(column, 0.0) for column in sensor_columns},
)

balance_summary = summarize_balance(
    balance_domain_df,
    experiment_col=experiment_col,
    group_col=group_col,
    value_col="Leitura",
    instrument_uncertainty=workspace.balance_uncertainty,
    confidence_level=float(confidence_level),
)

instance_summary = instance_summary.merge(
    balance_summary,
    on=["experiment_id", "replicate_group"],
    how="left",
)
st.dataframe(instance_summary, use_container_width=True)

permeate_u_col = "permeate_u_comb"
permeate_mean_col = "permeate_mean"

if permeate_u_col not in instance_summary.columns or permeate_mean_col not in instance_summary.columns:
    st.error("Não foi possível calcular a incerteza do permeado.")
    st.stop()

st.subheader("5) Teste de Grubbs para réplicas (baseado no permeado)")
accepted_instances, rejected_instances = grubbs_iterative(
    instance_summary,
    group_col="replicate_group",
    value_col=permeate_mean_col,
    alpha=float(grubbs_alpha),
)
st.write("Instâncias aceitas após Grubbs")
st.dataframe(accepted_instances, use_container_width=True)

if not rejected_instances.empty:
    st.warning("Foram detectados outliers pelo teste de Grubbs.")
    st.dataframe(rejected_instances, use_container_width=True)
    csv_buffer = io.StringIO()
    rejected_instances.to_csv(csv_buffer, index=False)
    st.download_button(
        "Baixar CSV de experimentos recusados (Grubbs)",
        data=csv_buffer.getvalue(),
        file_name="experimentos_recusados_grubbs.csv",
        mime="text/csv",
    )
else:
    st.success("Nenhuma instância foi recusada pelo teste de Grubbs.")

st.subheader("6) Incerteza combinada por grupo de réplicas")
group_summary = combine_replicate_uncertainty(
    accepted_instances,
    group_col="replicate_group",
    value_col=permeate_mean_col,
    value_uncertainty_col=permeate_u_col,
    confidence_level=float(confidence_level),
)
st.dataframe(group_summary, use_container_width=True)

st.subheader("7) Propagação para experimentos sem réplica via qui-quadrado")
chi_single = chi_square_propagated_uncertainty(
    accepted_df=accepted_instances,
    group_summary=group_summary,
    group_col="replicate_group",
    permeate_col=permeate_mean_col,
    permeate_u_col=permeate_u_col,
    confidence_level=float(confidence_level),
)
if chi_single.empty:
    st.info("Não há grupos com instância única para propagação por qui-quadrado (ou faltam réplicas válidas).")
else:
    st.dataframe(chi_single, use_container_width=True)

final_report = build_final_report(
    accepted_instances=accepted_instances,
    group_summary=group_summary,
    chi_single=chi_single,
    registry=experiment_registry,
)

st.subheader("8) Resumo consolidado por condição AGMD")
st.dataframe(final_report, use_container_width=True)

fig_flow_cond, fig_temp_cond, fig_heatmap = plot_permeate_vs_conditions(final_report)

st.subheader("9) O que mais importa para o AGMD")
st.plotly_chart(fig_flow_cond, use_container_width=True)
st.plotly_chart(fig_temp_cond, use_container_width=True)
st.plotly_chart(fig_heatmap, use_container_width=True)

if st.session_state.view_mode == "Detalhe do experimento":
    selected_experiment_id = str(st.session_state.selected_experiment_id)
    selected_group = experiment_registry[experiment_registry["experimento"] == selected_experiment_id]
    if selected_group.empty:
        st.error("Experimento selecionado não encontrado.")
        st.stop()
    group_row = selected_group.iloc[0]

    st.subheader(f"Detalhamento do experimento {selected_experiment_id}")
    st.write(f"Grupo: {group_row['group_display_name']}")
    st.write(f"Condição: {group_row['flow_setting']} | {group_row['hot_side_inlet_setting']}")
    st.write(f"Experimentos do grupo: {group_row['experiments_label']}")

    detail_agilent = agilent_domain_df[agilent_domain_df["experiment_id"] == selected_experiment_id].copy()
    detail_balance = balance_domain_df[balance_domain_df["experiment_id"] == selected_experiment_id].copy()
    if detail_agilent.empty or detail_balance.empty:
        st.warning("Este experimento não possui dados completos no domínio selecionado.")
    else:
        detail_metrics = summarize_experiment_detail(detail_agilent, detail_balance, selected_experiment_id, group_row)
        st.dataframe(detail_metrics, use_container_width=True)

        fig_detail_temp, fig_detail_press, fig_detail_perm = plot_experiment_detail(detail_agilent, detail_balance)
        st.plotly_chart(fig_detail_temp, use_container_width=True)
        st.plotly_chart(fig_detail_press, use_container_width=True)
        st.plotly_chart(fig_detail_perm, use_container_width=True)

        domain_start = detail_agilent[time_col].min()
        domain_end = detail_agilent[time_col].max()
        st.caption(f"Domínio filtrado: {domain_start} até {domain_end}")

st.subheader("10) Domínio filtrado da balança")
st.dataframe(balance_domain_df.head(1000), use_container_width=True)
