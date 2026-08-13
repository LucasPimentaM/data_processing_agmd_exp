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

# ==========================================
# CONFIGURAÇÕES DE DIRETÓRIO E ARQUIVOS
# ==========================================
APP_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = APP_DIR / "Experimentos"
AUX_DIR = APP_DIR / "arquivos_auxiliares"
SENSOR_MAP_FILE = AUX_DIR / "mapeamento_sensor_output.csv"
EXPERIMENT_MAP_FILE = AUX_DIR / "mapeamento_experimentos_parametros.csv"
UNCERTAINTY_FILE = AUX_DIR / "incertezas_instrumentais.csv"
PRESSURE_CALIBRATION_FILE = AUX_DIR / "relacao_corrente_pressao.csv"

# ==========================================
# CONTROLE DE EXCEÇÕES E DADOS COMPROMETIDOS
# ==========================================
EXCLUDED_EXPERIMENTS = ["17", "18"]
EXCLUSION_REASON = "Erro experimental ocorrido durante as rodadas na bancada."

# ==========================================
# ESTRUTURAS DE DADOS
# ==========================================
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
    experiment_metadata: pd.DataFrame
    sensor_titles: Dict[str, str]

@dataclass
class PressureCalibration:
    coefficient_a_ma: Dict[str, float]
    coefficient_b_kpa: Dict[str, float]
    adjustment_uncertainty_kpa: Dict[str, float]

# ==========================================
# FUNÇÕES DE LEITURA E LIMPEZA
# ==========================================
def _read_reference_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sensor_map = pd.read_csv(SENSOR_MAP_FILE)
    experiment_map = pd.read_csv(EXPERIMENT_MAP_FILE)
    uncertainty_map = pd.read_csv(UNCERTAINTY_FILE)
    if PRESSURE_CALIBRATION_FILE.exists():
        pressure_calibration = pd.read_csv(PRESSURE_CALIBRATION_FILE)
    else:
        pressure_calibration = pd.DataFrame()
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
    metadata = metadata[~metadata["experimento"].isin(EXCLUDED_EXPERIMENTS)]
    
    metadata["replicate_group"] = (
        metadata["valor da vazao"].astype(str) + " | " + metadata["valor da Temperatura de entrada lado quente"].astype(str)
    )
    metadata["flow_ml_min"] = pd.to_numeric(metadata["valor da vazao"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    metadata["hot_temp_c"] = pd.to_numeric(
        metadata["valor da Temperatura de entrada lado quente"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce",
    )
    metadata = metadata.rename(
        columns={"valor da vazao": "flow_setting", "valor da Temperatura de entrada lado quente": "hot_side_inlet_setting"}
    )
    return metadata

def _load_sensor_reference(
    sensor_map: pd.DataFrame, uncertainty_map: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float], PressureCalibration]:
    sensor_code_to_name: Dict[str, str] = {}
    sensor_titles: Dict[str, str] = {}
    sensor_uncertainties: Dict[str, float] = {}

    uncertainty_lookup = dict(zip(uncertainty_map["instrumento"].astype(str), uncertainty_map["incerteza"]))
    pressure_calibration = PressureCalibration(coefficient_a_ma={}, coefficient_b_kpa={}, adjustment_uncertainty_kpa={})

    if PRESSURE_CALIBRATION_FILE.exists():
        pressure_df = pd.read_csv(PRESSURE_CALIBRATION_FILE)
        for _, row in pressure_df.iterrows():
            inst = str(row["instrumento"])
            pressure_calibration.coefficient_a_ma[inst] = float(row["A (mA)"])
            pressure_calibration.coefficient_b_kpa[inst] = float(row["B"])
            pressure_calibration.adjustment_uncertainty_kpa[inst] = float(row["incerteza_ajuste (AX+B=Y em KPa)"])

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
                sensor_uncertainties[canonical] = float(np.sqrt((coeff_a * base_uncertainty) ** 2 + adjustment_uncertainty**2) * 1000.0)
            else:
                sensor_uncertainties[canonical] = base_uncertainty

    return sensor_code_to_name, sensor_titles, sensor_uncertainties, pressure_calibration

def _load_agilent_file(path: Path, experiment_id: str, replicate_group: str, sensor_code_to_name: Dict[str, str], pressure_calibration: PressureCalibration) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", skiprows=14)
    measurement_columns = {col: sensor_code_to_name[m.group(1)] for col in df.columns if (m := re.match(r"^(\d+)\s+<.*>", str(col))) and m.group(1) in sensor_code_to_name}
    df = df.rename(columns=measurement_columns)
    df["timestamp"] = _parse_agilent_timestamp(df["Time"])
    
    for column in measurement_columns.values():
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in [c for c in measurement_columns.values() if c.startswith("P")]:
        instrument = "143025" if column == "P1" else "143026" if column == "P2" else None
        if instrument and instrument in pressure_calibration.coefficient_a_ma:
            coeff_a = pressure_calibration.coefficient_a_ma[instrument]
            coeff_b = pressure_calibration.coefficient_b_kpa[instrument]
            df[column] = (df[column] * 1000.0 * coeff_a + coeff_b) * 1000.0

    keep_columns = ["timestamp"] + list(measurement_columns.values())
    if "Scan" in df.columns: keep_columns = ["Scan"] + keep_columns

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
    sensor_code_to_name, sensor_titles, sensor_uncertainties, pressure_calibration = _load_sensor_reference(sensor_map, uncertainty_map)
    experiment_metadata = _load_experiment_metadata(experiment_map)

    agilent_frames, balance_frames = [], []

    if EXPERIMENTS_DIR.exists():
        for folder in sorted(EXPERIMENTS_DIR.glob("experimento_*"), key=lambda path: int(_extract_experiment_number(path))):
            experiment_id = _extract_experiment_number(folder)
            if experiment_id in EXCLUDED_EXPERIMENTS: continue
                
            metadata_row = experiment_metadata[experiment_metadata["experimento"] == experiment_id]
            if metadata_row.empty: continue
            replicate_group = str(metadata_row.iloc[0]["replicate_group"])

            agilent_file = next(folder.glob("Data *.csv"), None)
            balance_file = next(folder.glob("dados_*.csv"), None)

            if agilent_file is not None: agilent_frames.append(_load_agilent_file(agilent_file, experiment_id, replicate_group, sensor_code_to_name, pressure_calibration))
            if balance_file is not None: balance_frames.append(_load_balance_file(balance_file, experiment_id, replicate_group))

    agilent_df = pd.concat(agilent_frames, ignore_index=True) if agilent_frames else pd.DataFrame()
    balance_df = pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()
    metadata_for_merge = experiment_metadata[["experimento", "flow_setting", "hot_side_inlet_setting"]].copy()

    if not agilent_df.empty: agilent_df = agilent_df.merge(metadata_for_merge, left_on="experiment_id", right_on="experimento", how="left")
    if not balance_df.empty: balance_df = balance_df.merge(metadata_for_merge, left_on="experiment_id", right_on="experimento", how="left")

    return WorkspaceData(agilent_df, balance_df, sensor_uncertainties, experiment_metadata, sensor_titles)


# ==========================================
# FUNÇÕES ESTATÍSTICAS E MATEMÁTICAS (METROLOGIA GUM)
# ==========================================
def find_longest_true_block(mask: pd.Series) -> Optional[Tuple[int, int]]:
    values = mask.fillna(False).to_numpy()
    best_start, best_end = -1, -1
    current_start = -1
    for i, value in enumerate(values):
        if value and current_start == -1: current_start = i
        if not value and current_start != -1:
            if i - current_start > best_end - best_start + 1: best_start, best_end = current_start, i - 1
            current_start = -1
    if current_start != -1 and len(values) - current_start > best_end - best_start + 1:
        best_start, best_end = current_start, len(values) - 1
    if best_start == -1: return None
    return best_start, best_end

def compute_steady_state_domains(df: pd.DataFrame, experiment_col: str, time_col: str, temperature_cols: List[str], temp_variation_limit: float, rolling_window: int) -> List[DomainInterval]:
    domains = []
    for experiment_id, group in df.groupby(experiment_col):
        ordered = group.sort_values(time_col).reset_index(drop=True)
        available_temperature_cols = [column for column in temperature_cols if column in ordered.columns]
        if len(available_temperature_cols) != len(temperature_cols): continue

        rolling_ranges = pd.DataFrame(index=ordered.index)
        for column in available_temperature_cols:
            rolling_ranges[column] = ordered[column].rolling(window=rolling_window, min_periods=rolling_window).apply(lambda x: float(np.max(x) - np.min(x)), raw=True)

        steady_mask = (rolling_ranges <= temp_variation_limit).all(axis=1)
        block = find_longest_true_block(steady_mask)
        if block is None: continue
        start_idx = max(0, block[0] - rolling_window + 1)
        end_idx = block[1]
        domains.append(DomainInterval(str(experiment_id), pd.to_datetime(ordered.loc[start_idx, time_col]), pd.to_datetime(ordered.loc[end_idx, time_col]), end_idx - start_idx + 1))
    return domains

def filter_by_domains(df: pd.DataFrame, domains: List[DomainInterval], experiment_col: str, time_col: str) -> pd.DataFrame:
    domain_map = {d.experiment_id: d for d in domains}
    rows = []
    for exp_id, group in df.groupby(experiment_col):
        if str(exp_id) in domain_map:
            d = domain_map[str(exp_id)]
            rows.append(group[(group[time_col] >= d.start_time) & (group[time_col] <= d.end_time)])
    return pd.concat(rows, ignore_index=True) if rows else df.iloc[0:0].copy()

def standard_uncertainty_type_a(values: pd.Series) -> Tuple[float, int]:
    """Retorna Incerteza Padrão Tipo A (1 sigma) e graus de liberdade."""
    clean = values.dropna().to_numpy()
    n = len(clean)
    if n <= 1: return 0.0, 0
    std = float(np.std(clean, ddof=1))
    return std / np.sqrt(n), n - 1

def fit_time_trend(df: pd.DataFrame, time_col: str, value_col: str) -> Dict[str, float]:
    clean = df[[time_col, value_col]].dropna().sort_values(time_col)
    if len(clean) < 2: return {"slope_per_min": np.nan, "intercept": np.nan, "r2": np.nan, "stderr": np.nan}
    x = (clean[time_col] - clean[time_col].iloc[0]).dt.total_seconds() / 60.0
    y = clean[value_col].to_numpy(dtype=float)
    result = linregress(x, y)
    return {"slope_per_min": float(result.slope), "intercept": float(result.intercept), "r2": float(result.rvalue**2), "stderr": float(result.stderr)}

def summarize_sensors(df: pd.DataFrame, experiment_col: str, group_col: str, sensor_uncertainties: Dict[str, float], confidence_level: float = 0.95) -> pd.DataFrame:
    rows = []
    for (exp_id, grp_id), group in df.groupby([experiment_col, group_col]):
        base = {"experiment_id": exp_id, "replicate_group": grp_id}
        for sensor, u_b in sensor_uncertainties.items():
            if sensor in group.columns:
                u_a, dof_a = standard_uncertainty_type_a(group[sensor])
                base[f"{sensor}_mean"] = float(group[sensor].mean())
                base[f"{sensor}_u_type_a"] = u_a
                base[f"{sensor}_u_type_b"] = u_b
                
                u_c = float(np.sqrt(u_a**2 + u_b**2))
                base[f"{sensor}_u_comb_std"] = u_c
                
                dof_eff = (u_c**4) / ((u_a**4) / dof_a) if u_a > 0 else np.inf
                k = float(t.ppf((1 + confidence_level) / 2, df=dof_eff)) if dof_eff < np.inf and dof_eff >= 1 else 2.00
                base[f"{sensor}_u_expanded"] = k * u_c
                
        rows.append(base)
    return pd.DataFrame(rows)

def summarize_balance(df: pd.DataFrame, experiment_col: str, group_col: str) -> pd.DataFrame:
    """Extrai estritamente a taxa e a Incerteza Padrão (Erro Padrão) proveniente do ajuste linear."""
    rows = []
    for (exp_id, grp_id), group in df.groupby([experiment_col, group_col]):
        trend = fit_time_trend(group, "timestamp", "Leitura")
        slope = trend["slope_per_min"] 
        stderr = trend["stderr"] if not np.isnan(trend["stderr"]) else 0.0 
        n = len(group.dropna(subset=["Leitura"]))
        
        # Incerteza Padrão (Tipo A) vinda puramente da regressão linear
        u_a_reg = stderr if n > 2 else 0.0
        dof_reg = n - 2 if n > 2 else 0
            
        rows.append({
            "experiment_id": exp_id,
            "replicate_group": grp_id,
            "permeate_rate": slope,
            "permeate_rate_u_a_reg": u_a_reg,
            "permeate_rate_dof_reg": dof_reg,
            "r2": trend["r2"],
            "n_points": n,
        })
    return pd.DataFrame(rows)

def grubbs_iterative(grouped: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    accepted_frames, rejected_rows = [], []
    for group_name, group in grouped.groupby(group_col):
        working = group.copy().reset_index(drop=True)
        iteration = 1
        while len(working) >= 3:
            values = working[value_col].to_numpy(dtype=float)
            mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
            if std == 0: break
            deviations = np.abs(values - mean)
            max_idx = int(np.argmax(deviations))
            g_stat = float(deviations[max_idx] / std)
            n = len(working)
            t_crit = float(t.ppf(1 - alpha / (2 * n), df=n - 2))
            g_crit = float(((n - 1) / np.sqrt(n)) * np.sqrt((t_crit**2) / (n - 2 + t_crit**2)))
            if g_stat <= g_crit: break
            
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

def combine_replicate_uncertainty(df: pd.DataFrame, group_col: str, value_col: str, inst_u_col: str, inst_dof_col: str, confidence_level: float = 0.95) -> pd.DataFrame:
    """Combina as incertezas de regressão (internas) com a variabilidade das réplicas (externas)."""
    rows = []
    for group_name, group in df.groupby(group_col):
        n = len(group)
        group_mean = float(group[value_col].mean())
        
        # Propagação da Incerteza da Regressão Linear para a média das réplicas
        u_inst_arr = group[inst_u_col].to_numpy()
        dof_inst_arr = group[inst_dof_col].to_numpy()
        
        # Incerteza da média devida às regressões: 1/n * sqrt(sum(u_i^2))
        u_reg_mean = float(np.sqrt(np.sum(u_inst_arr**2)) / n)
        
        # Graus de liberdade efetivos da incerteza combinada das regressões
        if u_reg_mean > 0:
            num = (np.sum(u_inst_arr**2))**2
            den = np.sum((u_inst_arr**4) / np.where(dof_inst_arr > 0, dof_inst_arr, np.inf))
            dof_reg_mean = float(num / den) if den > 0 else np.inf
        else:
            dof_reg_mean = np.inf
        
        if n > 1:
            # Variabilidade Tipo A Estatística da Média das réplicas (Variabilidade externa)
            std_rep = float(np.std(group[value_col].to_numpy(), ddof=1))
            u_a_rep = std_rep / np.sqrt(n)
            dof_a_rep = n - 1
        else:
            u_a_rep = 0.0
            dof_a_rep = 0
            
        # Combinação estritamente de Incertezas Padrão
        u_c = float(np.sqrt(u_a_rep**2 + u_reg_mean**2))
        
        # Expansão Final (Welch-Satterthwaite)
        term_a = (u_a_rep**4) / dof_a_rep if dof_a_rep > 0 else 0
        term_reg = (u_reg_mean**4) / dof_reg_mean if (dof_reg_mean < np.inf and dof_reg_mean > 0) else 0
        
        if u_c > 0 and (term_a + term_reg) > 0:
            dof_eff = (u_c**4) / (term_a + term_reg)
        else:
            dof_eff = np.inf
            
        k = float(t.ppf((1 + confidence_level) / 2, df=dof_eff)) if dof_eff < np.inf and dof_eff >= 1 else 2.00
        u_expanded = k * u_c
            
        rows.append({
            "replicate_group": group_name, "n_instances": n, "permeate_rate_mean": group_mean,
            "u_type_a_rep": u_a_rep, "dof_a_rep": dof_a_rep, 
            "u_reg_mean": u_reg_mean, "dof_reg_mean": dof_reg_mean,
            "u_combined_standard": u_c, "dof_eff": dof_eff, "k_factor": k,
            "u_expanded": u_expanded
        })
    return pd.DataFrame(rows)

def chi_square_propagated_uncertainty(accepted_df: pd.DataFrame, group_summary: pd.DataFrame, group_col: str, permeate_col: str, inst_u_col: str, inst_dof_col: str, confidence_level: float = 0.95) -> pd.DataFrame:
    """Usa a variância agrupada (pooled variance) + Erro da Regressão para experimentos n=1."""
    replicated_groups = group_summary[group_summary["n_instances"] > 1]
    if replicated_groups.empty: return pd.DataFrame(columns=[group_col, "u_proxy_type_a", "u_single_expanded"])

    per_group_var = accepted_df.groupby(group_col)[permeate_col].var(ddof=1).dropna()
    counts = accepted_df.groupby(group_col)[permeate_col].count()
    rep_vars = per_group_var[per_group_var.index.isin(replicated_groups["replicate_group"])]
    rep_counts = counts[counts.index.isin(replicated_groups["replicate_group"])]
    dof_total = int(np.sum(rep_counts - 1))
    
    if dof_total <= 0: return pd.DataFrame(columns=[group_col, "u_proxy_type_a", "u_single_expanded"])

    # Raiz da variância agrupada substituindo o limite superior do Qui-Quadrado (Variabilidade base)
    pooled_var = float(np.sum((rep_counts - 1) * rep_vars) / dof_total)
    u_proxy_type_a = float(np.sqrt(pooled_var))

    singles = group_summary[group_summary["n_instances"] == 1].copy()
    if singles.empty: return pd.DataFrame(columns=[group_col, "u_proxy_type_a", "u_single_expanded"])

    singles[group_col] = singles["replicate_group"]
    single_inst = accepted_df[[group_col, inst_u_col, inst_dof_col]].drop_duplicates(subset=[group_col])
    singles = singles.merge(single_inst, on=group_col, how="left")
    
    rows = []
    for _, row in singles.iterrows():
        u_reg = float(row[inst_u_col])
        dof_reg = float(row[inst_dof_col])
        
        # Combina a variância base com o erro da regressão específica do ensaio n=1
        u_c = float(np.sqrt(u_proxy_type_a**2 + u_reg**2))
        
        # Welch-Satterthwaite para n=1
        term_proxy = (u_proxy_type_a**4) / dof_total if dof_total > 0 else 0
        term_reg = (u_reg**4) / dof_reg if dof_reg > 0 else 0
        
        if u_c > 0 and (term_proxy + term_reg) > 0:
            dof_eff = (u_c**4) / (term_proxy + term_reg)
        else:
            dof_eff = np.inf
            
        k = float(t.ppf((1 + confidence_level) / 2, df=dof_eff)) if dof_eff < np.inf and dof_eff >= 1 else 2.00
        u_expanded = k * u_c
        
        rows.append({
            group_col: row[group_col],
            "u_proxy_type_a": u_proxy_type_a,
            "u_reg_single": u_reg,
            "dof_proxy": dof_total,
            "u_single_combined_standard": u_c,
            "dof_eff": dof_eff,
            "k_factor": k,
            "u_single_expanded": u_expanded
        })
        
    return pd.DataFrame(rows)

def aggregate_sensors(accepted_df: pd.DataFrame, sensor_columns: List[str], confidence_level: float = 0.95) -> pd.DataFrame:
    """Agrega e propaga a incerteza para os sensores de Temperatura e Pressão."""
    rows = []
    for group_name, group in accepted_df.groupby("replicate_group"):
        row_data = {"replicate_group": group_name}
        for sensor in sensor_columns:
            mean_col = f"{sensor}_mean"
            if mean_col not in group.columns: 
                continue
            
            sensor_data = group.dropna(subset=[mean_col])
            n = len(sensor_data)
            
            if n == 0:
                row_data[f"{sensor}_mean"] = np.nan
                row_data[f"{sensor}_u_expanded"] = np.nan
                continue
                
            if n > 1:
                # Tipo A: Variabilidade da média das réplicas
                u_a_rep = float(np.std(sensor_data[mean_col].to_numpy(), ddof=1) / np.sqrt(n))
                dof_a = n - 1
                
                # Tipo B: Erro instrumental (constante para o mesmo sensor/instrumento)
                u_b = float(sensor_data[f"{sensor}_u_type_b"].iloc[0])
                
                # Incerteza Combinada
                u_c = float(np.sqrt(u_a_rep**2 + u_b**2))
                
                # Graus de liberdade efetivos e fator k
                dof_eff = (u_c**4) / ((u_a_rep**4) / dof_a) if u_a_rep > 0 else np.inf
                k = float(t.ppf((1 + confidence_level) / 2, df=dof_eff)) if dof_eff < np.inf and dof_eff >= 1 else 2.00
                
                mean_val = float(sensor_data[mean_col].mean())
                u_exp = k * u_c
            else:
                # n=1: Utiliza a propagação da série temporal já calculada em summarize_sensors
                mean_val = float(sensor_data[mean_col].iloc[0])
                u_exp = float(sensor_data[f"{sensor}_u_expanded"].iloc[0])
                
            row_data[f"{sensor}_mean"] = mean_val
            row_data[f"{sensor}_u_expanded"] = u_exp
        rows.append(row_data)
    return pd.DataFrame(rows)

def build_experiment_registry(experiment_metadata: pd.DataFrame) -> pd.DataFrame:
    registry = experiment_metadata.copy()
    if "flow_ml_min" not in registry.columns: registry["flow_ml_min"] = pd.to_numeric(registry["flow_setting"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    if "hot_temp_c" not in registry.columns: registry["hot_temp_c"] = pd.to_numeric(registry["hot_side_inlet_setting"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    registry = registry.sort_values(["flow_ml_min", "hot_temp_c", "experimento"], na_position="last").reset_index(drop=True)

    group_rows = []
    for index, (replicate_group, group) in enumerate(registry.groupby("replicate_group", sort=False), start=1):
        experiments = group["experimento"].tolist()
        group_rows.append({
            "replicate_group": replicate_group, "group_display_name": f"G{index:02d} · {replicate_group}",
            "is_replicate_group": len(group) > 1, "experiments_label": ", ".join(experiments),
        })

    registry = registry.merge(pd.DataFrame(group_rows), on="replicate_group", how="left")
    registry["experiment_display_name"] = registry.apply(lambda row: f"E{row['experimento']} · {row['group_display_name']}", axis=1)
    return registry

def build_grouping_tables(registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    replicate_groups = registry[registry["is_replicate_group"]].drop_duplicates(subset=["replicate_group"]).copy()
    singles = registry[~registry["is_replicate_group"]].copy()
    return (
        replicate_groups[["group_display_name", "experiments_label", "flow_setting", "hot_side_inlet_setting"]].reset_index(drop=True),
        singles[["group_display_name", "experimento", "flow_setting", "hot_side_inlet_setting"]].rename(columns={"experimento": "experiment_id"}).reset_index(drop=True),
    )

def build_final_report(group_summary: pd.DataFrame, chi_single: pd.DataFrame, sensors_summary: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    group_metadata = registry.drop_duplicates(subset=["replicate_group"])[
        ["replicate_group", "group_display_name", "experiments_label", "flow_setting", "hot_side_inlet_setting", "flow_ml_min", "hot_temp_c"]
    ].copy()
    
    report = group_summary.merge(group_metadata, on="replicate_group", how="left")
    report = report.merge(chi_single[["replicate_group", "u_single_combined_standard", "u_single_expanded", "u_proxy_type_a", "u_reg_single"]], on="replicate_group", how="left")
    
    # Faz a união com os dados dos sensores processados
    if not sensors_summary.empty:
        report = report.merge(sensors_summary, on="replicate_group", how="left")
    
    report["report_type"] = np.where(report["n_instances"] > 1, "grupo com réplicas", "experimento único")
    
    # A incerteza final reportada assume a Expansão de Confiança baseada no k final
    report["reported_uncertainty"] = np.where(report["n_instances"] > 1, report["u_expanded"], report["u_single_expanded"])
    report["report_label"] = report["group_display_name"]
    
    keep_columns = [
        "replicate_group", "report_label", "report_type", "experiments_label",
        "flow_setting", "hot_side_inlet_setting", "flow_ml_min", "hot_temp_c",
        "permeate_rate_mean", "reported_uncertainty", "n_instances",
        "u_type_a_rep", "u_proxy_type_a", "u_reg_mean", "u_reg_single", "u_combined_standard", "dof_eff", "k_factor"
    ]
    
    # Adiciona as colunas dos sensores dinamicamente para exibição final
    for col in sensors_summary.columns:
        if col != "replicate_group" and col not in keep_columns:
            keep_columns.append(col)
            
    for col in keep_columns:
        if col not in report.columns:
            report[col] = np.nan
            
    return report[keep_columns].sort_values(["flow_ml_min", "hot_temp_c", "replicate_group"], na_position="last").reset_index(drop=True)

# ==========================================
# GERAÇÃO DE GRÁFICOS (2D LINHAS)
# ==========================================
def plot_2d_faceted_lines(report_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    plot_df = report_df.copy()
    plot_df["reported_uncertainty"] = pd.to_numeric(plot_df["reported_uncertainty"], errors="coerce")
    plot_df["permeate_rate_mean"] = pd.to_numeric(plot_df["permeate_rate_mean"], errors="coerce")
    plot_df["flow_ml_min"] = pd.to_numeric(plot_df["flow_ml_min"], errors="coerce")
    plot_df["hot_temp_c"] = pd.to_numeric(plot_df["hot_temp_c"], errors="coerce")
    plot_df = plot_df.dropna(subset=["permeate_rate_mean", "flow_ml_min", "hot_temp_c"])
    
    plot_df["Temp_Label"] = plot_df["hot_temp_c"].astype(str) + " °C"
    plot_df["Flow_Label"] = plot_df["flow_ml_min"].astype(str) + " mL/min"

    # Gráfico 1: Variação da Vazão (X) para cada Temperatura Fixada (Cor)
    fig_flow = px.line(
        plot_df.sort_values(["flow_ml_min", "hot_temp_c"]),
        x="flow_ml_min", y="permeate_rate_mean", color="Temp_Label",
        markers=True, error_y="reported_uncertainty",
        title="Impacto da Vazão na Taxa de Permeado (Para cada Temperatura)",
        labels={"flow_ml_min": "Vazão (mL/min)", "permeate_rate_mean": "Taxa de Permeado (g/min)", "Temp_Label": "Temperatura Fixada"},
        template="plotly_white"
    )
    fig_flow.update_traces(marker=dict(size=10))
    fig_flow.update_layout(margin=dict(l=10, r=10, t=50, b=10))

    # Gráfico 2: Variação da Temperatura (X) para cada Vazão Fixada (Cor)
    fig_temp = px.line(
        plot_df.sort_values(["hot_temp_c", "flow_ml_min"]),
        x="hot_temp_c", y="permeate_rate_mean", color="Flow_Label",
        markers=True, error_y="reported_uncertainty",
        title="Impacto da Temperatura na Taxa de Permeado (Para cada Vazão)",
        labels={"hot_temp_c": "Temperatura Quente (°C)", "permeate_rate_mean": "Taxa de Permeado (g/min)", "Flow_Label": "Vazão Fixada"},
        template="plotly_white"
    )
    fig_temp.update_traces(marker=dict(size=10))
    fig_temp.update_layout(margin=dict(l=10, r=10, t=50, b=10))

    return fig_flow, fig_temp

def plot_experiment_detail(agilent_detail: pd.DataFrame, balance_detail: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    temp_df = agilent_detail.melt(id_vars=["timestamp"], value_vars=[c for c in ["T1", "T2", "T3", "T4"] if c in agilent_detail.columns], var_name="sensor", value_name="value").dropna()
    fig_temp = px.line(temp_df, x="timestamp", y="value", color="sensor", template="plotly_white", title="Temperaturas ao longo do tempo")
    
    press_df = agilent_detail.melt(id_vars=["timestamp"], value_vars=[c for c in ["P1", "P2"] if c in agilent_detail.columns], var_name="sensor", value_name="value").dropna()
    fig_press = px.line(press_df, x="timestamp", y="value", color="sensor", template="plotly_white", title="Pressões ao longo do tempo")
    
    fig_perm = px.scatter(balance_detail, x="timestamp", y="Leitura", template="plotly_white", title="Evolução da Massa Acumulada no Permeado")
    return fig_temp, fig_press, fig_perm

def ensure_replicate_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "replicate_group" in out.columns: return out
    if {"flow_setting", "hot_side_inlet_setting"}.issubset(out.columns): out["replicate_group"] = out["flow_setting"].astype(str) + " | " + out["hot_side_inlet_setting"].astype(str)
    else: out["replicate_group"] = out["experiment_id"].astype(str)
    return out


# ==========================================
# INTERFACE DO USUÁRIO (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Análise AGMD - Regime Permanente", layout="wide")
st.title("Análise de Regime Permanente (AGMD)")

# ALERTA VISUAL DE EXCLUSÃO
if EXCLUDED_EXPERIMENTS:
    st.warning(
        f"⚠️ **Aviso de Exclusão Manual:** Os experimentos **{', '.join(EXCLUDED_EXPERIMENTS)}** "
        f"foram retirados da análise. \n\n**Motivo:** *{EXCLUSION_REASON}*"
    )

workspace = load_workspace_data()
if workspace.agilent.empty or workspace.balance.empty:
    st.warning("Não foi possível carregar os dados locais na pasta 'Experimentos/'.")
    st.stop()

experiment_registry = build_experiment_registry(workspace.experiment_metadata)
replicate_groups_table, single_experiments_table = build_grouping_tables(experiment_registry)
steady_state_temperature_cols = ["T1", "T2", "T3", "T4"]

with st.sidebar:
    st.header("⚙️ Parâmetros de Análise")
    temp_variation_limit = st.number_input("Var. máx. temp. (°C)", min_value=0.0, value=0.5, step=0.05)
    rolling_window = st.number_input("Janela estabilidade (pontos)", min_value=3, value=10, step=1)
    grubbs_alpha = st.number_input("Alpha do Grubbs", min_value=0.001, max_value=0.2, value=0.05, step=0.001)
    confidence_level = st.number_input("Nível de Confiança", min_value=0.80, max_value=0.999, value=0.95, step=0.01)
    
    st.divider()
    view_mode = st.radio("Navegação", ["Visão Geral", "Detalhe do Experimento"])
    
    if view_mode == "Detalhe do Experimento":
        exp_selecionado = st.selectbox("Selecione o Experimento", options=experiment_registry["experimento"].tolist() if not experiment_registry.empty else [])

# Core de Processamento de Dados
agilent_df = ensure_replicate_group(workspace.agilent.copy())
balance_df = ensure_replicate_group(workspace.balance.copy())

domains = compute_steady_state_domains(agilent_df, "experiment_id", "timestamp", steady_state_temperature_cols, temp_variation_limit, int(rolling_window))
if not domains:
    st.error("Nenhum domínio detectado. Aumente a variação máxima permitida.")
    st.stop()

agilent_domain_df = filter_by_domains(agilent_df, domains, "experiment_id", "timestamp")
balance_domain_df = filter_by_domains(balance_df, domains, "experiment_id", "timestamp")

# Cálculo de Taxa e Resumo
sensor_columns = [col for col in ["T1", "T2", "T3", "T4", "P1", "P2"] if col in agilent_domain_df.columns]
instance_summary = summarize_sensors(agilent_domain_df, "experiment_id", "replicate_group", {col: workspace.sensor_uncertainties.get(col, 0.0) for col in sensor_columns}, float(confidence_level))

# Note que a incerteza da balança da tabela não é mais repassada
balance_summary = summarize_balance(balance_domain_df, "experiment_id", "replicate_group")
instance_summary = instance_summary.merge(balance_summary, on=["experiment_id", "replicate_group"], how="left")

# Teste Grubbs Exclusivo para Taxa de Permeado
accepted_instances, rejected_instances = grubbs_iterative(instance_summary, "replicate_group", "permeate_rate", float(grubbs_alpha))
group_summary = combine_replicate_uncertainty(accepted_instances, "replicate_group", "permeate_rate", "permeate_rate_u_a_reg", "permeate_rate_dof_reg", float(confidence_level))
chi_single = chi_square_propagated_uncertainty(accepted_instances, group_summary, "replicate_group", "permeate_rate", "permeate_rate_u_a_reg", "permeate_rate_dof_reg", float(confidence_level))

# Consolidação dos Sensores para a tabela final
sensors_summary = aggregate_sensors(accepted_instances, sensor_columns, float(confidence_level))
final_report = build_final_report(group_summary, chi_single, sensors_summary, experiment_registry)

# Visualizações
if view_mode == "Visão Geral":
    tab1, tab2, tab3 = st.tabs(["📊 Resultados e Gráficos de Tendência", "🛠️ Diagnóstico de Regime & Outliers", "📁 Tabelas Auxiliares"])
    
    with tab1:
        st.subheader("Gráficos de Linha: Comportamento do AGMD")
        st.caption(f"Barras de incerteza exibem a Incerteza Expandida de **{confidence_level*100:.0f}%** baseada em Welch-Satterthwaite.")
        
        fig_flow_2d, fig_temp_2d = plot_2d_faceted_lines(final_report)
        
        col_graf_1, col_graf_2 = st.columns(2)
        with col_graf_1: st.plotly_chart(fig_flow_2d, use_container_width=True)
        with col_graf_2: st.plotly_chart(fig_temp_2d, use_container_width=True)
        
        st.divider()
        st.subheader("Tabela Consolidada (Taxa de Permeado, Temperaturas e Pressões)")
        st.dataframe(final_report, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Outliers Detectados (Teste de Grubbs)")
        if not rejected_instances.empty:
            st.error(f"Foram rejeitados {len(rejected_instances)} experimento(s) baseados na taxa calculada.")
            outliers_view = rejected_instances[["experiment_id", "replicate_group", "permeate_rate", "grubbs_stat", "grubbs_critical"]]
            outliers_view = outliers_view.rename(columns={"permeate_rate": "Taxa Rejeitada (g/min)", "grubbs_stat": "Estatística Grubbs", "grubbs_critical": "Limite Crítico"})
            st.dataframe(outliers_view, use_container_width=True, hide_index=True)
        else:
            st.success("Nenhum experimento foi caracterizado como anomalia (outlier) pelo Grubbs.")

        st.divider()
        st.subheader("Domínios de Regime Permanente")
        st.dataframe(pd.DataFrame([d.__dict__ for d in domains]), use_container_width=True, hide_index=True)

        st.subheader("Propagação das Incertezas (GUM)")
        with st.expander("Incerteza Combinada (Grupos de Réplicas)"): st.dataframe(group_summary, use_container_width=True, hide_index=True)
        with st.expander("Incerteza Testes Únicos (Pooled Variance + Erro Regressão)"): st.dataframe(chi_single, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Tabelas Estruturais")
        with st.expander("Mapeamento de Sensores"): st.dataframe(pd.DataFrame([{"Sensor": k, "Incerteza": v} for k, v in workspace.sensor_uncertainties.items()]), use_container_width=True, hide_index=True)
        with st.expander("Mapeamento de Grupos"): st.dataframe(replicate_groups_table, use_container_width=True, hide_index=True)
        with st.expander("Dados Brutos Agilent (Top 500)"): st.dataframe(workspace.agilent.head(500), use_container_width=True)
        with st.expander("Dados Brutos Balança (Top 500)"): st.dataframe(workspace.balance.head(500), use_container_width=True)

elif view_mode == "Detalhe do Experimento":
    if not 'exp_selecionado' in locals(): st.stop()
    selected_group = experiment_registry[experiment_registry["experimento"] == exp_selecionado]
    if selected_group.empty: st.stop()
    group_row = selected_group.iloc[0]

    st.subheader(f"Detalhamento Profundo: Experimento {exp_selecionado}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Grupo", group_row['group_display_name'])
    c2.metric("Vazão Fixada", group_row['flow_setting'])
    c3.metric("Temp. Quente", group_row['hot_side_inlet_setting'])

    detail_agilent = agilent_domain_df[agilent_domain_df["experiment_id"] == exp_selecionado]
    detail_balance = balance_domain_df[balance_domain_df["experiment_id"] == exp_selecionado]
    
    if detail_agilent.empty or detail_balance.empty:
        st.warning("Este experimento não possui dados suficientes no domínio de regime.")
    else:
        st.caption(f"**Janela:** `{detail_agilent['timestamp'].min()}` até `{detail_agilent['timestamp'].max()}`")
        
        metrics = []
        for s in ["T1", "T2", "T3", "T4", "P1", "P2"]:
            if s in detail_agilent: metrics.append({"Métrica": s, "Valor Estável (Média)": detail_agilent[s].mean()})
        if not detail_balance.empty:
            tr = fit_time_trend(detail_balance, "timestamp", "Leitura")
            metrics.append({"Métrica": "Taxa de Permeado (Regressão Linear)", "Valor Estável (Média)": f"{tr['slope_per_min']:.4f} g/min"})
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

        fig_temp, fig_press, fig_perm = plot_experiment_detail(detail_agilent, detail_balance)
        st.plotly_chart(fig_temp, use_container_width=True)
        st.plotly_chart(fig_press, use_container_width=True)
        st.plotly_chart(fig_perm, use_container_width=True)