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
from scipy.stats import chi2, linregress, t, shapiro, levene

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
    coefficient_a: Dict[str, float]
    coefficient_b: Dict[str, float]
    u_I: Dict[str, float]
    u_A: Dict[str, float]
    u_B: Dict[str, float]
    cov_AB: Dict[str, float]

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
    
    # Inicializa com as novas variáveis estatísticas
    pressure_calibration = PressureCalibration(
        coefficient_a={}, coefficient_b={}, u_I={}, u_A={}, u_B={}, cov_AB={}
    )

    if PRESSURE_CALIBRATION_FILE.exists():
        pressure_df = pd.read_csv(PRESSURE_CALIBRATION_FILE)
        for _, row in pressure_df.iterrows():
            inst = str(row["instrumento"]).split('.')[0].strip()
            
            pressure_calibration.coefficient_a[inst] = float(row["A"])
            pressure_calibration.coefficient_b[inst] = float(row["B"])
            pressure_calibration.u_I[inst] = float(row["u_I_mA"])
            pressure_calibration.u_A[inst] = float(row["u_A"])
            pressure_calibration.u_B[inst] = float(row["u_B"])
            pressure_calibration.cov_AB[inst] = float(row["cov_AB"])

    for _, row in sensor_map.iterrows():
        code = str(row["referencia_output_agilent"])
        canonical = str(row["referencia_tratamento_dados"])
        title = str(row["titulo_tratamento_dados"])
        instrument = str(row["instrumento"])
        
        sensor_code_to_name[code] = canonical
        sensor_titles[canonical] = title
        
        if instrument in uncertainty_lookup:
            # Adequação INMETRO: Divide por 2 para obter a Incerteza Padrão (Tipo B)
            base_uncertainty = float(uncertainty_lookup[instrument]) / 2.0
            
            # Como a pressão agora é dinâmica, guardamos apenas as temperaturas no dicionário
            if not canonical.startswith("P"):
                sensor_uncertainties[canonical] = base_uncertainty

    return sensor_code_to_name, sensor_titles, sensor_uncertainties, pressure_calibration

def _load_agilent_file(path: Path, experiment_id: str, replicate_group: str, sensor_code_to_name: Dict[str, str], pressure_calibration: PressureCalibration) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", skiprows=14)
    measurement_columns = {col: sensor_code_to_name[m.group(1)] for col in df.columns if (m := re.match(r"^(\d+)\s+<.*>", str(col))) and m.group(1) in sensor_code_to_name}
    df = df.rename(columns=measurement_columns)
    df["timestamp"] = _parse_agilent_timestamp(df["Time"])
    
    for column in measurement_columns.values():
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # ==========================================
    # TRATAMENTO DE PRESSÃO E INCERTEZA (GUM)
    # ==========================================
    for column in [c for c in measurement_columns.values() if c.startswith("P")]:
        instrument = "143025" if column == "P1" else "143026" if column == "P2" else None
        chaves_limpas = {str(k).strip(): k for k in pressure_calibration.coefficient_a.keys()}
        
        if instrument and instrument in chaves_limpas:
            chave_real = chaves_limpas[instrument]
            A = pressure_calibration.coefficient_a[chave_real]
            B = pressure_calibration.coefficient_b[chave_real]
            u_I = pressure_calibration.u_I[chave_real]
            u_A = pressure_calibration.u_A[chave_real]
            u_B = pressure_calibration.u_B[chave_real]
            cov_AB = pressure_calibration.cov_AB[chave_real]
            
            I_mA = df[column] * 1000.0
            P_kPa = A * I_mA + B
            
            # Cálculo GUM com proteção contra ruído transiente negativo
            radicand = (A**2 * u_I**2) + (I_mA**2 * u_A**2) + (u_B**2) + (2 * I_mA * cov_AB)
            u_P_kPa = np.sqrt(np.clip(radicand, 0, None))
            
            df[column] = P_kPa * 1000.0
            df[f"{column}_u_type_b"] = u_P_kPa * 1000.0 
        else:
            df[column] = np.nan
            df[f"{column}_u_type_b"] = np.nan

    # ==========================================
    # BUG FIX: Protegendo as incertezas dinâmicas geradas!
    # ==========================================
    keep_columns = ["timestamp"] + list(measurement_columns.values())
    
    for col in list(measurement_columns.values()):
        if f"{col}_u_type_b" in df.columns:
            keep_columns.append(f"{col}_u_type_b") # Salva a coluna do apagamento!
            
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

def compute_steady_state_domains(df: pd.DataFrame, experiment_col: str, time_col: str, temperature_cols: List[str], temp_variation_limit: float, rolling_window: int, max_drift_rate: float) -> List[DomainInterval]:
    domains = []
    for experiment_id, group in df.groupby(experiment_col):
        ordered = group.sort_values(time_col).reset_index(drop=True)
        available_temperature_cols = [column for column in temperature_cols if column in ordered.columns]
        
        # Falhou por falta de sensores
        if len(available_temperature_cols) != len(temperature_cols): 
            domains.append(DomainInterval(str(experiment_id), pd.NaT, pd.NaT, 0))
            continue

        rolling_ranges = pd.DataFrame(index=ordered.index)
        rolling_drifts = pd.DataFrame(index=ordered.index)
        half_w = max(1, rolling_window // 2)

        for column in available_temperature_cols:
            rolling_ranges[column] = ordered[column].rolling(window=rolling_window, min_periods=rolling_window).apply(lambda x: float(np.max(x) - np.min(x)), raw=True)
            mean_recent = ordered[column].rolling(window=half_w, min_periods=half_w).mean()
            mean_older = mean_recent.shift(half_w)
            delta_time_min = ordered[time_col].diff(periods=half_w).dt.total_seconds() / 60.0
            delta_time_min = delta_time_min.replace(0, np.nan)
            rolling_drifts[column] = (mean_recent - mean_older).abs() / delta_time_min

        steady_mask = (rolling_ranges <= temp_variation_limit).all(axis=1) & (rolling_drifts <= max_drift_rate).all(axis=1)
        
        block = find_longest_true_block(steady_mask)
        
        # Falhou por falta de bloco estável
        if block is None: 
            domains.append(DomainInterval(str(experiment_id), pd.NaT, pd.NaT, 0))
            continue
            
        # ==========================================
        # NORMALIZAÇÃO DO DOMÍNIO PARA VARIÂNCIA JUSTA
        # ==========================================
        end_idx = block[1]
        start_idx = max(0, end_idx - rolling_window + 1)
        pontos_validos = end_idx - start_idx + 1

        domains.append(DomainInterval(
            str(experiment_id), 
            pd.to_datetime(ordered.loc[start_idx, time_col]), 
            pd.to_datetime(ordered.loc[end_idx, time_col]), 
            pontos_validos
        ))
        
    return domains

def filter_by_domains(df: pd.DataFrame, domains: List[DomainInterval], experiment_col: str, time_col: str) -> pd.DataFrame:
    # FILTRA: Só aplica o domínio se ele de fato tiver pontos válidos (>0)
    domain_map = {d.experiment_id: d for d in domains if d.points > 0}
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
    
    # Definimos os sensores possíveis de procurar na bancada
    sensores_bancada = ["T1", "T2", "T3", "T4", "P1", "P2"]

    for (exp_id, grp_id), group in df.groupby([experiment_col, group_col]):
        base = {"experiment_id": exp_id, "replicate_group": grp_id}
        
        for sensor in sensores_bancada:
            if sensor in group.columns:
                u_a, dof_a = standard_uncertainty_type_a(group[sensor])
                base[f"{sensor}_mean"] = float(group[sensor].mean())
                base[f"{sensor}_u_type_a"] = u_a
                
                # --- IDENTIFICA DE ONDE VEM A INCERTEZA TIPO B ---
                if f"{sensor}_u_type_b" in group.columns:
                    # É pressão! Pega a média da coluna dinâmica calculada no GUM
                    u_b = float(group[f"{sensor}_u_type_b"].mean())
                else:
                    # É temperatura! Pega o valor estático do dicionário
                    u_b = sensor_uncertainties.get(sensor, 0.0)
                    
                base[f"{sensor}_u_type_b"] = u_b
                
                # Incerteza Combinada e Expandida
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

def mad_outlier_test(grouped: pd.DataFrame, group_col: str, value_col: str, threshold: float = 3.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifica outliers usando o Desvio Absoluto da Mediana (MAD-e) modificado.
    Método estatístico robusto não-paramétrico, que NÃO exige normalidade dos resíduos.
    Recomendado por Iglewicz e Hoaglin para amostras pequenas com distribuições desconhecidas.
    """
    accepted_frames, rejected_rows = [], []
    
    for group_name, group in grouped.groupby(group_col):
        working = group.copy().reset_index(drop=True)
        
        # Só testa se houver 3 ou mais réplicas. N=1 ou N=2 não têm poder estatístico para detectar outliers.
        if len(working) >= 3:
            values = working[value_col].to_numpy(dtype=float)
            
            # Calcula a Mediana do grupo (robusta contra o outlier)
            median_val = np.median(values)
            
            # Calcula o Desvio Absoluto de cada ponto para a Mediana
            abs_deviations = np.abs(values - median_val)
            
            # Calcula a Mediana desses Desvios (MAD)
            mad = np.median(abs_deviations)
            
            # Se o MAD for zero (pontos idênticos), não há outlier
            if mad == 0:
                accepted_frames.append(working)
                continue
                
            # Calcula o Z-Score Modificado (O fator 0.6745 escala o MAD para uma variância teórica comparável)
            modified_z_scores = (0.6745 * abs_deviations) / mad
            
            # Se o Z-score for maior que o limiar (padrão literatura: 3.5), é outlier
            outlier_mask = modified_z_scores > threshold
            
            if outlier_mask.any():
                # Separa os rejeitados
                rejected = working[outlier_mask].copy()
                rejected["mad_z_score"] = modified_z_scores[outlier_mask]
                rejected["mad_threshold"] = threshold
                rejected_rows.append(rejected)
                
                # Mantém os aceitos
                working = working[~outlier_mask].copy()
        
        accepted_frames.append(working)
        
    accepted_df = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else grouped.iloc[0:0]
    rejected_df = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else grouped.iloc[0:0]
    
    return accepted_df, rejected_df

def run_normality_and_homoscedasticity_tests(accepted_df: pd.DataFrame, group_col: str, permeate_col: str) -> dict:
    """Executa o Teste de Shapiro-Wilk (normalidade) e o Teste de Levene (homoscedasticidade)."""
    df_temp = accepted_df.copy()
    
    # Extrai a temperatura para agrupar
    df_temp['temp_level'] = df_temp[group_col].astype(str).str.extract(r'\|\s*(\d+(?:\.\d+)?)\s*°?C?')[0].astype(float)
    
    # 1. Teste de Shapiro-Wilk nos resíduos
    df_temp['residual'] = df_temp.groupby(group_col)[permeate_col].transform(lambda x: x - x.mean())
    residuals = df_temp['residual'].dropna().values
    
    if len(residuals) >= 3:
        shapiro_stat, shapiro_p = shapiro(residuals)
    else:
        shapiro_stat, shapiro_p = 0.0, 1.0
        
    # 2. Teste de Levene por faixa de temperatura (usando 'median', mais robusto)
    samples = [group[permeate_col].dropna().values for _, group in df_temp.groupby('temp_level') if len(group.dropna()) > 1]
    
    if len(samples) < 2:
        levene_stat, levene_p = 0.0, 1.0
    else:
        levene_stat, levene_p = levene(*samples, center='median')
        
    return {
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "is_normal": shapiro_p > 0.05,
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "is_homoscedastic": levene_p > 0.05
    }

def plot_variance_by_temperature(accepted_df: pd.DataFrame, group_col: str, permeate_col: str) -> go.Figure:
    """Gera um gráfico de barras comparando a variância e o desvio padrão em cada temperatura."""
    df_temp = accepted_df.copy()
    df_temp['temp_level'] = df_temp[group_col].astype(str).str.extract(r'\|\s*(\d+(?:\.\d+)?)\s*°?C?')[0].astype(float)
    
    stats = df_temp.groupby('temp_level')[permeate_col].agg(
        variance=lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else 0.0,
        std_dev='std',
        count='count'
    ).reset_index()
    
    fig = px.bar(
        stats, x='temp_level', y='variance', text=stats['variance'].round(4),
        title="Dispersão da Taxa de Permeado por Faixa de Temperatura (Evidência Física)",
        labels={'temp_level': 'Temperatura de Entrada Lado Quente (°C)', 'variance': 'Variância Amostral ($s^2$)'},
        template='plotly_white'
    )
    fig.update_traces(textposition='outside', marker_color='indianred')
    fig.update_layout(
        yaxis_title="Variância (g/min)²",
        xaxis_title="Temperatura (°C)",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def calculate_stratified_uncertainty(accepted_df: pd.DataFrame, group_col: str, permeate_col: str, confidence_level: float = 0.95) -> pd.DataFrame:
    """Calcula a incerteza Tipo A Estratificada por Temperatura com Fallback Global para n=1 isolados."""
    
    # Extrai a temperatura do nome do grupo
    temp_series = accepted_df[group_col].astype(str).str.extract(r'\|\s*(\d+(?:\.\d+)?)\s*°?C?')[0]
    accepted_df['temp_level'] = pd.to_numeric(temp_series, errors='coerce')
    
    # Passo 1: Estatísticas brutas de cada grupo
    group_stats = accepted_df.groupby([group_col, 'temp_level']).agg(
        permeate_rate_mean=(permeate_col, 'mean'),
        var_i=(permeate_col, lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan),
        n_instances=(permeate_col, 'count')
    ).reset_index()
    
    group_stats['df_i'] = group_stats['n_instances'] - 1
    
    # Passo 2: Variância agrupada (Pooled) por faixa de temperatura
    replicated = group_stats[group_stats['df_i'] > 0]
    pooled_data = []
    for temp, t_group in replicated.groupby('temp_level'):
        df_T = int(t_group['df_i'].sum())
        if df_T > 0:
            var_pooled = float(np.sum(t_group['df_i'] * t_group['var_i']) / df_T)
        else:
            var_pooled = np.nan
        pooled_data.append({'temp_level': temp, 'var_pooled': var_pooled, 'df_T': df_T})
        
    pooled_df = pd.DataFrame(pooled_data)
    
    if not pooled_df.empty:
        report = group_stats.merge(pooled_df, on='temp_level', how='left')
    else:
        report = group_stats.copy()
        report['var_pooled'] = np.nan
        report['df_T'] = 0
        
    # FALLBACK GLOBAL: Se uma temperatura inteira não tiver nenhuma réplica, 
    # usamos a média de todas as variâncias disponíveis na bancada (ou Qui-Quadrado global).
    global_mean_var = pooled_df['var_pooled'].mean() if not pooled_df.empty else 0.01
    global_df_total = int(pooled_df['df_T'].sum()) if not pooled_df.empty else 1
    
    # Passo 3: Propagação da Incerteza com suporte a patamares vazios
    rows = []
    for _, row in report.iterrows():
        var_p = row['var_pooled']
        df_p = row['df_T']
        n = row['n_instances']
        mean_val = row['permeate_rate_mean']
        
        # Se a temperatura não tem réplica própria, aplica o fallback estatístico global
        if pd.isna(var_p) or df_p <= 0:
            var_p = global_mean_var
            df_p = max(1, global_df_total)
            
        u_proxy = float(np.sqrt(var_p))
        u_c = float(np.sqrt(var_p / n))
        dof_eff = df_p
        
        k = float(t.ppf((1 + confidence_level) / 2, df=dof_eff)) if dof_eff >= 1 else 2.00
        u_expanded = k * u_c
        
        incerteza_relativa = (u_expanded / mean_val * 100) if mean_val != 0 else np.nan
        
        rows.append({
            group_col: row[group_col],
            "n_instances": n,
            "permeate_rate_mean": mean_val,
            "u_proxy_type_a": u_proxy,
            "u_combined_standard": u_c,
            "dof_eff": dof_eff,
            "k_factor": k,
            "reported_uncertainty": u_expanded,
            "incerteza_relativa_perc": incerteza_relativa
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

def build_final_report(uncertainty_summary: pd.DataFrame, sensors_summary: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    group_metadata = registry.drop_duplicates(subset=["replicate_group"])[
        ["replicate_group", "group_display_name", "experiments_label", "flow_setting", "hot_side_inlet_setting", "flow_ml_min", "hot_temp_c"]
    ].copy()
    
    report = uncertainty_summary.merge(group_metadata, on="replicate_group", how="left")
    
    if not sensors_summary.empty:
        report = report.merge(sensors_summary, on="replicate_group", how="left")
    
    report["report_type"] = np.where(report["n_instances"] > 1, "grupo com réplicas", "experimento único")
    report["report_label"] = report["group_display_name"]
    
    keep_columns = [
        "replicate_group", "report_label", "report_type", "experiments_label",
        "flow_setting", "hot_side_inlet_setting", "flow_ml_min", "hot_temp_c",
        "permeate_rate_mean", "reported_uncertainty", "incerteza_relativa_perc", 
        "n_instances", "u_proxy_type_a", "u_combined_standard", "dof_eff", "k_factor"
    ]
    
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
    temp_variation_limit = st.number_input("Var. máx. temp. (°C)", min_value=0.0, value=2.00, step=0.50)
    rolling_window = st.number_input("Janela estabilidade (pontos)", min_value=3, value=300, step=60)

    max_drift_rate = st.number_input("Taxa máx. deriva (°C/min)", min_value=0.001, max_value=2.0, value=0.03, step=0.01)
    
    mad_threshold = st.number_input("Corte do Outlier (MAD Z-Score)", min_value=1.5, max_value=5.0, value=3.5, step=0.5, 
                                    help="Use 3.5 para alta rigidez (ISO). Valores menores ejetam mais facilmente.")

    confidence_level = st.number_input("Nível de Confiança", min_value=0.80, max_value=0.999, value=0.95, step=0.01)
    
    st.divider()
    view_mode = st.radio("Navegação", ["Visão Geral", "Detalhe do Experimento"])
    
    if view_mode == "Detalhe do Experimento":
        exp_selecionado = st.selectbox("Selecione o Experimento", options=experiment_registry["experimento"].tolist() if not experiment_registry.empty else [])

# Core de Processamento de Dados
agilent_df = ensure_replicate_group(workspace.agilent.copy())
balance_df = ensure_replicate_group(workspace.balance.copy())

domains = compute_steady_state_domains(
    agilent_df, "experiment_id", "timestamp", steady_state_temperature_cols, 
    temp_variation_limit, int(rolling_window), float(max_drift_rate)
)

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

#teste não-paramétrico:
accepted_instances, rejected_instances = mad_outlier_test(instance_summary, "replicate_group", "permeate_rate", threshold=3.5)

# EXECUTAR TESTES DE PRESSUPOSISTOS (Shapiro-Wilk + Bartlett)
stats_validation = run_normality_and_homoscedasticity_tests(accepted_instances, "replicate_group", "permeate_rate")

# Cálculo da Incerteza Estratificada por Temperatura
uncertainty_summary = calculate_stratified_uncertainty(accepted_instances, "replicate_group", "permeate_rate", float(confidence_level))

# Consolidação dos Sensores para a tabela final
sensors_summary = aggregate_sensors(accepted_instances, sensor_columns, float(confidence_level))
final_report = build_final_report(uncertainty_summary, sensors_summary, experiment_registry)

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
       with tab2:
        st.subheader("Outliers Detectados (Teste MAD Modificado)")
        if not rejected_instances.empty:
            st.error(f"Foram rejeitados {len(rejected_instances)} experimento(s) baseados na taxa calculada.")
            outliers_view = rejected_instances[["experiment_id", "replicate_group", "permeate_rate", "mad_z_score", "mad_threshold"]]
            outliers_view = outliers_view.rename(columns={
                "permeate_rate": "Taxa Rejeitada (g/min)", 
                "mad_z_score": "Z-Score Modificado (MAD)", 
                "mad_threshold": "Limite Crítico"
            })
            st.dataframe(outliers_view, use_container_width=True, hide_index=True)
        else:
            st.success("Nenhum experimento foi caracterizado como anomalia (outlier) pelo teste do MAD.")

        st.divider()
        
        st.subheader("Domínios de Regime Permanente")
        
        # Converte para DataFrame
        domain_df = pd.DataFrame([d.__dict__ for d in domains])
        
        # Cria uma coluna numérica temporária para forçar a ordenação matemática (1, 2, 3...)
        domain_df["exp_num"] = pd.to_numeric(domain_df["experiment_id"], errors="coerce")
        domain_df = domain_df.sort_values("exp_num").drop(columns=["exp_num"])
        
        # Adiciona a coluna de Status
        domain_df["status"] = np.where(domain_df["points"] > 0, "✔️ Atingido", "❌ Falhou")
        
        # Formata as datas para não mostrar o "NaT" feio no Streamlit
        domain_df["start_time"] = domain_df["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("-")
        domain_df["end_time"] = domain_df["end_time"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("-")
        
        # Reorganiza as colunas para ficar visualmente amigável
        domain_df = domain_df[["experiment_id", "status", "start_time", "end_time", "points"]]
        
        st.dataframe(domain_df, use_container_width=True, hide_index=True)

        st.divider()
       
        st.subheader("Validação de Pressupostos Estatísticos")
        st.caption("Verificação da normalidade (Shapiro-Wilk) e da homoscedasticidade (Teste de Levene, adequado para dados não-normais).")
        
        # Bloco de Métricas do Shapiro-Wilk (Normalidade)
        st.markdown("##### 1. Teste de Normalidade dos Resíduos (Shapiro-Wilk)")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Estatística W", f"{stats_validation['shapiro_stat']:.4f}")
        col_s2.metric("Valor-p (p-value)", f"{stats_validation['shapiro_p']:.4f}")
        status_norm = "✅ Normal (p > 0.05)" if stats_validation['is_normal'] else "⚠️ Não-normal (p <= 0.05 - Justifica o uso de Levene)"
        col_s3.metric("Conclusão Normalidade", status_norm)
        
        st.divider()
        
        # Bloco de Métricas do Levene (Homoscedasticidade)
        st.markdown("##### 2. Teste de Homoscedasticidade (Teste de Levene)")
        col_l1, col_l2, col_l3 = st.columns(3)
        col_l1.metric("Estatística de Levene", f"{stats_validation['levene_stat']:.2f}")
        col_l2.metric("Valor-p (p-value)", f"{stats_validation['levene_p']:.4f}")
        status_homo = "✅ Homoscedástico" if stats_validation['is_homoscedastic'] else "⚠️ Heteroscedástico (Estratificação Térmica Necessária)"
        col_l3.metric("Conclusão Variância", status_homo)
        
        if not stats_validation['is_homoscedastic']:
            st.info(
                "💡 **Justificativa Metodológica:** Como os dados não atendem à premissa de normalidade estrita, o Teste de Levene (centrado na mediana) "
                "foi empregado com sucesso. O resultado (p < 0,05) confirma estatisticamente a heteroscedasticidade e valida a adoção do "
                "**Agrupamento Estratificado por Temperatura** na propagação de incertezas."
            )
        
        # Exibe o Gráfico de Barras da Variância por Temperatura
        st.divider()
        fig_var = plot_variance_by_temperature(accepted_instances, "replicate_group", "permeate_rate")
        st.plotly_chart(fig_var, use_container_width=True)
        
        st.divider()
        st.subheader("Propagação das Incertezas (GUM - Estratificada)")
        st.dataframe(uncertainty_summary, use_container_width=True, hide_index=True)
        
    with tab3:
        st.subheader("Tabelas Estruturais")
        with st.expander("Mapeamento de Sensores"): st.dataframe(pd.DataFrame([{"Sensor": k, "Incerteza padrão": v} for k, v in workspace.sensor_uncertainties.items()]), use_container_width=True, hide_index=True)
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