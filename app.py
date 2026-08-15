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
GRAVIMETRIC_MAP_FILE = AUX_DIR / "mapeamento_bomba_gravimetrica.csv"

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

def apply_pump_calibration(agilent_df: pd.DataFrame, experiment_metadata: pd.DataFrame) -> pd.DataFrame:
    """Busca a vazão gravimétrica real da bomba com base na data do experimento do Agilent."""
    meta = experiment_metadata.copy()
    meta["flow_real_ml_min"] = meta["flow_ml_min"]
    meta["flow_u_bomba"] = 0.0
    
    if not GRAVIMETRIC_MAP_FILE.exists():
        return meta
        
    calib_df = pd.read_csv(GRAVIMETRIC_MAP_FILE)
    # Converte as datas das pastas (ex: '08_05_2026') para datetime
    calib_df['data_calib'] = pd.to_datetime(calib_df['Data_Teste'], format='%d_%m_%Y')
    
    # Extrai o Timestamp exato em que cada experimento iniciou
    exp_dates = agilent_df.groupby('experiment_id')['timestamp'].min().reset_index()
    exp_dates = exp_dates.rename(columns={'timestamp': 'exp_date'})
    
    # Cruza a tabela mestre com as datas
    meta = meta.merge(exp_dates, left_on='experimento', right_on='experiment_id', how='left')
    
    for idx, row in meta.iterrows():
        exp_date = row['exp_date']
        nominal = row['flow_ml_min']
        
        if pd.isna(exp_date) or pd.isna(nominal): continue
            
        # Pega todas as calibrações que ocorreram ANTES ou NO MESMO DIA do experimento
        valid_calibs = calib_df[calib_df['data_calib'] <= exp_date]
        
        # Se o ensaio for mais antigo que a 1ª calibração, usa a 1ª como referência
        if valid_calibs.empty: valid_calibs = calib_df
            
        # Pega a calibração válida mais recente
        target_date = valid_calibs['data_calib'].max()
        
        # Encontra a curva exata para o nível nominal (800, 1000 ou 1200)
        match = calib_df[(calib_df['data_calib'] == target_date) & (calib_df['Vazao_Alvo_Nominal'] == nominal)]
        
        if not match.empty:
            meta.at[idx, 'flow_real_ml_min'] = match.iloc[0]['Vazao_Real_Media']
            meta.at[idx, 'flow_u_bomba'] = match.iloc[0]['Incerteza_uA_Bomba']
            
    return meta.drop(columns=['experiment_id', 'exp_date'], errors='ignore')

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
    
    # --- NOVO BLOCO (Substitui o antigo) ---
    if not agilent_df.empty:
        # Chama a mágica da calibração
        experiment_metadata = apply_pump_calibration(agilent_df, experiment_metadata)
        
        metadata_for_merge = experiment_metadata[["experimento", "flow_setting", "hot_side_inlet_setting", "flow_real_ml_min"]].copy()
        
        agilent_df = agilent_df.merge(metadata_for_merge, left_on="experiment_id", right_on="experimento", how="left")
        if not balance_df.empty: 
            balance_df = balance_df.merge(metadata_for_merge, left_on="experiment_id", right_on="experimento", how="left")
    # ---------------------------------------

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

def water_density_g_ml(T: pd.Series) -> pd.Series:
    """
    Calcula a densidade da água líquida (g/mL) em função da temperatura (°C)
    usando a formulação de Kell (1975).
    """
    num = (999.83952 + 16.945176 * T - 7.9870401e-3 * T**2 - 
           46.170461e-6 * T**3 + 105.56302e-9 * T**4 - 280.54253e-12 * T**5)
    den = 1 + 16.897850e-3 * T
    return (num / den) / 1000.0  # Retorna em g/mL (equivalente a kg/L)

def calculate_energy_metrics(report_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula STEC, GOR, propagação de incerteza expandida (GUM) e incerteza percentual."""
    df = report_df.copy()
    
    # Constantes termodinâmicas
    cp_water = 4.184  # Calor específico da água (J/g·°C)
    h_fg = 2257.0     # Calor latente de vaporização da água (J/g)
    
    # 1. Extração dos Valores Nominais
    V_h = pd.to_numeric(df.get("flow_real_mean", np.nan), errors="coerce")
    m_p = pd.to_numeric(df.get("permeate_rate_mean", np.nan), errors="coerce") 
    T_in = pd.to_numeric(df.get("T1_mean", np.nan), errors="coerce") 
    T_out = pd.to_numeric(df.get("T2_mean", np.nan), errors="coerce") 
    
    # 2. Extração das Incertezas Expandidas
    u_V_h = pd.to_numeric(df.get("flow_real_u_expanded", 0.0), errors="coerce")
    u_m_p = pd.to_numeric(df.get("reported_uncertainty", 0.0), errors="coerce")
    u_T_in = pd.to_numeric(df.get("T1_u_expanded", 0.0), errors="coerce")
    u_T_out = pd.to_numeric(df.get("T2_u_expanded", 0.0), errors="coerce")

    # 3. Correção Dinâmica de Densidade (Avaliando na temperatura média do canal)
    T_avg = (T_in + T_out) / 2.0
    rho = water_density_g_ml(T_avg)
    
    m_h = V_h * rho
    u_m_h = u_V_h * rho
    
    # 4. Cálculos Nominais de Balanço
    delta_T = T_in - T_out
    Q_in = m_h * cp_water * delta_T
    
    df["GOR"] = np.where((Q_in > 0) & (m_p > 0), (m_p * h_fg) / Q_in, np.nan)
    df["STEC_kWh_m3"] = np.where((m_p > 0), Q_in / (m_p * 3.6), np.nan)
    df["rho_avg_g_ml"] = rho
    
    # 5. Propagação de Incerteza Relativa e Percentual
    u_delta_T = np.sqrt(u_T_in**2 + u_T_out**2)
    
    rel_unc = np.sqrt(
        (u_m_h / m_h)**2 + 
        (u_m_p / m_p)**2 + 
        (u_delta_T / delta_T)**2
    )
    
    df["GOR_u_expanded"] = df["GOR"] * rel_unc
    df["STEC_u_expanded"] = df["STEC_kWh_m3"] * rel_unc
    
    # Nova coluna de incerteza percentual unificada para as métricas energéticas
    df["energy_relative_unc_perc"] = rel_unc * 100
    
    return df

def build_final_report(uncertainty_summary: pd.DataFrame, sensors_summary: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    group_metadata = registry.drop_duplicates(subset=["replicate_group"])[
        ["replicate_group", "group_display_name", "experiments_label", "flow_setting", "hot_side_inlet_setting", "flow_ml_min", "hot_temp_c"]
    ].copy()
    
    # ==========================================
    # CÁLCULO GUM: Agrupamento da Vazão Real
    # ==========================================
    flow_stats_list = []
    for group_name, group in registry.groupby("replicate_group"):
        n = len(group)
        real_flows = group["flow_real_ml_min"].dropna().to_numpy()
        u_bombas = group["flow_u_bomba"].dropna().to_numpy() # Incertezas Padrão da calibração
        
        if len(real_flows) == 0:
            flow_stats_list.append({"replicate_group": group_name, "flow_real_mean": np.nan, "flow_real_u_expanded": np.nan})
            continue
            
        mean_flow = float(np.mean(real_flows))
        
        # Tipo A: Variação da vazão real entre as réplicas
        if len(real_flows) > 1:
            u_A_flow = float(np.std(real_flows, ddof=1) / np.sqrt(n))
            dof_A = n - 1
        else:
            u_A_flow = 0.0
            dof_A = 0
            
        # Tipo B: Incerteza combinada (RMS) das calibrações da bomba
        u_B_flow = float(np.sqrt(np.mean(u_bombas**2))) if len(u_bombas) > 0 else 0.0
        
        # Incerteza Combinada Padrão e Graus de Liberdade (Welch-Satterthwaite)
        u_c_flow = np.sqrt(u_A_flow**2 + u_B_flow**2)
        
        if u_A_flow > 0:
            dof_eff = (u_c_flow**4) / ((u_A_flow**4) / dof_A)
            k_flow = float(t.ppf((1 + 0.95) / 2, df=dof_eff)) if dof_eff >= 1 else 2.00
        else:
            k_flow = 2.00
            
        flow_stats_list.append({
            "replicate_group": group_name,
            "flow_real_mean": round(mean_flow, 2),
            "flow_real_u_expanded": round(k_flow * u_c_flow, 4)
        })
        
    flow_stats_df = pd.DataFrame(flow_stats_list)
    group_metadata = group_metadata.merge(flow_stats_df, on="replicate_group", how="left")
    
    # Mescla tudo
    report = uncertainty_summary.merge(group_metadata, on="replicate_group", how="left")
    if not sensors_summary.empty:
        report = report.merge(sensors_summary, on="replicate_group", how="left")
        
    report["report_type"] = np.where(report["n_instances"] > 1, "grupo com réplicas", "experimento único")
    report["report_label"] = report["group_display_name"]
    
   # ==========================================
    # ORGANIZAÇÃO FINAL DAS COLUNAS (LÓGICA INTERNA)
    # ==========================================
    keep_columns = [
        "replicate_group", "report_label", "report_type", "experiments_label",
        "flow_setting", "hot_side_inlet_setting", "hot_temp_c", "flow_ml_min", # <-- Mantidos para o gráfico usar!
        "flow_real_mean", "flow_real_u_expanded", 
        "permeate_rate_mean", "reported_uncertainty", "incerteza_relativa_perc", 
        "n_instances", "u_proxy_type_a", "u_combined_standard", "dof_eff", "k_factor"
    ]
    
    for col in sensors_summary.columns:
        if col != "replicate_group" and col not in keep_columns:
            keep_columns.append(col)
            
    for col in keep_columns:
        if col not in report.columns: report[col] = np.nan
            
    # Retorna o dataframe original para NÃO quebrar os gráficos!
    report = report.sort_values(["flow_real_mean", "hot_temp_c", "replicate_group"], na_position="last")
    return report[keep_columns].reset_index(drop=True)

# ==========================================
# GERAÇÃO DE GRÁFICOS (2D LINHAS)
# ==========================================
# Mude a definição da função para receber as listas selecionadas
def plot_2d_faceted_lines(report_df: pd.DataFrame, sel_temps: list, sel_flows: list) -> Tuple[go.Figure, go.Figure]:
    plot_df = report_df.copy()
    
    plot_df["reported_uncertainty"] = pd.to_numeric(plot_df["reported_uncertainty"], errors="coerce")
    plot_df["permeate_rate_mean"] = pd.to_numeric(plot_df["permeate_rate_mean"], errors="coerce")
    plot_df["flow_real_mean"] = pd.to_numeric(plot_df["flow_real_mean"], errors="coerce")
    plot_df["flow_real_u_expanded"] = pd.to_numeric(plot_df["flow_real_u_expanded"], errors="coerce")
    plot_df["hot_temp_c"] = pd.to_numeric(plot_df["hot_temp_c"], errors="coerce")
    plot_df["flow_ml_min"] = pd.to_numeric(plot_df["flow_ml_min"], errors="coerce")
    
    plot_df = plot_df.dropna(subset=["permeate_rate_mean", "flow_real_mean", "hot_temp_c"])
    
    # CRIAR DATAFRAMES SEPARADOS E FILTRADOS PARA CADA GRÁFICO
    df_g1 = plot_df[plot_df["hot_temp_c"].isin(sel_temps)].copy()
    df_g2 = plot_df[plot_df["flow_ml_min"].isin(sel_flows)].copy()
    
    # Aplica Labels no G1
    df_g1["Temp_Label"] = df_g1["hot_temp_c"].apply(lambda x: f"{x:02.0f} °C (Target)")
    temp_order = sorted(df_g1["Temp_Label"].unique())

    # ==========================================
    # Gráfico 1: Usando df_g1
    # ==========================================
    fig_flow = px.line(
        df_g1.sort_values(["flow_real_mean", "hot_temp_c"]),
        x="flow_real_mean", 
        y="permeate_rate_mean", 
        color="Temp_Label",
        symbol="Temp_Label", 
        markers=True, 
        error_y="reported_uncertainty", 
        error_x="flow_real_u_expanded", 
        hover_data={"flow_ml_min": True},
        category_orders={"Temp_Label": temp_order}, 
        title="Effect of Measured Flow Rate on Permeate Flux",
        labels={
            "flow_real_mean": "Measured Gravimetric Flow Rate (mL/min)", 
            "permeate_rate_mean": "Permeate Flux (g/min)", 
            "Temp_Label": "Target Temperature",
            "flow_ml_min": "Target Flow (Nominal)"
        },
        template="simple_white",
        color_discrete_sequence=px.colors.sequential.Plasma_r 
    )
    fig_flow.update_traces(
        error_y=dict(thickness=1.5, width=4, color='rgba(100,100,100,0.5)'),
        error_x=dict(thickness=1.5, width=4, color='rgba(100,100,100,0.5)'),
        marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey'))
    )
    fig_flow.update_layout(font=dict(family="Times New Roman", size=14), legend=dict(title_font_family="Times New Roman"), margin=dict(l=10, r=10, t=50, b=10))

    # Aplica Labels no G2 (Ordem matemática real sem o zero na frente)
    numeric_flows = sorted(df_g2["flow_ml_min"].unique())
    flow_order = [f"{x:.0f} mL/min (Target)" for x in numeric_flows]
    df_g2["Flow_Label"] = df_g2["flow_ml_min"].apply(lambda x: f"{x:.0f} mL/min (Target)")

    # ==========================================
    # Gráfico 2: Usando df_g2
    # ==========================================
    if "T1_mean" in df_g2.columns and "T1_u_expanded" in df_g2.columns:
        x_col, error_x_col, x_label = "T1_mean", "T1_u_expanded", "Measured Hot Inlet Temperature - T1 (°C)"
        df_g2[x_col] = pd.to_numeric(df_g2[x_col], errors="coerce")
        df_g2[error_x_col] = pd.to_numeric(df_g2[error_x_col], errors="coerce")
    else:
        x_col, error_x_col, x_label = "hot_temp_c", None, "Target Hot Inlet Temperature (°C)"

    fig_temp = px.line(
        df_g2.sort_values(["Flow_Label", x_col]),
        x=x_col, 
        y="permeate_rate_mean", 
        color="Flow_Label",
        symbol="Flow_Label", 
        markers=True, 
        error_y="reported_uncertainty",
        error_x=error_x_col, 
        hover_data={"hot_temp_c": True, "flow_real_mean": ":.1f", "flow_real_u_expanded": ":.2f", "Flow_Label": False}, 
        category_orders={"Flow_Label": flow_order},
        title="Effect of Measured Temperature on Permeate Flux",
        labels={x_col: x_label, "permeate_rate_mean": "Permeate Flux (g/min)", "Flow_Label": "Target Flow Rate", "hot_temp_c": "Target Temp.", "flow_real_mean": "Real Flow", "flow_real_u_expanded": "Flow Unc. (±)"},
        template="simple_white",
        color_discrete_sequence=px.colors.sequential.Viridis_r 
    )
    fig_temp.update_traces(
        error_y=dict(thickness=1.5, width=4, color='rgba(100,100,100,0.5)'),
        error_x=dict(thickness=1.5, width=4, color='rgba(100,100,100,0.5)'),
        marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey'))
    )
    fig_temp.update_layout(font=dict(family="Times New Roman", size=14), legend=dict(title_font_family="Times New Roman"), margin=dict(l=10, r=10, t=50, b=10))

    return fig_flow, fig_temp

def plot_covariance_matrix(report_df: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    # Adicionando T3 (Entrada Fria) e T4 (Saída Fria)
    cols_interesse = ["flow_real_mean", "T1_mean", "T2_mean", "T3_mean", "T4_mean", "P1_mean", "P2_mean", "permeate_rate_mean"]
    avail_cols = [c for c in cols_interesse if c in report_df.columns]
    
    df_calc = report_df[avail_cols].dropna()
    
    # Dicionário com os símbolos de Delta (Δ) matemáticos
    rename_dict = {
        "flow_real_mean": "Flow Rate",
        "T1_mean": "T Inlet Hot Side",
        "T2_mean": "T outlet Hot Side",
        "T3_mean": "T Inlet Cold Side",
        "T4_mean": "T outlet Cold Side",
        "P1_mean": "ΔP Hot Side",   
        "P2_mean": "ΔP Cold Side",  
        "permeate_rate_mean": "Permeate Flux"
    }
    df_calc = df_calc.rename(columns=rename_dict)
    
    cov_matrix = df_calc.cov()
    corr_matrix = df_calc.corr()
    
    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale="RdBu_r", 
        zmin=-1, zmax=1,
        title="Pearson Correlation Matrix of Stabilized Variables",
        labels=dict(color="Correlation")
    )
    fig.update_layout(
        template="simple_white", 
        font=dict(family="Times New Roman", size=13),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig, corr_matrix, cov_matrix

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
    # 1. Removida a 4ª Aba (Raw Data)
    tab1, tab2, tab3 = st.tabs([
        "📊 Trends & Results", 
        "🛠️ Steady-State & Outliers", 
        "🧩 Covariance Matrix"
    ])
    
    with tab1:
        st.subheader("AGMD Performance Curves")
        st.caption(f"Error bars represent Expanded Uncertainty (**{confidence_level*100:.0f}%** confidence).")
        
        # 2. CÓDIGO DOS FILTROS DINÂMICOS
        # Levanta as opções disponíveis que existem no DataFrame
        available_temps = sorted(final_report["hot_temp_c"].dropna().unique())
        available_flows = sorted(final_report["flow_ml_min"].dropna().unique())
        
        # Cria duas colunas para os botões ficarem lado a lado em cima dos gráficos
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_temps = st.multiselect(
                "Select Target Temperatures (Left Graph):", 
                options=available_temps, 
                default=available_temps # Por padrão, todos começam ativados
            )
        with col_f2:
            sel_flows = st.multiselect(
                "Select Target Flows (Right Graph):", 
                options=available_flows, 
                default=available_flows
            )
        
        # Gera os gráficos passando apenas o que o usuário selecionou nas caixas!
        fig_flow_2d, fig_temp_2d = plot_2d_faceted_lines(final_report, sel_temps, sel_flows)
        
        col_graf_1, col_graf_2 = st.columns(2)
        with col_graf_1: st.plotly_chart(fig_flow_2d, use_container_width=True)
        with col_graf_2: st.plotly_chart(fig_temp_2d, use_container_width=True)
        
        st.divider()
        st.subheader("Consolidated Table (Permeate Flux, Temperatures, and Pressures)")
        
        rename_dict = {
            "replicate_group": "Group ID", "report_label": "Label", "report_type": "Data Type",
            "experiments_label": "Valid Experiments", "flow_setting": "Target Flow",
            "hot_side_inlet_setting": "Target T_in", "flow_real_mean": "Real Flow (mL/min)",
            "flow_real_u_expanded": "Flow Exp. Unc. (±)", "permeate_rate_mean": "Permeate Flux (g/min)",
            "reported_uncertainty": "Flux Exp. Unc. (±)", "incerteza_relativa_perc": "Relative Unc. (%)",
            "n_instances": "Valid Replicates (n)", "u_proxy_type_a": "Std. Unc. Type A",
            "u_combined_standard": "Comb. Std. Unc. (uc)", "dof_eff": "Effective DOF", "k_factor": "Coverage Factor (k)"
        }
        display_df = final_report.drop(columns=["hot_temp_c", "flow_ml_min"]).rename(columns=rename_dict)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Energy Metrics Overview (STEC & GOR)")
        st.caption("Derived from steady-state hot side enthalpy drop, with propagated GUM uncertainty.")
        
        energy_report = calculate_energy_metrics(final_report)
        
        rename_energy = {
            "replicate_group": "Group ID",
            "T1_mean": "T_in Hot (°C)",
            "T2_mean": "T_out Hot (°C)",
            "flow_real_mean": "Vol. Flow (mL/min)",
            "permeate_rate_mean": "Permeate Flux (g/min)",
            "GOR": "GOR (-)",
            "GOR_u_expanded": "GOR Unc. (±)",
            "STEC_kWh_m3": "STEC (kWh/m³)",
            "STEC_u_expanded": "STEC Unc. (±)",
            "energy_relative_unc_perc": "Relative Unc. (%)"
        }
        
        display_energy = energy_report[[c for c in rename_energy.keys() if c in energy_report.columns]].copy()
        display_energy = display_energy.rename(columns=rename_energy)
        
        st.dataframe(display_energy.style.format(precision=3), use_container_width=True, hide_index=True)
        
    with tab2:
        st.subheader("Detected Outliers (Modified MAD Test)")
        if not rejected_instances.empty:
            st.error(f"{len(rejected_instances)} experiment(s) were rejected based on flux anomaly.")
            outliers_view = rejected_instances[["experiment_id", "replicate_group", "permeate_rate", "mad_z_score", "mad_threshold"]]
            outliers_view = outliers_view.rename(columns={
                "experiment_id": "Exp ID", "replicate_group": "Group",
                "permeate_rate": "Rejected Flux (g/min)", 
                "mad_z_score": "Modified Z-Score (MAD)", 
                "mad_threshold": "Critical Limit"
            })
            st.dataframe(outliers_view, use_container_width=True, hide_index=True)
        else:
            st.success("No experiments were characterized as outliers by the MAD test.")

        st.divider()
        st.subheader("Steady-State Domains")
        domain_df = pd.DataFrame([d.__dict__ for d in domains])
        domain_df["exp_num"] = pd.to_numeric(domain_df["experiment_id"], errors="coerce")
        domain_df = domain_df.sort_values("exp_num").drop(columns=["exp_num"])
        domain_df["status"] = np.where(domain_df["points"] > 0, "✔️ Achieved", "❌ Failed")
        domain_df["start_time"] = domain_df["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("-")
        domain_df["end_time"] = domain_df["end_time"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("-")
        domain_df = domain_df[["experiment_id", "status", "start_time", "end_time", "points"]].rename(
            columns={"experiment_id": "Exp ID", "status": "Status", "start_time": "Start Time", "end_time": "End Time", "points": "Data Points"}
        )
        st.dataframe(domain_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Statistical Assumptions Validation")
        st.markdown("##### 1. Normality of Residuals (Shapiro-Wilk)")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("W Statistic", f"{stats_validation['shapiro_stat']:.4f}")
        col_s2.metric("p-value", f"{stats_validation['shapiro_p']:.4f}")
        col_s3.metric("Conclusion", "✅ Normal (p > 0.05)" if stats_validation['is_normal'] else "⚠️ Non-normal (p ≤ 0.05)")
        
        st.markdown("##### 2. Homoscedasticity (Levene's Test)")
        col_l1, col_l2, col_l3 = st.columns(3)
        col_l1.metric("Levene Statistic", f"{stats_validation['levene_stat']:.2f}")
        col_l2.metric("p-value", f"{stats_validation['levene_p']:.4f}")
        col_l3.metric("Conclusion", "✅ Homoscedastic" if stats_validation['is_homoscedastic'] else "⚠️ Heteroscedastic")

        st.divider()
        st.markdown("##### Permeate Flux Variance by Temperature Range (Physical Evidence)")
        
        # 1. Cria um mapa ligando o nome do grupo à temperatura nominal dele
        temp_mapping = experiment_registry[["replicate_group", "hot_temp_c"]].drop_duplicates()
        
        # 2. Cruza as taxas de permeado validadas com as temperaturas
        var_data = accepted_instances.merge(temp_mapping, on="replicate_group", how="left")
        
        # 3. Calcula a variância (var) agrupando apenas pela temperatura
        var_df = var_data.groupby("hot_temp_c")["permeate_rate"].var().reset_index()
        var_df = var_df.dropna()
        
        if not var_df.empty:
            fig_var = px.bar(
                var_df,
                x="hot_temp_c",
                y="permeate_rate",
                text="permeate_rate",
                labels={
                    "hot_temp_c": "Target Temperature (°C)", 
                    "permeate_rate": "Variance (g/min)²"
                },
                template="simple_white"
            )
            fig_var.update_traces(
                texttemplate='%{text:.4f}', 
                textposition='outside', 
                marker_color='#d62728' # Vermelho padrão metrológico
            )
            fig_var.update_layout(
                font=dict(family="Times New Roman", size=14),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(type='category') # Força o X a tratar as temperaturas como categorias (barras separadas)
            )
            st.plotly_chart(fig_var, use_container_width=True)
        else:
            st.warning("Não foi possível calcular a variância. Verifique os dados de entrada.")

    with tab3: 
        st.subheader("Correlation & Covariance Analysis")
        st.caption("Investigates the linear interdependence and physical coupling between inputs and permeate flux.")
        
        fig_corr, corr_df, cov_df = plot_covariance_matrix(final_report)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.divider()
        st.markdown("##### Experimental Covariance Matrix")
        st.dataframe(cov_df.style.format("{:.4f}"), use_container_width=True)

elif view_mode == "Detalhe do Experimento":
        st.subheader("Single Experiment Analysis")
        st.caption("Detailed view of steady-state behavior and stability.")
        
        # 1. Puxa os IDs de forma segura
        if "experimento" in experiment_registry.columns:
            raw_ids = experiment_registry["experimento"].dropna().unique()
        else:
            raw_ids = experiment_registry["experiment_id"].dropna().unique()
            
        # 2. Força a conversão para Inteiro para manter a ordem (1, 2, 3...)
        exp_list = sorted([int(x) for x in raw_ids if str(x).strip().isdigit()])
        selected_exp = st.selectbox("Select Experiment ID:", exp_list)
        
        if selected_exp:
            sel_exp_str = str(selected_exp)
            num_exp = int(selected_exp)
            st.markdown(f"### Steady-State Data for Exp: **{selected_exp}**")
            
            # --- 1. Tabela de Resumo do Experimento ---
            st.markdown("##### Steady-State Aggregated Results")
            import re
            mask = final_report["experiments_label"].astype(str).apply(lambda x: bool(re.search(rf'\b{sel_exp_str}\b', x)))
            exp_summary = final_report[mask]
            
            if not exp_summary.empty:
                rename_dict_detail = {
                    "replicate_group": "Group ID",
                    "flow_real_mean": "Real Flow (mL/min)",
                    "flow_real_u_expanded": "Flow Unc. (±)",
                    "permeate_rate_mean": "Permeate Flux (g/min)",
                    "reported_uncertainty": "Flux Unc. (±)",
                    "T1_mean": "T1_in Hot (°C)",
                    "T1_u_expanded": "T1 Unc. (±)",
                    "T2_mean": "T2_out Hot (°C)",
                    "T3_mean": "T3_in Cold (°C)",
                    "T4_mean": "T4_out Cold (°C)",
                    "P1_mean": "ΔP Hot Side",
                    "P2_mean": "ΔP Cold Side"
                }
                
                cols_to_show = [c for c in rename_dict_detail.keys() if c in exp_summary.columns]
                disp_summary = exp_summary[cols_to_show].rename(columns=rename_dict_detail)
                st.dataframe(disp_summary.style.format(precision=4), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # --- 2. Filtro Exclusivo do Domínio de Regime Permanente ---
            # Identifica as colunas de ID nos dataframes de domínio filtrado
            ag_col = "experiment_id" if "experiment_id" in agilent_domain_df.columns else "experimento"
            bal_col = "experiment_id" if "experiment_id" in balance_domain_df.columns else "experimento"
            
            # Puxa apenas os dados estabilizados (matematicamente convertidos para evitar erros)
            detail_agilent = agilent_domain_df[pd.to_numeric(agilent_domain_df[ag_col], errors='coerce') == num_exp]
            detail_balance = balance_domain_df[pd.to_numeric(balance_domain_df[bal_col], errors='coerce') == num_exp]
            
            if detail_agilent.empty or detail_balance.empty:
                st.warning("No steady-state domain data available for this experiment. It may have failed the stabilization criteria.")
            else:
                start_time = detail_agilent['timestamp'].min()
                end_time = detail_agilent['timestamp'].max()
                st.caption(f"**Stabilized Window:** `{start_time}` to `{end_time}`")
                
                # --- 3. Gráficos de Série Temporal Filtrada (Padrão Artigo) ---
                
                # GRÁFICO 1: TEMPERATURAS
                temp_cols = [c for c in ["T1", "T2", "T3", "T4"] if c in detail_agilent.columns]
                if temp_cols:
                    fig_t = px.line(
                        detail_agilent, 
                        x="timestamp", 
                        y=temp_cols,
                        title="Thermal Steady-State Profiles",
                        labels={"timestamp": "Time (Local)", "value": "Temperature (°C)", "variable": "Thermocouple"},
                        template="simple_white",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig_t.update_layout(
                        font=dict(family="Times New Roman", size=14), 
                        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_t, use_container_width=True)

                # GRÁFICO 2: PRESSÕES
                press_cols = [c for c in ["P1", "P2"] if c in detail_agilent.columns]
                if press_cols:
                    fig_p = px.line(
                        detail_agilent, 
                        x="timestamp", 
                        y=press_cols,
                        title="Hydrodynamic Steady-State Profiles",
                        labels={"timestamp": "Time (Local)", "value": "Pressure (Pa)", "variable": "Transducer"},
                        template="simple_white",
                        color_discrete_sequence=px.colors.qualitative.Dark2
                    )
                    fig_p.update_layout(
                        font=dict(family="Times New Roman", size=14), 
                        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

                # GRÁFICO 3: MASSA DA BALANÇA
                mass_col = "Leitura" if "Leitura" in detail_balance.columns else "mass" if "mass" in detail_balance.columns else detail_balance.columns[-1]
                
                fig_m = px.scatter(
                    detail_balance, 
                    x="timestamp", 
                    y=mass_col,
                    title="Permeate Mass Accumulation (Regression Zone)",
                    labels={"timestamp": "Time (Local)", mass_col: "Accumulated Mass (g)"},
                    template="simple_white",
                    color_discrete_sequence=["#d62728"]
                )
                fig_m.update_traces(marker=dict(size=6, opacity=0.8))
                fig_m.update_layout(font=dict(family="Times New Roman", size=14))
                st.plotly_chart(fig_m, use_container_width=True)