import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from pathlib import Path
import re

# ---------------------------------------------------------
# 1. MAPEAMENTO DINÂMICO DE DIRETÓRIOS E ARQUIVOS
# ---------------------------------------------------------
BASE_DIR = Path('dados_analise_graviometrica')
OUTPUT_DIR = Path('arquivos_auxiliares')
OUTPUT_DIR.mkdir(exist_ok=True)

resultados_testes = []

print("="*85)
print("Iniciando varredura Gravimétrica Ponto a Ponto...")
print("="*85)

if not BASE_DIR.exists():
    raise FileNotFoundError(f"A pasta '{BASE_DIR}' não foi encontrada.")

# Varredura dos arquivos
for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir(): continue
    data_teste = folder.name
    
    for file_path in folder.glob('*.csv'):
        # Extrai: (Curva) _ teste _ (Número) _ (Vazão digitada na bomba)
        match = re.match(r'(\d+)_teste_(\d+)_(\d+)', file_path.stem)
        
        if not match: continue
            
        curva_id = int(match.group(1))
        teste_id = int(match.group(2))
        vazao_arquivo = int(match.group(3)) # Ex: 727, 868, 1014
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8', sep=',')
        except Exception as e:
            continue
            
        df['Leitura'] = df['Leitura'].astype(str).str.replace('+', '', regex=False).str.strip()
        df['Leitura'] = pd.to_numeric(df['Leitura'], errors='coerce')
        
        df['DataHora'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'])
        df['Tempo_Segundos'] = (df['DataHora'] - df['DataHora'].iloc[0]).dt.total_seconds()
        
        df['Derivada'] = df['Leitura'].diff()
        inicio = df[df['Derivada'] > 1.0].index.min()
        if pd.isna(inicio): inicio = 0
            
        t0 = df.loc[inicio, 'Tempo_Segundos']
        df['Tempo_Sincronizado'] = df['Tempo_Segundos'] - t0
        
        # REGRESSÃO VOLUME vs TEMPO (Para achar a Vazão Real daquele teste)
        df_linear = df[(df['Tempo_Sincronizado'] >= 50) & (df['Tempo_Sincronizado'] <= 100)]
        
        if len(df_linear) > 1:
            t_min = df_linear['Tempo_Sincronizado'] / 60.0
            slope, intercept, r2, _, std_err = linregress(t_min, df_linear['Leitura'])
            
            resultados_testes.append({
                'Data_Teste': data_teste,
                'Curva_ID': curva_id,
                'Teste_ID': teste_id,
                'Vazao_Painel': vazao_arquivo,
                'Vazao_Exp_Real': slope
            })

df_testes = pd.DataFrame(resultados_testes)

# ---------------------------------------------------------
# 2. LÓGICA DE ASSOCIAÇÃO COM A VAZÃO ALVO NOMINAL
# ---------------------------------------------------------
# Sabemos que os testes correspondem a 800, 1000 e 1200 em ordem crescente
alvos_padrao = [800, 1000, 1200]
df_testes['Vazao_Alvo_Nominal'] = np.nan

for data_teste, group in df_testes.groupby('Data_Teste'):
    # Pega as vazões únicas digitadas na bomba (Ex: [727, 868, 1014]) e ordena
    vazoes_painel_unicas = sorted(group['Vazao_Painel'].unique())
    
    if len(vazoes_painel_unicas) <= 3:
        # Cria um dicionário casando a ordem (menor=800, meio=1000, maior=1200)
        map_dict = {v: alvos_padrao[i] for i, v in enumerate(vazoes_painel_unicas)}
        
        mask = df_testes['Data_Teste'] == data_teste
        df_testes.loc[mask, 'Vazao_Alvo_Nominal'] = df_testes.loc[mask, 'Vazao_Painel'].map(map_dict)

# ---------------------------------------------------------
# 3. CONSOLIDAÇÃO E CÁLCULO DE INCERTEZA (GUM)
# ---------------------------------------------------------
# Agora agrupamos pelas 3 réplicas e tiramos a média e a incerteza
resumo = df_testes.groupby(['Data_Teste', 'Curva_ID', 'Vazao_Alvo_Nominal']).agg(
    Vazao_Real_Media=('Vazao_Exp_Real', 'mean'),
    Desvio_Padrao_Amostral=('Vazao_Exp_Real', lambda x: np.std(x, ddof=1) if len(x) > 1 else 0),
    Qtd_Testes=('Teste_ID', 'count')
).reset_index()

# Incerteza Padrão Tipo A da Bomba = Desvio Padrão / sqrt(n)
resumo['Incerteza_uA_Bomba'] = resumo['Desvio_Padrao_Amostral'] / np.sqrt(resumo['Qtd_Testes'])

# Arredondamento para estética
resumo['Vazao_Real_Media'] = resumo['Vazao_Real_Media'].round(2)
resumo['Incerteza_uA_Bomba'] = resumo['Incerteza_uA_Bomba'].round(4)

arquivo_saida = OUTPUT_DIR / 'mapeamento_bomba_gravimetrica.csv'
resumo.to_csv(arquivo_saida, index=False)
print(f"✅ Tabela de mapeamento exportada em: {arquivo_saida}\n")

# ---------------------------------------------------------
# 4. EXIBIÇÃO TERMINAL E GRÁFICO
# ---------------------------------------------------------
print(resumo[['Data_Teste', 'Vazao_Alvo_Nominal', 'Vazao_Real_Media', 'Incerteza_uA_Bomba', 'Qtd_Testes']].to_string(index=False))

fig = go.Figure()
cores = {800: '#1f77b4', 1000: '#ff7f0e', 1200: '#2ca02c'}

for data_teste, group in resumo.groupby('Data_Teste'):
    fig.add_trace(go.Bar(
        x=group['Vazao_Alvo_Nominal'].astype(str) + " mL/min",
        y=group['Vazao_Real_Media'],
        name=f"Curva de {data_teste.replace('_', '/')}",
        error_y=dict(type='data', array=group['Incerteza_uA_Bomba']*2, visible=True), # Barra de erro com k=2
        text=group['Vazao_Real_Media'].astype(str),
        textposition='auto'
    ))

fig.update_layout(
    title='<b>Vazão Real Média por Nível Nominal (Com Incerteza Expandida)</b>',
    xaxis_title='<b>Vazão Nominal Alvo</b>',
    yaxis_title='<b>Vazão Experimental Gravimétrica (mL/min)</b>',
    barmode='group',
    template='plotly_white'
)
fig.show()