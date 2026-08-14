import numpy as np
import statsmodels.api as sm
import pandas as pd

def calcular_parametros_sensor(instrumento, y_ref, x_c1_up, x_c1_dn, x_c2_up, x_c2_dn, inc_exp, k):
    """
    Função que reconstrói a regressão linear de um certificado de calibração
    e extrai os parâmetros de incerteza exigidos pelo GUM.
    """
    y_all = np.tile(y_ref, 4)
    x_all = np.array(x_c1_up + x_c1_dn + x_c2_up + x_c2_dn)

    # Regressão OLS: Y = A*X + B
    X_matrix = sm.add_constant(x_all)
    modelo = sm.OLS(y_all, X_matrix).fit()

    B = modelo.params[0] # Intercepto
    A = modelo.params[1] # Inclinação

    # Matriz de covariância dos parâmetros
    cov_matrix = modelo.cov_params()
    u_B = np.sqrt(cov_matrix[0, 0])
    u_A = np.sqrt(cov_matrix[1, 1])
    cov_AB = cov_matrix[0, 1]

    # Incerteza do equipamento convertida para incerteza padrão
    u_I = inc_exp / k

    return {
        'instrumento': instrumento,
        'A': f"{A:.8f}", 'B': f"{B:.8f}",
        'u_I_mA': f"{u_I:.4f}",
        'u_A': f"{u_A:.8f}", 'u_B': f"{u_B:.8f}",
        'cov_AB': f"{cov_AB:.8f}"
    }

# ==========================================
# 1. DADOS DO SENSOR P1 (143025)
# ==========================================
y_ref = [0.000, 7.507, 15.011, 22.515, 30.019]

res_p1 = calcular_parametros_sensor(
    "143025", y_ref,
    [4.001, 7.999, 12.002, 16.001, 20.002], # 1º Ciclo Crescente
    [4.000, 8.000, 12.001, 16.002, 20.002], # 1º Ciclo Decrescente
    [3.999, 7.999, 12.002, 16.000, 20.001], # 2º Ciclo Crescente
    [4.001, 7.998, 12.000, 16.001, 20.001], # 2º Ciclo Decrescente
    inc_exp=0.007, k=2.00
)

# ==========================================
# 2. DADOS DO SENSOR P2 (143026)
# ==========================================
res_p2 = calcular_parametros_sensor(
    "143026", y_ref,
    [4.003, 8.004, 12.003, 16.000, 19.998], # 1º Ciclo Crescente
    [4.001, 8.002, 12.001, 15.998, 19.998], # 1º Ciclo Decrescente
    [4.002, 8.003, 12.002, 15.999, 19.999], # 2º Ciclo Crescente
    [4.000, 8.001, 12.000, 15.997, 19.999], # 2º Ciclo Decrescente
    inc_exp=0.007, k=2.00
)

# ==========================================
# 3. EXPORTAR PARA CSV
# ==========================================
df = pd.DataFrame([res_p1, res_p2])
df.to_csv('./arquivos_auxiliares/relacao_corrente_pressao.csv', index=False)
print("Arquivo gerado com sucesso!")