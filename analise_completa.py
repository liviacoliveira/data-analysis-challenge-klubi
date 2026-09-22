"""
Análise Completa — Hábitos e Desempenho Estudantil (Klubi)
==========================================================
Script unificado com todas as etapas do desafio de Analytics Engineer.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ── Configurações visuais globais ─────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150

# ==============================================================================
# TAREFA 1: EXPLORAÇÃO INICIAL
# ==============================================================================
def tarefa_1_exploracao(df):
    print("\n" + "="*70)
    print("TAREFA 1 — EXPLORAÇÃO INICIAL E DIAGNÓSTICO")
    print("="*70)
    
    out_dir = "outputs/01_exploracao"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1.1 Descrevendo a base
    print(f"\n📐 Dimensões originais: {df.shape[0]} linhas × {df.shape[1]} colunas")
    
    print("\n── Tipos de Dados ──")
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Numéricas ({len(numericas)}): {numericas}")
    print(f"Categóricas ({len(categoricas)}): {categoricas}")
    
    # 1.2 Diagnóstico de Qualidade (Valores ausentes e Inconsistências)
    print("\n── Diagnóstico de Valores Ausentes ──")
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if nulos.empty:
        print("✅ Nenhum valor ausente encontrado!")
    else:
        for col, count in nulos.items():
            print(f"⚠️ {col}: {count} nulos ({count/len(df)*100:.1f}%)")
            
    print("\n── Verificações de Consistência ──")
    dup = df.duplicated().sum()
    print(f"Linhas duplicadas: {dup}")
    
    return df

# ==============================================================================
# TAREFA 2: ENGENHARIA DE DADOS
# ==============================================================================
def tarefa_2_engenharia(df):
    print("\n" + "="*70)
    print("TAREFA 2 — ENGENHARIA DE DADOS")
    print("="*70)
    
    out_dir = "outputs/02_engenharia"
    os.makedirs(out_dir, exist_ok=True)
    
    df_transformed = df.copy()
    
    # 2.1 Tratamento de dados ausentes
    print("\n── Tratamento de Valores Ausentes ──")
    col_nula = "parental_education_level"
    if df_transformed[col_nula].isnull().any():
        moda = df_transformed[col_nula].mode()[0]
        df_transformed[col_nula] = df_transformed[col_nula].fillna(moda)
        print(f"✅ Preenchidos nulos de '{col_nula}' com a moda: '{moda}'")
        print("   (Justificativa: Variável ordinal com baixa % de nulos [9.1%]. A moda preserva a distribuição sem ruído extra.)")

    # 2.2 Criação de variáveis derivadas
    print("\n── Criação de Variáveis Derivadas ──")
    
    # a) Faixa Redes Sociais
    q33, q66 = df_transformed["social_media_hours"].quantile([0.33, 0.66])
    df_transformed["faixa_redes_sociais"] = pd.cut(
        df_transformed["social_media_hours"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Low", "Medium", "High"]
    )
    print("📌 Criada: faixa_redes_sociais (Low, Medium, High)")

    # b) Horas totais de distração (redes sociais + netflix)
    df_transformed["horas_distracao"] = df_transformed["social_media_hours"] + df_transformed["netflix_hours"]
    print("📌 Criada: horas_distracao")

    # c) Razão Estudo/Distração
    df_transformed["razao_estudo_distracao"] = df_transformed["study_hours_per_day"] / df_transformed["horas_distracao"].replace(0, np.nan)
    df_transformed["razao_estudo_distracao"] = df_transformed["razao_estudo_distracao"].fillna(0)
    print("📌 Criada: razao_estudo_distracao (horas estudo ÷ horas distração)")

    # d) Faixa Desempenho
    df_transformed["faixa_desempenho"] = pd.cut(
        df_transformed["exam_score"],
        bins=[-np.inf, 50, 75, np.inf],
        labels=["Baixo", "Médio", "Alto"]
    )
    print("📌 Criada: faixa_desempenho (Baixo <50, Médio 50-75, Alto >75)")

    # e) Classificação Ordinal
    mapa_dieta = {"Poor": 1, "Fair": 2, "Good": 3}
    mapa_internet = {"Poor": 1, "Average": 2, "Good": 3}
    mapa_pais = {"High School": 1, "Bachelor": 2, "Master": 3}
    
    df_transformed["diet_quality_cod"] = df_transformed["diet_quality"].map(mapa_dieta)
    df_transformed["internet_quality_cod"] = df_transformed["internet_quality"].map(mapa_internet)
    df_transformed["parental_education_level_cod"] = df_transformed["parental_education_level"].map(mapa_pais)
    print("📌 Criadas: variáveis ordinais codificadas para (dieta, internet, escolaridade pais)")

    # Salvar dataset limpo
    output_csv = "dados_transformados.csv"
    df_transformed.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset transformado salvo como '{output_csv}' ({df_transformed.shape[1]} colunas)")
    
    return df_transformed

# ==============================================================================
# FLUXO PRINCIPAL (MAIN)
# ==============================================================================
def main():
    print("Iniciando pipeline de análise de dados...")
    
    # Carregar dados originais
    file_path = "habitos_e_desempenho_estudantil.csv"
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return
        
    df_raw = pd.read_csv(file_path)
    
    # Executar Tarefa 1
    tarefa_1_exploracao(df_raw)
    
    # Executar Tarefa 2
    df_limpo = tarefa_2_engenharia(df_raw)
    
    # As Tarefas 3, 4, 5 e 6 serão adicionadas nas próximas etapas!
    print("\n" + "="*70)
    print("🚀 Pipeline executado com sucesso até a Tarefa 2!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
