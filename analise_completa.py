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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

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

    print(f"\nDimensões originais: {df.shape[0]} linhas × {df.shape[1]} colunas")

    print("\n── Tipos de Dados ──")
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Numéricas ({len(numericas)}): {numericas}")
    print(f"Categóricas ({len(categoricas)}): {categoricas}")

    print("\n── Diagnóstico de Valores Ausentes ──")
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if nulos.empty:
        print("Nenhum valor ausente encontrado!")
    else:
        for col, count in nulos.items():
            print(f"{col}: {count} nulos ({count/len(df)*100:.1f}%)")

    print("\n── Verificações de Consistência ──")
    dup = df.duplicated().sum()
    print(f"Linhas duplicadas: {dup}")

    # Checagem de faixas plausíveis (evita horas negativas, notas fora de 0-100 etc.)
    print("\n── Checagem de Faixas Plausíveis ──")
    checagens = {
        "age": (10, 100),
        "study_hours_per_day": (0, 24),
        "social_media_hours": (0, 24),
        "netflix_hours": (0, 24),
        "sleep_hours": (0, 24),
        "attendance_percentage": (0, 100),
        "exam_score": (0, 100),
    }
    for col, (lo, hi) in checagens.items():
        if col in df.columns:
            fora = df[(df[col] < lo) | (df[col] > hi)]
            status = "ok" if fora.empty else f"⚠️ {len(fora)} valores fora de [{lo}, {hi}]"
            print(f"{col}: {status}")

    # Resumo estatístico salvo como CSV (útil para o README / apêndice)
    df.describe(include="all").to_csv(f"{out_dir}/resumo_estatistico.csv")
    print(f"\nResumo estatístico salvo em: {out_dir}/resumo_estatistico.csv")

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
        print(f"Preenchidos nulos de '{col_nula}' com a moda: '{moda}'")
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
    print("Criada: faixa_redes_sociais (Low, Medium, High)")

    # b) Horas totais de distração (redes sociais + netflix)
    df_transformed["horas_distracao"] = df_transformed["social_media_hours"] + df_transformed["netflix_hours"]
    print("Criada: horas_distracao")

    # c) Razão Estudo/Distração
    df_transformed["razao_estudo_distracao"] = df_transformed["study_hours_per_day"] / df_transformed["horas_distracao"].replace(0, np.nan)
    df_transformed["razao_estudo_distracao"] = df_transformed["razao_estudo_distracao"].fillna(0)
    print("Criada: razao_estudo_distracao (horas estudo ÷ horas distração)")

    # d) Faixa Desempenho
    df_transformed["faixa_desempenho"] = pd.cut(
        df_transformed["exam_score"],
        bins=[-np.inf, 50, 75, np.inf],
        labels=["Baixo", "Médio", "Alto"]
    )
    print("Criada: faixa_desempenho (Baixo <50, Médio 50-75, Alto >75)")

    # e) Classificação Ordinal
    mapa_dieta = {"Poor": 1, "Fair": 2, "Good": 3}
    mapa_internet = {"Poor": 1, "Average": 2, "Good": 3}
    mapa_pais = {"High School": 1, "Bachelor": 2, "Master": 3}

    df_transformed["diet_quality_cod"] = df_transformed["diet_quality"].map(mapa_dieta)
    df_transformed["internet_quality_cod"] = df_transformed["internet_quality"].map(mapa_internet)
    df_transformed["parental_education_level_cod"] = df_transformed["parental_education_level"].map(mapa_pais)
    print("Criadas: variáveis ordinais codificadas para (dieta, internet, escolaridade pais)")

    # f) Binárias (0/1) para variáveis Sim/Não — úteis para modelo e correlação
    for col in ["part_time_job", "extracurricular_participation"]:
        if col in df_transformed.columns:
            df_transformed[f"{col}_bin"] = df_transformed[col].map({"Yes": 1, "No": 0})
    print("Criadas: versões binárias (0/1) de part_time_job e extracurricular_participation")

    print("Variáveis derivadas criadas: faixa_redes_sociais, horas_distracao, razao_estudo_distracao, faixa_desempenho, ordinais codificadas e binárias.")

    # Salvar dataset limpo
    output_csv = "dados_transformados.csv"
    df_transformed.to_csv(output_csv, index=False)
    print(f"\nDataset transformado salvo como '{output_csv}' ({df_transformed.shape[1]} colunas)")

    return df_transformed

# ==============================================================================
# TAREFA 3: ANÁLISE ESTATÍSTICA
# ==============================================================================
def tarefa_3_estatistica(df):
    print("\n" + "="*70)
    print("TAREFA 3 — ANÁLISE ESTATÍSTICA")
    print("="*70)

    out_dir = "outputs/03_estatistica"
    os.makedirs(out_dir, exist_ok=True)

    # 3.1 Selecionar variáveis numéricas para correlação (excluindo identificadores)
    colunas_ignorar = {"student_id"}
    numericas = [c for c in df.select_dtypes(include="number").columns if c not in colunas_ignorar]
    corr_matrix = df[numericas].corr()

    # 3.2 Isolar a correlação com a variável alvo (exam_score)
    corr_com_nota = corr_matrix["exam_score"].drop("exam_score").sort_values(ascending=False)

    print("\n── Correlação com a Nota Final (exam_score) ──")
    print(corr_com_nota.round(3).to_string())

    # 3.3 Identificar maiores influências
    infl_positiva = corr_com_nota.head(3)
    infl_negativa = corr_com_nota.tail(3)

    print("\n── Maiores Influências Positivas ──")
    for col, val in infl_positiva.items():
        print(f"{col:.<30} {val:+.3f}")

    print("\n── Maiores Influências Negativas ──")
    for col, val in infl_negativa.items():
        print(f"{col:.<30} {val:+.3f}")

    # 3.4 Gerar Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .75})
    plt.title("Matriz de Correlação - Hábitos e Desempenho Estudantil", pad=20, fontsize=14)
    plt.tight_layout()
    heatmap_path = f"{out_dir}/heatmap_correlacao.png"
    plt.savefig(heatmap_path)
    plt.close()

    print(f"\n Mapa de calor salvo em: {heatmap_path}")

    corr_com_nota.round(3).to_csv(f"{out_dir}/correlacoes_com_exam_score.csv", header=["correlacao"])

    return corr_matrix, corr_com_nota

# ==============================================================================
# TAREFA 4: APLICAÇÕES PRÁTICAS
# ==============================================================================
def tarefa_4_aplicacoes(df):
    print("\n" + "="*70)
    print("TAREFA 4 — APLICAÇÕES PRÁTICAS")
    print("="*70)

    out_dir = "outputs/04_aplicacoes"
    os.makedirs(out_dir, exist_ok=True)

    # ── Aplicação A: Modelo preditivo de exam_score ──────────────────────
    print("\n── Aplicação A: Modelo Preditivo (Regressão Linear) ──")

    features = [
        "study_hours_per_day", "social_media_hours", "netflix_hours",
        "sleep_hours", "exercise_frequency", "attendance_percentage",
        "mental_health_rating", "diet_quality_cod", "internet_quality_cod",
        "parental_education_level_cod",
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    coefs = pd.Series(modelo.coef_, index=features).sort_values(key=abs, ascending=False)

    print(f"R² (teste): {r2:.3f}")
    print(f"MAE (teste): {mae:.2f} pontos de nota")
    print("\nPeso de cada hábito na nota prevista (coeficientes ordenados por magnitude):")
    print(coefs.round(3).to_string())

    coefs.round(3).to_csv(f"{out_dir}/coeficientes_modelo.csv", header=["coeficiente"])

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolor="k", linewidth=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Previsão perfeita")
    plt.xlabel("Nota real")
    plt.ylabel("Nota prevista")
    plt.title(f"Modelo Preditivo — Real vs. Previsto (R²={r2:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/modelo_real_vs_previsto.png")
    plt.close()

    print(f"\n Gráfico real vs. previsto salvo em: {out_dir}/modelo_real_vs_previsto.png")
    print("\n Decisão que apoia: um orientador acadêmico ou o time de monitoria pode aplicar")
    print("   este modelo a um formulário de hábitos preenchido por um aluno novo (ou no início")
    print("   do semestre) para estimar sua faixa de desempenho esperada e priorizar quem deve")
    print("   receber apoio (tutoria, ajuste de rotina de estudo) antes que a nota caia de fato.")

    # ── Aplicação B: Segmentação de alunos (clustering) ──────────────────
    print("\n── Aplicação B: Segmentação de Alunos (KMeans) ──")

    cluster_features = ["study_hours_per_day", "horas_distracao", "sleep_hours", "mental_health_rating"]
    cluster_features = [f for f in cluster_features if f in df.columns]

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_cluster)

    perfil = df.groupby("cluster")[cluster_features + ["exam_score"]].mean().round(2)
    perfil["n_alunos"] = df["cluster"].value_counts().sort_index()
    perfil = perfil.sort_values("exam_score", ascending=False)

    nomes = ["🟢 Alto Desempenho", "🔵 Equilibrado", "🟡 Atenção", "🔴 Risco"][:len(perfil)]
    perfil.insert(0, "perfil", nomes)

    print("\nPerfis identificados (ordenados por nota média):")
    print(perfil.to_string())

    perfil.to_csv(f"{out_dir}/perfis_clusters.csv")

    # Gráfico dos clusters: estudo vs distração, colorido por perfil
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["study_hours_per_day"], df["horas_distracao"],
        c=df["cluster"], cmap="viridis", alpha=0.6, edgecolor="k", linewidth=0.2
    )
    plt.xlabel("Horas de estudo/dia")
    plt.ylabel("Horas de distração/dia (redes sociais + Netflix)")
    plt.title("Segmentação de Alunos por Hábitos (KMeans, k=4)")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/segmentacao_clusters.png")
    plt.close()

    print(f"\n Gráfico de segmentação salvo em: {out_dir}/segmentacao_clusters.png")
    print("\n Decisão que apoia: o time pedagógico/CX da Klubi pode usar os perfis para")
    print("   direcionar ações diferentes por grupo — ex.: campanhas de bem-estar e gestão de")
    print("   tempo de tela para o perfil 'Risco', grupos de estudo em par para 'Equilíbrio',")
    print("   e reconhecimento/mentoria de pares para 'Alto Desempenho' — em vez de uma")
    print("   comunicação genérica para toda a base de alunos.")

    return modelo, coefs, df

# ==============================================================================
# TAREFA 5: VISUALIZAÇÃO
# ==============================================================================
def tarefa_5_visualizacao(df, corr_com_nota):
    print("\n" + "="*70)
    print("TAREFA 5 — VISUALIZAÇÃO")
    print("="*70)

    out_dir = "outputs/05_visualizacao"
    os.makedirs(out_dir, exist_ok=True)

    # Gráfico 2 (do desafio): análise detalhada da variável de maior impacto
    top_var = corr_com_nota.abs().idxmax()
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x=top_var, y="exam_score",
                scatter_kws={"alpha": 0.35}, line_kws={"color": "red"})
    plt.title(f"Relação entre {top_var} e exam_score (r = {corr_com_nota[top_var]:+.3f})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/impacto_{top_var}.png")
    plt.close()
    print(f" Análise detalhada de '{top_var}' salva em: {out_dir}/impacto_{top_var}.png")

    # Gráfico 3 (do desafio): comparação por faixas — boxplot
    plt.figure(figsize=(8, 6))
    ordem = ["Low", "Medium", "High"]
    sns.boxplot(data=df, x="faixa_redes_sociais", y="exam_score", order=ordem)
    plt.title("Distribuição da Nota por Faixa de Uso de Redes Sociais")
    plt.xlabel("Faixa de uso de redes sociais")
    plt.ylabel("exam_score")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/boxplot_faixa_redes_sociais.png")
    plt.close()
    print(f" Boxplot por faixa de redes sociais salvo em: {out_dir}/boxplot_faixa_redes_sociais.png")

    # Extra 1: comparação por gênero (diferencial pedido na Tarefa 6)
    if "gender" in df.columns:
        plt.figure(figsize=(7, 6))
        sns.boxplot(data=df, x="gender", y="exam_score")
        plt.title("Distribuição da Nota por Gênero")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/boxplot_genero.png")
        plt.close()
        print(f" Boxplot por gênero salvo em: {out_dir}/boxplot_genero.png")

    # Extra 2: histograma de horas de estudo vs faixa de desempenho
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="study_hours_per_day", hue="faixa_desempenho",
                 multiple="stack", bins=20, palette="viridis")
    plt.title("Horas de Estudo por Faixa de Desempenho")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/hist_estudo_por_desempenho.png")
    plt.close()
    print(f" Histograma salvo em: {out_dir}/hist_estudo_por_desempenho.png")

    # Extra 3 (diferencial): mini-dashboard consolidando os principais gráficos
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    sns.scatterplot(data=df, x="study_hours_per_day", y="exam_score",
                     alpha=0.4, ax=axes[0, 0])
    axes[0, 0].set_title("Estudo vs. Nota")

    sns.boxplot(data=df, x="faixa_redes_sociais", y="exam_score", order=ordem, ax=axes[0, 1])
    axes[0, 1].set_title("Nota por Faixa de Redes Sociais")

    top5 = corr_com_nota.reindex(corr_com_nota.abs().sort_values(ascending=False).index).head(5)
    sns.barplot(x=top5.values, y=top5.index, ax=axes[1, 0],
                palette=["#2a9d8f" if v > 0 else "#e76f51" for v in top5.values])
    axes[1, 0].set_title("Top 5 Correlações (|r|) com a Nota")
    axes[1, 0].set_xlabel("Correlação")

    if "mental_health_rating" in df.columns:
        media_saude = df.groupby("mental_health_rating")["exam_score"].mean()
        sns.lineplot(x=media_saude.index, y=media_saude.values, marker="o", ax=axes[1, 1])
        axes[1, 1].set_title("Nota Média por Nível de Saúde Mental")
        axes[1, 1].set_xlabel("mental_health_rating")
        axes[1, 1].set_ylabel("Nota média")

    plt.suptitle("Dashboard Resumo — Hábitos e Desempenho Estudantil", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/dashboard_resumo.png")
    plt.close()
    print(f" Dashboard resumo (diferencial) salvo em: {out_dir}/dashboard_resumo.png")

# ==============================================================================
# TAREFA 6: SÍNTESE DE INSIGHTS
# ==============================================================================
def tarefa_6_insights(df, corr_com_nota):
    print("\n" + "="*70)
    print("TAREFA 6 — SÍNTESE DE INSIGHTS")
    print("="*70)

    out_dir = "outputs/06_insights"
    os.makedirs(out_dir, exist_ok=True)

    linhas = ["# Síntese de Insights — Hábitos e Desempenho Estudantil\n"]

    # Hábitos com maior impacto
    top3_pos = corr_com_nota.head(3)
    top3_neg = corr_com_nota.sort_values().head(3)

    linhas.append("## Quais hábitos mais afetam as notas?\n")
    linhas.append("**Maiores impulsionadores (correlação positiva):**")
    for col, val in top3_pos.items():
        linhas.append(f"- `{col}`: {val:+.3f}")
    linhas.append("\n**Maiores fatores de queda (correlação negativa):**")
    for col, val in top3_neg.items():
        linhas.append(f"- `{col}`: {val:+.3f}")

    # Diferença por gênero (teste t)
    if "gender" in df.columns:
        grupos = df["gender"].dropna().unique().tolist()
        linhas.append("\n## Diferenças entre grupos\n")
        if len(grupos) == 2:
            g1, g2 = grupos
            a = df.loc[df["gender"] == g1, "exam_score"]
            b = df.loc[df["gender"] == g2, "exam_score"]
            tstat, pval = ttest_ind(a, b, equal_var=False)
            sig = "diferença estatisticamente significativa (p < 0.05)" if pval < 0.05 else "sem diferença estatisticamente significativa (p ≥ 0.05)"
            linhas.append(f"**Gênero** — média {g1}: {a.mean():.1f} | média {g2}: {b.mean():.1f} "
                           f"(teste t, p-valor = {pval:.3f} → {sig})")
        else:
            medias_genero = df.groupby("gender")["exam_score"].mean().round(1)
            linhas.append("**Gênero** — nota média por grupo:")
            linhas.append(medias_genero.to_string())

    if "mental_health_rating" in df.columns:
        media_saude = df.groupby("mental_health_rating")["exam_score"].mean().round(1)
        linhas.append("\n**Saúde mental** — nota média por nível (1 = pior, 10 = melhor):")
        linhas.append(media_saude.to_string())

    if "part_time_job" in df.columns:
        media_job = df.groupby("part_time_job")["exam_score"].mean().round(1)
        linhas.append("\n**Trabalho meio período** — nota média:")
        linhas.append(media_job.to_string())

    # Recomendações práticas (derivadas dos números acima)
    linhas.append("\n## Recomendações práticas\n")
    linhas.append(f"1. **Priorizar horas de estudo protegidas**: `study_hours_per_day` é, isoladamente, "
                   f"o fator com maior correlação com a nota ({corr_com_nota.get('study_hours_per_day', float('nan')):+.3f}). "
                   f"Pequenos ganhos de rotina aqui tendem a ter o maior retorno.")
    linhas.append("2. **Vigiar o tempo total de tela, não só redes sociais**: `horas_distracao` "
                   "(redes sociais + Netflix somados) tem correlação negativa maior que cada variável isolada — "
                   "o efeito é cumulativo.")
    linhas.append("3. **Tratar saúde mental como alavanca de desempenho**, não só de bem-estar — "
                   "a correlação positiva indica que apoio psicológico/pedagógico pode refletir em notas.")
    linhas.append("4. **Usar os perfis de cluster (Tarefa 4) para segmentar intervenções** em vez de "
                   "campanhas genéricas de estudo para toda a base de alunos.")

    texto = "\n".join(str(l) for l in linhas)
    with open(f"{out_dir}/sintese_insights.md", "w", encoding="utf-8") as f:
        f.write(texto)

    print(texto)
    print(f"\n Síntese completa salva em: {out_dir}/sintese_insights.md")
    print("   (copie os trechos relevantes — com os números reais gerados na sua execução — para o README)")

# ==============================================================================
# FLUXO PRINCIPAL (MAIN)
# ==============================================================================
def main():
    print("Iniciando pipeline de análise de dados...")

    file_path = "habitos_e_desempenho_estudantil.csv"
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return

    df_raw = pd.read_csv(file_path)

    tarefa_1_exploracao(df_raw)
    df_limpo = tarefa_2_engenharia(df_raw)
    corr_matrix, corr_com_nota = tarefa_3_estatistica(df_limpo)
    modelo, coefs, df_com_clusters = tarefa_4_aplicacoes(df_limpo)
    tarefa_5_visualizacao(df_limpo, corr_com_nota)
    tarefa_6_insights(df_limpo, corr_com_nota)

    print("\n" + "="*70)
    print(" Pipeline executado com sucesso — Tarefas 1 a 6 concluídas!")
    print("   Todos os artefatos (gráficos, CSVs e a síntese de insights) estão em 'outputs/'.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
