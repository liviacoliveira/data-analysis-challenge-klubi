"""
01 — Exploração Inicial dos Dados
==================================
Importação, descrição, diagnóstico de qualidade e visão geral do dataset
de hábitos e desempenho estudantil.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend não-interativo para salvar figuras
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Configurações visuais ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150

OUTPUT_DIR = "outputs/01_exploracao"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. IMPORTAÇÃO                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
df = pd.read_csv("habitos_e_desempenho_estudantil.csv")

print("=" * 70)
print("EXPLORAÇÃO INICIAL — Hábitos e Desempenho Estudantil")
print("=" * 70)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. DIMENSÕES E PRIMEIRAS LINHAS                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
print(f"\n📐 Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas\n")
print("── Primeiras 5 linhas ──")
print(df.head().to_string())

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. TIPOS DE DADOS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n── Tipos de dados por coluna ──")
tipos = df.dtypes.reset_index()
tipos.columns = ["Coluna", "Tipo"]
print(tipos.to_string(index=False))

# Classificação semântica
numericas = df.select_dtypes(include="number").columns.tolist()
categoricas = df.select_dtypes(include="object").columns.tolist()
print(f"\n  → Numéricas ({len(numericas)}): {numericas}")
print(f"  → Categóricas/Texto ({len(categoricas)}): {categoricas}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. VALORES AUSENTES                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n── Diagnóstico de Valores Ausentes ──")
nulos = df.isnull().sum()
nulos_pct = (df.isnull().mean() * 100).round(2)
resumo_nulos = pd.DataFrame({
    "Ausentes": nulos,
    "% do Total": nulos_pct
}).sort_values("Ausentes", ascending=False)
resumo_nulos = resumo_nulos[resumo_nulos["Ausentes"] > 0]

if resumo_nulos.empty:
    print("  ✅ Nenhum valor ausente encontrado!")
else:
    print(resumo_nulos.to_string())

total_nulos = df.isnull().sum().sum()
total_celulas = df.shape[0] * df.shape[1]
print(f"\n  Total de células nulas: {total_nulos} / {total_celulas} "
      f"({total_nulos/total_celulas*100:.2f}%)")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. VALORES DUPLICADOS                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n── Duplicatas ──")
n_dup = df.duplicated().sum()
n_dup_id = df["student_id"].duplicated().sum()
print(f"  Linhas duplicadas (todas as colunas): {n_dup}")
print(f"  IDs de estudante duplicados: {n_dup_id}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. ESTATÍSTICAS DESCRITIVAS — VARIÁVEIS NUMÉRICAS                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n── Estatísticas Descritivas (Numéricas) ──")
desc = df[numericas].describe().T
desc["IQR"] = desc["75%"] - desc["25%"]
desc["coef_var_%"] = ((desc["std"] / desc["mean"]) * 100).round(2)
print(desc.round(2).to_string())

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7. ESTATÍSTICAS DESCRITIVAS — VARIÁVEIS CATEGÓRICAS               ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n── Distribuição das Variáveis Categóricas ──")
for col in categoricas:
    print(f"\n  📌 {col}:")
    vc = df[col].value_counts()
    vc_pct = (df[col].value_counts(normalize=True) * 100).round(1)
    resumo_cat = pd.DataFrame({"Contagem": vc, "%": vc_pct})
    print(resumo_cat.to_string())

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  8. DETECÇÃO DE OUTLIERS (IQR)                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n── Detecção de Outliers (método IQR × 1.5) ──")
outlier_report = []
for col in numericas:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report.append({
        "Coluna": col,
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2),
        "Lim. Inferior": round(lower, 2),
        "Lim. Superior": round(upper, 2),
        "N° Outliers": n_out,
        "% Outliers": round(n_out / len(df) * 100, 2),
    })
out_df = pd.DataFrame(outlier_report)
print(out_df.to_string(index=False))

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  9. VERIFICAÇÕES DE CONSISTÊNCIA                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n── Verificações de Consistência ──")

# Percentuais devem estar entre 0 e 100
if "attendance_percentage" in df.columns:
    fora = df[(df["attendance_percentage"] < 0) | (df["attendance_percentage"] > 100)]
    print(f"  attendance_percentage fora de [0, 100]: {len(fora)} registros")

# Horas por dia não devem ser negativas nem > 24
for col in ["study_hours_per_day", "social_media_hours", "netflix_hours", "sleep_hours"]:
    if col in df.columns:
        neg = (df[col] < 0).sum()
        exc = (df[col] > 24).sum()
        print(f"  {col} — negativos: {neg}, > 24h: {exc}")

# exam_score — verificar range
if "exam_score" in df.columns:
    print(f"  exam_score — min: {df['exam_score'].min()}, max: {df['exam_score'].max()}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  10. VISUALIZAÇÕES EXPLORATÓRIAS                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n── Gerando visualizações exploratórias ──")

# 10a. Histogramas das variáveis numéricas
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(numericas):
    if i < len(axes):
        df[col].hist(ax=axes[i], bins=25, edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel("Frequência")
# Remover eixos extras
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Distribuição das Variáveis Numéricas", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/histogramas_numericas.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/histogramas_numericas.png")

# 10b. Boxplots das variáveis numéricas
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(numericas):
    if i < len(axes):
        sns.boxplot(data=df, y=col, ax=axes[i], color="steelblue")
        axes[i].set_title(col, fontsize=10)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Boxplots — Identificação de Outliers", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/boxplots_numericas.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/boxplots_numericas.png")

# 10c. Contagem das variáveis categóricas
fig, axes = plt.subplots(1, len(categoricas), figsize=(5 * len(categoricas), 5))
if len(categoricas) == 1:
    axes = [axes]
for i, col in enumerate(categoricas):
    sns.countplot(data=df, x=col, ax=axes[i], palette="Set2",
                  order=df[col].value_counts().index)
    axes[i].set_title(col, fontsize=12)
    axes[i].tick_params(axis="x", rotation=30)
    # Anotar com porcentagens
    total = len(df)
    for p in axes[i].patches:
        pct = f"{p.get_height()/total*100:.1f}%"
        axes[i].annotate(pct, (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=9)
fig.suptitle("Distribuição das Variáveis Categóricas", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/categoricas_contagem.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/categoricas_contagem.png")

# 10d. Distribuição da variável-alvo (exam_score)
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["exam_score"], kde=True, bins=30, color="coral", ax=ax)
ax.axvline(df["exam_score"].mean(), color="red", ls="--", label=f'Média: {df["exam_score"].mean():.1f}')
ax.axvline(df["exam_score"].median(), color="blue", ls="--", label=f'Mediana: {df["exam_score"].median():.1f}')
ax.set_title("Distribuição do Desempenho (exam_score)", fontsize=14)
ax.set_xlabel("Nota")
ax.set_ylabel("Frequência")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/dist_exam_score.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/dist_exam_score.png")

plt.close("all")

print("\n" + "=" * 70)
print("✅ Exploração inicial concluída! Gráficos salvos em:", OUTPUT_DIR)
print("=" * 70)

