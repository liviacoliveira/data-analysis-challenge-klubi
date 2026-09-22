"""
02 — Engenharia de Dados
=========================
Tratamento de valores ausentes, criação de variáveis derivadas
e codificação de variáveis ordinais.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ── Configurações visuais ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150

OUTPUT_DIR = "outputs/02_engenharia"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. CARREGAR DADOS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
df = pd.read_csv("habitos_e_desempenho_estudantil.csv")

print("=" * 70)
print("ENGENHARIA DE DADOS — Hábitos e Desempenho Estudantil")
print("=" * 70)
print(f"\n📐 Dimensões originais: {df.shape[0]} linhas × {df.shape[1]} colunas")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. TRATAMENTO DE VALORES AUSENTES                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n" + "─" * 70)
print("2. TRATAMENTO DE VALORES AUSENTES")
print("─" * 70)

col_nula = "parental_education_level"
n_nulos = df[col_nula].isnull().sum()
print(f"\n  Coluna: {col_nula}")
print(f"  Valores ausentes: {n_nulos} ({n_nulos/len(df)*100:.1f}%)")

# Analisar a distribuição da coluna antes do tratamento
print(f"\n  Distribuição ANTES do tratamento:")
dist_antes = df[col_nula].value_counts(dropna=False)
print(f"  {dist_antes.to_dict()}")

# ── Estratégia: Imputação pela moda ──
# Justificativa: como a variável é categórica ordinal e a proporção
# de nulos é relativamente baixa (9.1%), a imputação pela moda preserva
# a distribuição original sem introduzir viés significativo.
# Alternativa considerada: criar categoria "Desconhecido", mas isso
# adicionaria ruído desnecessário em modelos preditivos.
moda = df[col_nula].mode()[0]
print(f"\n  ✅ Estratégia escolhida: imputação pela MODA ('{moda}')")
print(f"     Justificativa: variável categórica ordinal com apenas 9.1% de")
print(f"     nulos — a moda preserva a distribuição original.")

# Verificar se os nulos têm perfil diferente
print(f"\n  📊 Comparação de exam_score: nulos vs. preenchidos:")
media_nulos = df[df[col_nula].isnull()]["exam_score"].mean()
media_preenchidos = df[df[col_nula].notna()]["exam_score"].mean()
print(f"     Média (nulos):       {media_nulos:.1f}")
print(f"     Média (preenchidos): {media_preenchidos:.1f}")
print(f"     Diferença:           {abs(media_nulos - media_preenchidos):.1f} pontos")

if abs(media_nulos - media_preenchidos) < 5:
    print(f"     → Diferença pequena → ausência parece aleatória (MCAR/MAR)")
else:
    print(f"     → Diferença relevante → ausência pode não ser aleatória (MNAR)")

# Aplicar imputação
df[col_nula] = df[col_nula].fillna(moda)

print(f"\n  Distribuição DEPOIS do tratamento:")
dist_depois = df[col_nula].value_counts()
print(f"  {dist_depois.to_dict()}")
print(f"  Nulos restantes: {df[col_nula].isnull().sum()}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. CRIAÇÃO DE VARIÁVEIS DERIVADAS                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n" + "─" * 70)
print("3. CRIAÇÃO DE VARIÁVEIS DERIVADAS")
print("─" * 70)

# ── 3.1 Faixas de uso de redes sociais ──
print("\n  📌 3.1 — faixa_redes_sociais (Low / Medium / High)")
print("     Critério: tercis da distribuição de social_media_hours")
q33 = df["social_media_hours"].quantile(0.33)
q66 = df["social_media_hours"].quantile(0.66)
print(f"     Limites: Low < {q33:.1f}h | Medium {q33:.1f}–{q66:.1f}h | High > {q66:.1f}h")

df["faixa_redes_sociais"] = pd.cut(
    df["social_media_hours"],
    bins=[-np.inf, q33, q66, np.inf],
    labels=["Low", "Medium", "High"]
)
print(f"     Distribuição: {df['faixa_redes_sociais'].value_counts().to_dict()}")

# ── 3.2 Total de horas de distração ──
print("\n  📌 3.2 — horas_distracao (social_media_hours + netflix_hours)")
df["horas_distracao"] = df["social_media_hours"] + df["netflix_hours"]
print(f"     Média: {df['horas_distracao'].mean():.2f}h")
print(f"     Mediana: {df['horas_distracao'].median():.2f}h")
print(f"     Range: {df['horas_distracao'].min():.1f}–{df['horas_distracao'].max():.1f}h")

# ── 3.3 Razão estudo / distração ──
print("\n  📌 3.3 — razao_estudo_distracao (study_hours / horas_distracao)")
print("     Interpretação: >1 = estuda mais do que se distrai")
# Evitar divisão por zero
df["razao_estudo_distracao"] = df["study_hours_per_day"] / df["horas_distracao"].replace(0, np.nan)
df["razao_estudo_distracao"] = df["razao_estudo_distracao"].fillna(0)
print(f"     Média: {df['razao_estudo_distracao'].mean():.2f}")
print(f"     Mediana: {df['razao_estudo_distracao'].median():.2f}")
print(f"     % alunos que estudam mais do que se distraem: "
      f"{(df['razao_estudo_distracao'] > 1).mean() * 100:.1f}%")

# ── 3.4 Faixa de desempenho ──
print("\n  📌 3.4 — faixa_desempenho (Baixo / Médio / Alto)")
print("     Critério: <50 = Baixo | 50–75 = Médio | >75 = Alto")
df["faixa_desempenho"] = pd.cut(
    df["exam_score"],
    bins=[-np.inf, 50, 75, np.inf],
    labels=["Baixo", "Médio", "Alto"]
)
print(f"     Distribuição: {df['faixa_desempenho'].value_counts().to_dict()}")

# ── 3.5 Horas de tela total ──
print("\n  📌 3.5 — horas_tela_total (social_media + netflix)")
print("     (Sinônimo de horas_distracao — já criado acima)")

# ── 3.6 Faixa de sono ──
print("\n  📌 3.6 — faixa_sono (Insuficiente / Adequado / Excessivo)")
print("     Critério (recomendação OMS/NSF para jovens adultos):")
print("     <6h = Insuficiente | 6–9h = Adequado | >9h = Excessivo")
df["faixa_sono"] = pd.cut(
    df["sleep_hours"],
    bins=[-np.inf, 6, 9, np.inf],
    labels=["Insuficiente", "Adequado", "Excessivo"]
)
print(f"     Distribuição: {df['faixa_sono'].value_counts().to_dict()}")

# ── 3.7 Classificação de saúde mental ──
print("\n  📌 3.7 — faixa_saude_mental (Baixa / Moderada / Boa)")
print("     Critério: 1–3 = Baixa | 4–7 = Moderada | 8–10 = Boa")
df["faixa_saude_mental"] = pd.cut(
    df["mental_health_rating"],
    bins=[0, 3, 7, 10],
    labels=["Baixa", "Moderada", "Boa"],
    include_lowest=True
)
print(f"     Distribuição: {df['faixa_saude_mental'].value_counts().to_dict()}")

# ── 3.8 Nível de dedicação acadêmica (composta) ──
print("\n  📌 3.8 — dedicacao_academica (score composto)")
print("     Fórmula: normalização min-max de (study_hours + attendance + extracurricular)")

# Normalizar cada componente para [0, 1]
def minmax(s):
    return (s - s.min()) / (s.max() - s.min())

study_norm = minmax(df["study_hours_per_day"])
attend_norm = minmax(df["attendance_percentage"])
extra_norm = (df["extracurricular_participation"] == "Yes").astype(float)

df["dedicacao_academica"] = ((study_norm + attend_norm + extra_norm) / 3 * 100).round(1)
print(f"     Média: {df['dedicacao_academica'].mean():.1f}")
print(f"     Mediana: {df['dedicacao_academica'].median():.1f}")
print(f"     Range: {df['dedicacao_academica'].min():.1f}–{df['dedicacao_academica'].max():.1f}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. CODIFICAÇÃO DE VARIÁVEIS ORDINAIS                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n" + "─" * 70)
print("4. CODIFICAÇÃO DE VARIÁVEIS ORDINAIS")
print("─" * 70)

# Mapeamentos com justificativa de ordem
mapeamentos = {
    "diet_quality": {"Poor": 1, "Fair": 2, "Good": 3},
    "internet_quality": {"Poor": 1, "Average": 2, "Good": 3},
    "parental_education_level": {"High School": 1, "Bachelor": 2, "Master": 3},
}

for col, mapa in mapeamentos.items():
    col_cod = f"{col}_cod"
    df[col_cod] = df[col].map(mapa)
    print(f"\n  📌 {col} → {col_cod}")
    print(f"     Mapeamento: {mapa}")
    print(f"     Valores únicos resultantes: {sorted(df[col_cod].dropna().unique())}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. RESUMO DO DATASET TRANSFORMADO                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n" + "─" * 70)
print("5. RESUMO DO DATASET TRANSFORMADO")
print("─" * 70)

print(f"\n  📐 Dimensões finais: {df.shape[0]} linhas × {df.shape[1]} colunas")
print(f"  Colunas originais: 16")
print(f"  Colunas novas: {df.shape[1] - 16}")

novas_colunas = [c for c in df.columns if c not in pd.read_csv(
    "habitos_e_desempenho_estudantil.csv", nrows=0).columns]
print(f"\n  Novas colunas criadas:")
for col in novas_colunas:
    print(f"    • {col} ({df[col].dtype})")

print(f"\n  Valores nulos restantes:")
nulos = df.isnull().sum()
nulos = nulos[nulos > 0]
if nulos.empty:
    print("    ✅ Nenhum valor nulo!")
else:
    for col, n in nulos.items():
        print(f"    ⚠️  {col}: {n}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. VISUALIZAÇÕES DAS VARIÁVEIS DERIVADAS                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n" + "─" * 70)
print("6. GERANDO VISUALIZAÇÕES")
print("─" * 70)

# 6.1 — Distribuição de horas_distracao vs exam_score
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["horas_distracao"], kde=True, bins=25, color="coral", ax=axes[0])
axes[0].set_title("Distribuição: Horas de Distração Total")
axes[0].set_xlabel("Horas (redes sociais + Netflix)")

sns.scatterplot(data=df, x="horas_distracao", y="exam_score",
                hue="faixa_desempenho", alpha=0.5, ax=axes[1],
                palette={"Baixo": "#e74c3c", "Médio": "#f39c12", "Alto": "#27ae60"})
axes[1].set_title("Distração vs. Nota (por faixa de desempenho)")
axes[1].set_xlabel("Horas de Distração")
axes[1].set_ylabel("Nota (exam_score)")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/horas_distracao_analise.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/horas_distracao_analise.png")

# 6.2 — Faixas categóricas criadas
variaveis_faixa = ["faixa_redes_sociais", "faixa_desempenho",
                   "faixa_sono", "faixa_saude_mental"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(variaveis_faixa):
    order = df[col].value_counts().index.tolist()
    sns.countplot(data=df, x=col, ax=axes[i], hue=col,
                  order=order, palette="Set2", legend=False)
    axes[i].set_title(f"Distribuição: {col}", fontsize=12)
    total = len(df)
    for p in axes[i].patches:
        if p.get_height() > 0:
            pct = f"{p.get_height()/total*100:.1f}%"
            axes[i].annotate(pct, (p.get_x() + p.get_width()/2, p.get_height()),
                             ha="center", va="bottom", fontsize=9)
    axes[i].tick_params(axis="x", rotation=15)

fig.suptitle("Distribuição das Novas Variáveis Categóricas", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/faixas_categoricas.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/faixas_categoricas.png")

# 6.3 — Razão estudo/distração vs nota
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df, x="razao_estudo_distracao", y="exam_score",
                hue="faixa_desempenho", alpha=0.5, ax=ax,
                palette={"Baixo": "#e74c3c", "Médio": "#f39c12", "Alto": "#27ae60"})
ax.axvline(x=1, color="gray", ls="--", alpha=0.6, label="Razão = 1 (equilíbrio)")
ax.set_title("Razão Estudo/Distração vs. Nota", fontsize=14)
ax.set_xlabel("Razão (estudo ÷ distração)")
ax.set_ylabel("Nota (exam_score)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/razao_estudo_distracao.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/razao_estudo_distracao.png")

# 6.4 — Boxplot de exam_score por faixas criadas
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
faixas_box = ["faixa_redes_sociais", "faixa_sono",
              "faixa_saude_mental", "faixa_desempenho"]

for i, col in enumerate(faixas_box):
    sns.boxplot(data=df, x=col, y="exam_score", ax=axes[i],
                hue=col, palette="Set2", legend=False)
    axes[i].set_title(f"Nota por {col}", fontsize=11)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=15)

fig.suptitle("Desempenho (exam_score) por Faixas Derivadas", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/boxplot_faixas_vs_nota.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/boxplot_faixas_vs_nota.png")

# 6.5 — Dedicação acadêmica vs nota
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df, x="dedicacao_academica", y="exam_score",
                hue="faixa_desempenho", alpha=0.5, ax=ax,
                palette={"Baixo": "#e74c3c", "Médio": "#f39c12", "Alto": "#27ae60"})
ax.set_title("Dedicação Acadêmica (score composto) vs. Nota", fontsize=14)
ax.set_xlabel("Dedicação Acadêmica (0–100)")
ax.set_ylabel("Nota (exam_score)")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/dedicacao_vs_nota.png", bbox_inches="tight")
print(f"  ✅ Salvo: {OUTPUT_DIR}/dedicacao_vs_nota.png")

plt.close("all")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7. SALVAR DATASET TRANSFORMADO                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("\n\n" + "─" * 70)
print("7. SALVANDO DATASET TRANSFORMADO")
print("─" * 70)

output_csv = "dados_transformados.csv"
df.to_csv(output_csv, index=False)
print(f"\n  ✅ Dataset salvo: {output_csv}")
print(f"  Dimensões finais: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Resumo das estatísticas das novas variáveis numéricas
print("\n  Estatísticas das novas variáveis numéricas:")
novas_num = ["horas_distracao", "razao_estudo_distracao", "dedicacao_academica"]
print(df[novas_num].describe().round(2).to_string())

print("\n" + "=" * 70)
print("✅ Engenharia de dados concluída!")
print("=" * 70)

