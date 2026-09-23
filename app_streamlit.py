"""
App interativo — Simulador de Desempenho (Klubi)
=================================================
Streamlit app que treina o mesmo modelo preditivo do pipeline principal
e permite simular a nota esperada de um aluno a partir dos hábitos dele.

Rodar com: streamlit run app_streamlit.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Simulador de Desempenho — Klubi", page_icon="🎓", layout="centered")

# ── Carregar e preparar dados ──────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    df = pd.read_csv("habitos_e_desempenho_estudantil.csv")
    df["parental_education_level"] = df["parental_education_level"].fillna(
        df["parental_education_level"].mode()[0]
    )
    mapa_dieta = {"Poor": 1, "Fair": 2, "Good": 3}
    mapa_internet = {"Poor": 1, "Average": 2, "Good": 3}
    mapa_pais = {"High School": 1, "Bachelor": 2, "Master": 3}
    df["diet_quality_cod"] = df["diet_quality"].map(mapa_dieta)
    df["internet_quality_cod"] = df["internet_quality"].map(mapa_internet)
    df["parental_education_level_cod"] = df["parental_education_level"].map(mapa_pais)
    return df


@st.cache_resource
def treinar_modelo(df):
    features = [
        "study_hours_per_day", "social_media_hours", "netflix_hours",
        "sleep_hours", "exercise_frequency", "attendance_percentage",
        "mental_health_rating", "diet_quality_cod", "internet_quality_cod",
        "parental_education_level_cod",
    ]
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(df[features].median())
    y = df["exam_score"]
    modelo = LinearRegression().fit(X, y)
    return modelo, features


df = carregar_dados()
modelo, features = treinar_modelo(df)

st.title("🎓 Simulador de Desempenho Estudantil")
st.write(
    "Ajuste os hábitos abaixo para simular a nota esperada (`exam_score`) de um aluno. "
    "O modelo é uma regressão linear treinada nos 1.000 registros da base da Klubi — "
    "pensado como ferramenta de apoio para orientadores acadêmicos identificarem, cedo, "
    "quem pode precisar de suporte extra."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("Horas de estudo por dia", 0.0, 12.0, float(df["study_hours_per_day"].median()), 0.5)
    social_media = st.slider("Horas em redes sociais/dia", 0.0, 10.0, float(df["social_media_hours"].median()), 0.5)
    netflix = st.slider("Horas de Netflix/dia", 0.0, 10.0, float(df["netflix_hours"].median()), 0.5)
    sleep = st.slider("Horas de sono/dia", 3.0, 12.0, float(df["sleep_hours"].median()), 0.5)
    attendance = st.slider("Frequência escolar (%)", 0, 100, int(df["attendance_percentage"].median()))

with col2:
    exercise = st.slider("Frequência de exercícios (dias/semana)", 0, 7, int(df["exercise_frequency"].median()))
    mental_health = st.slider("Saúde mental (1 = pior, 10 = melhor)", 1, 10, int(df["mental_health_rating"].median()))
    diet = st.selectbox("Qualidade da dieta", ["Poor", "Fair", "Good"], index=1)
    internet = st.selectbox("Qualidade da internet", ["Poor", "Average", "Good"], index=1)
    parental_edu = st.selectbox("Escolaridade dos pais", ["High School", "Bachelor", "Master"], index=0)

entrada = pd.DataFrame([{
    "study_hours_per_day": study_hours,
    "social_media_hours": social_media,
    "netflix_hours": netflix,
    "sleep_hours": sleep,
    "exercise_frequency": exercise,
    "attendance_percentage": attendance,
    "mental_health_rating": mental_health,
    "diet_quality_cod": {"Poor": 1, "Fair": 2, "Good": 3}[diet],
    "internet_quality_cod": {"Poor": 1, "Average": 2, "Good": 3}[internet],
    "parental_education_level_cod": {"High School": 1, "Bachelor": 2, "Master": 3}[parental_edu],
}])[features]

nota_prevista = float(np.clip(modelo.predict(entrada)[0], 0, 100))
media_base = df["exam_score"].mean()

st.divider()
c1, c2 = st.columns(2)
c1.metric("Nota prevista", f"{nota_prevista:.1f}", delta=f"{nota_prevista - media_base:+.1f} vs. média da base")
c2.metric("Média da base (1.000 alunos)", f"{media_base:.1f}")

if nota_prevista < 50:
    st.warning("Perfil de risco — combinação de hábitos associada a notas abaixo de 50.")
elif nota_prevista < 75:
    st.info("ℹPerfil intermediário — há espaço para ganhos ajustando estudo/distração.")
else:
    st.success(" Perfil de alto desempenho.")

st.divider()
st.subheader("Onde esse aluno simulado está em relação à base")
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["exam_score"], bins=25, color="#2a9d8f", alpha=0.7)
ax.axvline(nota_prevista, color="red", linestyle="--", linewidth=2, label="Aluno simulado")
ax.set_xlabel("exam_score")
ax.set_ylabel("Nº de alunos")
ax.legend()
st.pyplot(fig)

st.caption(
    "Uso pretendido: apoio à decisão de orientadores/monitoria da Klubi para priorizar "
    "intervenções (tutoria, ajuste de rotina) — não substitui avaliação pedagógica individual."
)
