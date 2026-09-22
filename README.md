# Desafio Analytics Engineer — Klubi

Este repositório contém a resolução do desafio técnico para a vaga de Analytics Engineer (Estágio) na Klubi. O projeto analisa os hábitos e o desempenho escolar de 1.000 alunos.

## 🚀 Como Rodar o Projeto

1. **Pré-requisitos:** Python 3.8+ e `pip`.
2. **Crie um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # no Linux/Mac
   ```
3. **Instale as dependências:**
   ```bash
   pip install pandas matplotlib seaborn numpy
   ```
4. **Execute a pipeline principal:**
   ```bash
   python analise_completa.py
   ```

A pipeline atual está dividida nas seguintes tarefas e irá gerar, como saída, logs no console e um novo arquivo de dados limpos (`dados_transformados.csv`). 

---

## 📋 Resumo das Etapas Executadas e Respostas

O código que resolve as tarefas abaixo está centralizado no script principal `analise_completa.py`.

### Tarefa 1: Exploração Inicial

**Pergunta:** *Importar e descrever a base (tipos, distribuição, valores ausentes). Diagnóstico de qualidade dos dados.*

**Nossas Conclusões:**
* **Dimensão:** A base contém 1.000 alunos e 16 variáveis.
* **Tipos de Dados:** Foram identificadas 9 variáveis numéricas (como `age`, `study_hours_per_day`, `exam_score`) e 7 categóricas (como `gender`, `diet_quality`, `parental_education_level`).
* **Diagnóstico de Qualidade (Valores Ausentes e Inconsistências):** 
  * A base apresentou excelente consistência em quase todos os atributos. Não foram identificadas linhas duplicadas, registros com percentuais fora de 0-100% ou idades/horas diárias irreais. 
  * A **única inconsistência** (ausência de dados) foi detectada na coluna `parental_education_level`, com exatos 91 valores ausentes (9.1% da base).

### Tarefa 2: Engenharia de Dados

**Pergunta:** *Criar variáveis derivadas se fizer sentido. Explicar como tratou dados ausentes ou inconsistentes.*

**Nossas Conclusões e Decisões:**
* **Tratamento de Ausentes:** Para lidar com os 91 valores nulos em `parental_education_level` (variável categórica ordinal), optei pela **imputação pela moda** (preenchendo os nulos com a categoria *"High School"*). 
  * *Justificativa:* Por ser apenas 9.1% do total e uma variável ordinal com poucas classes, imputar com a classe mais frequente evita a criação de ruído adicional nos dados (ex: uma categoria 'Desconhecido'), e manteve a distribuição original da amostra. A análise das médias das notas entre o grupo que possuía esse dado e o grupo nulo indicou uma ausência aleatória (diferença de apenas 0.5 na nota média).
* **Criação de Variáveis Derivadas:** Para enriquecer a base para os modelos e insights futuros, criei:
  1. `horas_distracao`: A soma de `social_media_hours` e `netflix_hours`.
  2. `razao_estudo_distracao`: A divisão entre as horas de estudo e as horas de distração. É um ótimo termômetro: valores > 1 indicam alunos que estudam mais do que se distraem.
  3. `faixa_redes_sociais`: Divisão da carga em redes sociais em percentis (Low, Medium, High).
  4. `faixa_desempenho`: Divisão categórica do `exam_score` (Baixo, Médio, Alto).
  5. **Codificação Ordinal:** Converteu atributos como `diet_quality` (Poor, Fair, Good) em inteiros (1, 2, 3), preparando o terreno para matrizes de correlação e ML.

---
*Nota: As Tarefas 3 (Análise Estatística), 4 (Aplicações), 5 (Visualização) e 6 (Síntese) serão adicionadas a este documento à medida que a análise avança!*

