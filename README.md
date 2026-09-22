# Desafio Analytics Engineer — Klubi

Este repositório documenta a resolução do desafio técnico para a vaga de Analytics Engineer (Estágio) na Klubi. O objetivo principal deste projeto é explorar, tratar e analisar uma base de dados que mapeia os hábitos e o desempenho acadêmico de 1.000 alunos, visando extrair padrões de comportamento e insights acionáveis.

## 🚀 Como Executar o Projeto

1. **Pré-requisitos:** Python 3.8+ e `pip`.
2. **Crie e ative um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # no Linux/Mac
   ```
3. **Instale as dependências:**
   ```bash
   pip install pandas matplotlib seaborn numpy
   ```
4. **Execute a pipeline completa:**
   ```bash
   python analise_completa.py
   ```

A pipeline automatizada realizará todas as etapas desde a exploração inicial até as análises estatísticas mais avançadas.

---

## 📖 Relatório de Análise e Desenvolvimento

O desenvolvimento da solução seguiu um fluxo natural de tratamento de dados e geração de insights estatísticos.

### 1. Exploração e Qualidade dos Dados
O ponto de partida do projeto consistiu em mapear o cenário dos dados fornecidos. A base contempla 1.000 registros sem nenhuma linha duplicada, distribuídos entre 16 variáveis. Estas variam desde características demográficas (idade, gênero e nível de escolaridade dos pais) até hábitos diários (horas dedicadas a estudos, lazer e exercícios), culminando no nosso alvo preditivo: a nota final do aluno (`exam_score`).

Uma auditoria da qualidade revelou um dataset bem íntegro: os valores mantinham coerência semântica e não exibiam falhas extremas como horas diárias negativas. A única exceção foi a variável que indica a escolaridade dos pais (`parental_education_level`), que possuía cerca de 9,1% de valores ausentes (91 registros). 

### 2. Transformação e Engenharia de Variáveis
Para tratar os nulos na coluna sobre a escolaridade parental, e considerando se tratar de uma categoria ordinal de baixo impacto nulo, optou-se pela imputação através da **moda** (categoria *"High School"*). O método foi preferido em relação à criação de uma categoria "Desconhecida", pois protege o dataset contra ruídos prematuros, mantendo a integridade distributiva original. Uma rápida verificação confirmou a validade da técnica: a média de nota geral entre o grupo íntegro (69.6) e o que continha falhas (70.0) se mostrou quase idêntica, descartando ausências com viés comportamental oculto.

Com os dados consistentes, avançamos para a criação de variáveis derivadas para facilitar análises dimensionais. A métrica de `horas_distracao` (somatório de tempo em redes sociais e Netflix) foi elaborada junto da `razao_estudo_distracao`. Esta razão serve como um excelente KPI: quantifica de imediato a dominância dos estudos ou do lazer no cotidiano de cada aluno. A conversão de atributos categóricos literais (como `diet_quality` ou `internet_quality`) em correspondentes ordinais e a divisão temporal em bandas (High, Medium, Low) formaram a preparação final do conjunto, hoje estruturado para interpretações matemáticas diretas.

### 3. Análises Estatísticas e Fatores de Influência
Superado o polimento dos dados, submetemos a amostra ampliada a um cálculo de correlação de Pearson. O resultado esclareceu matematicamente as dinâmicas de dedicação estudantil e como elas de fato se interligam ao sucesso nas avaliações.

O **maior impulsionador de sucesso** de longe reside no compromisso linear com o estudo. A variável `study_hours_per_day` apresenta uma fortíssima correlação positiva (**+0.825**) com a nota. Complementando isso, a nossa métrica recém-criada, `razao_estudo_distracao`, despontou como a segunda força de maior impacto (**+0.425**), mostrando que além de acumular horas brutas de livro, blindar esse tempo perante o ócio impulsiona notas altas. O terceiro fator de impacto é a estabilidade mental (`mental_health_rating`, em **+0.322**), atestando a importância do bem-estar.

No lado inverso, o tempo cedido às distrações constitui o **maior dreno de pontuação**. A consolidação total de tempo em tela recém gerada por nossa engenharia (`horas_distracao`) figura como a correlação de base mais negativa (**-0.238**). Se o analisarmos fracionado, o streaming de vídeo (`netflix_hours`, com **-0.172**) e as redes sociais (`social_media_hours`, com **-0.167**) punem o rendimento escolar de forma praticamente equiparável.

*Nota: As Tarefas 4, 5 e 6 serão executadas nas próximas iterações do desenvolvimento e unificadas nesta mesma rotina documental.*
