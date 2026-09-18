# Documentação do Projeto — SIM (Sistema de Informação sobre Mortalidade)

## 1. Corpus

O corpus utilizado no projeto é composto por informações relacionadas ao município de Santos e à região da Baixada Santista, complementadas por fontes públicas para contextualização histórica e geográfica.

### Fontes utilizadas

- [Fortaleza de Itaipu — Wikipédia](https://pt.wikipedia.org/wiki/Fortaleza_de_Itaipu)
- [Bolsa Oficial de Café — Wikipédia](https://pt.wikipedia.org/wiki/Bolsa_Oficial_de_Café)
- [São Vicente (São Paulo) — Wikipédia](https://pt.wikipedia.org/wiki/São_Vicente_(São_Paulo))

---

## 2. Regressão Linear

A regressão linear múltipla foi utilizada para analisar a relação entre a **idade no momento do óbito** e as variáveis **escolaridade, ano e sexo**.

### Código

```r
# 1) Leitura e preparo dos dados
dados <- read.csv2(
  "DatasetSIM_tratado.csv",
  fileEncoding = "UTF-8-BOM",
  stringsAsFactors = FALSE
)

dados$IDADE <- as.numeric(dados$IDADE)

mapa_escolaridade <- c(
  "Nenhuma" = 0,
  "1 a 3 anos" = 2,
  "4 a 7 anos" = 5.5,
  "8 a 11 anos" = 9.5,
  "12 anos ou mais" = 12
)

dados$ESCOLARIDADE_ANOS <- mapa_escolaridade[dados$ESCOLARIDADE]

dados$SEXO <- factor(
  dados$SEXO,
  levels = c("Feminino", "Masculino")
)

dados_reg <- dados[
  !is.na(dados$ESCOLARIDADE_ANOS) &
  !is.na(dados$IDADE) &
  !is.na(dados$ANO) &
  !is.na(dados$SEXO),
]

# 2) Ajuste do modelo múltiplo
m2 <- lm(
  IDADE ~ ESCOLARIDADE_ANOS + ANO + SEXO,
  data = dados_reg
)

# 3) Previsões
novos_dados <- data.frame(
  ESCOLARIDADE_ANOS = c(0, 5.5, 12),
  ANO = c(2023, 2023, 2023),
  SEXO = factor(
    c("Masculino", "Masculino", "Feminino"),
    levels = c("Feminino", "Masculino")
  )
)

previsoes <- predict(m2, novos_dados)

data.frame(
  novos_dados,
  IDADE_PREVISTA = round(previsoes, 1)
)
```

---

## 3. Regressão Logística com Escolaridade e Idade

A regressão logística foi utilizada para verificar a relação entre **sexo** e as variáveis **idade** e **escolaridade**.

A variável `SEXO` foi transformada em uma variável binária:

- `1` = Feminino
- `0` = Masculino

A categoria **"Nenhuma"** foi definida como referência para a escolaridade.

### Código

```r
dados <- read.csv2(
  "DatasetSIM_tratado.csv",
  encoding = "UTF-8",
  na.strings = c("", "NA", "nan", "NaN"),
  stringsAsFactors = FALSE
)

names(dados)[1] <- "ANO"

cat("Dimensões originais:", dim(dados), "\n")
str(dados)

cat("\nDistribuição original de SEXO:\n")
print(table(dados$SEXO, useNA = "ifany"))

dados_limpos <- subset(
  dados,
  SEXO %in% c("Masculino", "Feminino") &
  !is.na(IDADE) &
  ESCOLARIDADE != "Ignorado" &
  ESCOLARIDADE != "" &
  !is.na(ESCOLARIDADE)
)

cat(
  "\nLinhas após limpeza:",
  nrow(dados_limpos),
  "(de",
  nrow(dados),
  "originais)\n"
)

dados_limpos$SEXO_bin <- ifelse(
  dados_limpos$SEXO == "Feminino",
  1,
  0
)

cat("\nDistribuição da variável binária SEXO_bin:\n")
print(table(dados_limpos$SEXO_bin))

dados_limpos$ESCOLARIDADE <- factor(
  dados_limpos$ESCOLARIDADE,
  levels = c(
    "Nenhuma",
    "1 a 3 anos",
    "4 a 7 anos",
    "8 a 11 anos",
    "12 anos ou mais"
  ),
  ordered = FALSE
)

dados_limpos$ESCOLARIDADE <- relevel(
  dados_limpos$ESCOLARIDADE,
  ref = "Nenhuma"
)

# Divisão entre treino e teste
set.seed(42)

n <- nrow(dados_limpos)

indice_treino <- sample(
  seq_len(n),
  size = 0.7 * n
)

treino <- dados_limpos[indice_treino, ]
teste <- dados_limpos[-indice_treino, ]

cat(
  "\nLinhas treino:",
  nrow(treino),
  " | Linhas teste:",
  nrow(teste),
  "\n"
)

# Modelo
modelo <- glm(
  SEXO_bin ~ IDADE + ESCOLARIDADE,
  data = treino,
  family = binomial(link = "logit")
)

cat("\n===== RESUMO DO MODELO =====\n")
print(summary(modelo))

# Odds Ratios
cat("\n===== ODDS RATIOS =====\n")

odds_ratios <- exp(
  cbind(
    OR = coef(modelo),
    confint(modelo)
  )
)

print(round(odds_ratios, 3))

# Previsões
teste$prob_prevista <- predict(
  modelo,
  newdata = teste,
  type = "response"
)

teste$classe_prevista <- ifelse(
  teste$prob_prevista >= 0.5,
  1,
  0
)

# Matriz de confusão
matriz_confusao <- table(
  Observado = teste$SEXO_bin,
  Previsto = teste$classe_prevista
)

cat("\n===== MATRIZ DE CONFUSÃO =====\n")
print(matriz_confusao)

# Acurácia
acuracia <- sum(diag(matriz_confusao)) /
  sum(matriz_confusao)

cat(
  "\nAcurácia no teste:",
  round(acuracia, 4),
  "\n"
)

# Pseudo R² de McFadden
modelo_nulo <- glm(
  SEXO_bin ~ 1,
  data = treino,
  family = binomial
)

pseudo_r2 <- 1 -
  (logLik(modelo) / logLik(modelo_nulo))

cat(
  "Pseudo R² (McFadden):",
  round(as.numeric(pseudo_r2), 4),
  "\n"
)
```

---

## 4. Regressão Logística com Local da Morte

Nesta etapa, foi adicionada a variável **`LOCAL_MORTE`** ao modelo de regressão logística.

O objetivo é verificar se o local do óbito apresenta associação com a variável `SEXO`, considerando também a **idade** e a **escolaridade**.

As categorias utilizadas foram:

- Hospital
- Outros estabelecimentos de saúde
- Domicílio
- Outros
- Via pública

A categoria **Hospital** foi utilizada como referência.

### Código

```r
dados <- read.csv2(
  "DatasetSIM_tratado.csv",
  encoding = "UTF-8",
  na.strings = c("", "NA", "nan", "NaN"),
  stringsAsFactors = FALSE
)

names(dados)[1] <- "ANO"

cat("Dimensões originais:", dim(dados), "\n")
str(dados)

cat("\nDistribuição original de SEXO:\n")
print(table(dados$SEXO, useNA = "ifany"))

cat("\nDistribuição original de LOCAL_MORTE:\n")
print(table(dados$LOCAL_MORTE, useNA = "ifany"))

# Limpeza dos dados
dados_limpos <- subset(
  dados,
  SEXO %in% c("Masculino", "Feminino") &
  !is.na(IDADE) &
  ESCOLARIDADE %in% c(
    "Nenhuma",
    "1 a 3 anos",
    "4 a 7 anos",
    "8 a 11 anos",
    "12 anos ou mais"
  ) &
  LOCAL_MORTE %in% c(
    "Hospital",
    "Outros estabelecimentos de saúde",
    "Domicílio",
    "Outros",
    "Via pública"
  )
)

cat(
  "\nLinhas após limpeza:",
  nrow(dados_limpos),
  "(de",
  nrow(dados),
  "originais)\n"
)

# Transformação da variável SEXO
dados_limpos$SEXO_bin <- ifelse(
  dados_limpos$SEXO == "Feminino",
  1,
  0
)

cat("\nDistribuição da variável SEXO_bin:\n")
print(table(dados_limpos$SEXO_bin))

# Transformação da escolaridade
dados_limpos$ESCOLARIDADE <- factor(
  dados_limpos$ESCOLARIDADE,
  levels = c(
    "Nenhuma",
    "1 a 3 anos",
    "4 a 7 anos",
    "8 a 11 anos",
    "12 anos ou mais"
  ),
  ordered = FALSE
)

dados_limpos$ESCOLARIDADE <- relevel(
  dados_limpos$ESCOLARIDADE,
  ref = "Nenhuma"
)

# Transformação do local da morte
dados_limpos$LOCAL_MORTE <- factor(
  dados_limpos$LOCAL_MORTE,
  levels = c(
    "Hospital",
    "Outros estabelecimentos de saúde",
    "Domicílio",
    "Outros",
    "Via pública"
  )
)

dados_limpos$LOCAL_MORTE <- relevel(
  dados_limpos$LOCAL_MORTE,
  ref = "Hospital"
)

cat("\nDistribuição de LOCAL_MORTE após limpeza:\n")
print(table(dados_limpos$LOCAL_MORTE))

# Divisão entre treino e teste
set.seed(42)

n <- nrow(dados_limpos)

indice_treino <- sample(
  seq_len(n),
  size = 0.7 * n
)

treino <- dados_limpos[indice_treino, ]
teste <- dados_limpos[-indice_treino, ]

cat(
  "\nLinhas treino:",
  nrow(treino),
  " | Linhas teste:",
  nrow(teste),
  "\n"
)

# Modelo de regressão logística
modelo <- glm(
  SEXO_bin ~ IDADE + ESCOLARIDADE + LOCAL_MORTE,
  data = treino,
  family = binomial(link = "logit")
)

cat("\n===== RESUMO DO MODELO =====\n")
print(summary(modelo))

# Odds Ratios
cat("\n===== ODDS RATIOS =====\n")

odds_ratios <- exp(
  cbind(
    OR = coef(modelo),
    confint(modelo)
  )
)

print(round(odds_ratios, 3))

# Previsões
teste$prob_prevista <- predict(
  modelo,
  newdata = teste,
  type = "response"
)

teste$classe_prevista <- ifelse(
  teste$prob_prevista >= 0.5,
  1,
  0
)

# Matriz de confusão
matriz_confusao <- table(
  Observado = teste$SEXO_bin,
  Previsto = teste$classe_prevista
)

cat("\n===== MATRIZ DE CONFUSÃO =====\n")
print(matriz_confusao)

# Acurácia
acuracia <- sum(diag(matriz_confusao)) /
  sum(matriz_confusao)

cat(
  "\nAcurácia no teste:",
  round(acuracia, 4),
  "\n"
)

# Pseudo R² de McFadden
modelo_nulo <- glm(
  SEXO_bin ~ 1,
  data = treino,
  family = binomial
)

pseudo_r2 <- 1 -
  (logLik(modelo) / logLik(modelo_nulo))

cat(
  "Pseudo R² (McFadden):",
  round(as.numeric(pseudo_r2), 4),
  "\n"
)
```

---

# 5. Análise Exploratória dos Dados

A análise exploratória tem como objetivo compreender a distribuição das variáveis presentes no conjunto de dados, com foco nos óbitos registrados na **Baixada Santista**.

Foram considerados os seguintes municípios:

- Santos
- São Vicente
- Guarujá
- Praia Grande
- Cubatão

### Código

```r
# Leitura dos dados
dados <- read.csv(
  "DatasetSIM_tratado.csv",
  sep = ";",
  dec = ","
)

# Seleção dos municípios da Baixada Santista
baixada <- subset(
  dados,
  MUNICIPIO %in% c(
    "Santos",
    "São Vicente",
    "Guarujá",
    "Praia Grande",
    "Cubatão"
  )
)

# Conversão da idade
baixada$IDADE_ANOS <- as.numeric(
  as.character(baixada$IDADE)
)

# Resumo estatístico da idade
print("Resumo estatístico da Idade:")
cat("\n")

summary(baixada$IDADE_ANOS)

cat("\n")

print(
  paste(
    "Desvio Padrão:",
    sd(
      baixada$IDADE_ANOS,
      na.rm = TRUE
    )
  )
)

# Histograma e boxplot
par(
  mfrow = c(1, 2),
  mar = c(4, 4, 2, 1),
  pty = "s"
)

hist(
  baixada$IDADE_ANOS,
  prob = TRUE,
  col = "lightblue",
  border = "white",
  main = "Histograma de Idade",
  xlab = "Idade (anos)",
  ylab = "Densidade"
)

curve(
  dnorm(
    x,
    mean = mean(
      baixada$IDADE_ANOS,
      na.rm = TRUE
    ),
    sd = sd(
      baixada$IDADE_ANOS,
      na.rm = TRUE
    )
  ),
  add = TRUE,
  col = "darkorange",
  lwd = 2
)

boxplot(
  baixada$IDADE_ANOS,
  col = "lightblue",
  main = "Boxplot de Idade",
  ylab = "Idade (anos)"
)

# Idade por escolaridade
boxplot(
  IDADE_ANOS ~ ESCOLARIDADE,
  data = baixada,
  col = "lightgreen",
  main = "Idade do Óbito por Escolaridade",
  xlab = "Escolaridade",
  ylab = "Idade (anos)",
  cex.axis = 0.7
)

# Frequência de óbitos por sexo
print("Frequência de Óbitos por Sexo:")
table(baixada$SEXO)

# Cinco principais causas básicas de morte
print("Top 5 Causas Básicas de Morte:")

sort(
  table(baixada$CAUSABASMORTE),
  decreasing = TRUE
)[1:5]
```

---

## 6. Estrutura da Análise

O projeto segue, de forma geral, as seguintes etapas:

1. **Preparação dos dados**
   - Leitura do dataset;
   - Tratamento de valores ausentes;
   - Conversão das variáveis;
   - Seleção dos registros válidos.

2. **Análise exploratória**
   - Estatísticas descritivas;
   - Distribuição da idade;
   - Análise por escolaridade;
   - Frequência por sexo;
   - Principais causas de morte.

3. **Regressão linear**
   - Variável resposta: `IDADE`;
   - Preditoras: `ESCOLARIDADE_ANOS`, `ANO` e `SEXO`.

4. **Regressão logística**
   - Variável resposta: `SEXO`;
   - Preditoras: `IDADE` e `ESCOLARIDADE`.

5. **Expansão do modelo logístico**
   - Inclusão da variável `LOCAL_MORTE`;
   - Avaliação dos Odds Ratios;
   - Matriz de confusão;
   - Acurácia;
   - Pseudo R² de McFadden.

6. **Avaliação dos modelos**
   - Comparação dos resultados;
   - Interpretação dos coeficientes;
   - Análise da capacidade preditiva;
   - Identificação das variáveis mais relevantes.


     CONSOLIDADO: 
     # Relatório Consolidado do Projeto — SIM (Sistema de Informação sobre Mortalidade)

## 1. Visão Geral do Projeto
Este projeto analisa os microdados do **Sistema de Informação sobre Mortalidade (SIM)** focando na região da **Baixada Santista** (Santos, São Vicente, Guarujá, Praia Grande e Cubatão). O objetivo principal é identificar padrões socioeconômicos e demográficos associados à idade e ao perfil do óbito por meio de Análise Exploratória e Modelagem Estatística Avançada.

---

## 2. Corpus e Processamento de Dados

* **Base de Dados:** Dataset estruturado contendo registros de óbitos tratados (`DatasetSIM_tratado.csv`).
* **Contextualização Regional:** Complementado com fontes públicas de domínio histórico e geográfico da região.
* **Engenharia de Variáveis (Feature Engineering):**
  * **Escolaridade:** Conversão de categorias discretas para escala contínua em anos ($0, 2, 5.5, 9.5, 12$).
  * **Sexo:** Binarização da variável resposta/preditora ($1 = \text{Feminino}$, $0 = \text{Masculino}$).
  * **Limpeza e Filtragem:** Remoção sistemática de valores ausentes (`NA`), registros vazios e categorias não informadas ("Ignorado").

---

## 3. Análise Exploratória de Dados (EDA)

A etapa exploratória mapeia a distribuição e o perfil epidemiológico das mortes registradas na região:

* **Perfil Etário:** Levantamento de média, mediana, desvio padrão e distribuição de densidade da idade no momento do óbito.
* **Análise Bivariada:** Cruzamento entre nível de escolaridade e expectativa de vida ao falecer via *boxplots*.
* **Perfil Epidemiológico:** Identificação da frequência de óbitos por sexo e o ranking das **Top 5 causas básicas de morte**.

---

## 4. Modelagem Estatística

| Modelo | Tipo de Regressão | Variável Resposta (Alvo) | Variáveis Preditoras |
| :--- | :--- | :--- | :--- |
| **Modelo 1** | Linear Múltipla | `IDADE` (Contínua) | `ESCOLARIDADE_ANOS`, `ANO`, `SEXO` |
| **Modelo 2** | Logística (Base) | `SEXO_bin` (Binária) | `IDADE`, `ESCOLARIDADE` (Ref: "Nenhuma") |
| **Modelo 3** | Logística (Expandida) | `SEXO_bin` (Binária) | `IDADE`, `ESCOLARIDADE`, `LOCAL_MORTE` (Ref: "Hospital") |

---

## 5. Pipeline de Validação e Avaliação
Data Prep -> Divisão Treino/Teste (70/30) -> Ajuste do Modelo -> Avaliação das Métricas


1. **Ajuste dos Modelos:** Executado no conjunto de treino ($70\%$ dos dados) com semente aleatória fixada (`set.seed(42)`).
2. **Razão de Chances (Odds Ratios):** Cálculo dos coeficientes exponenciados $exp(\beta)$ com intervalos de confiança de $95\%$.
3. **Desempenho Preditivo (Massa de Teste - 30%):**
   * **Matriz de Confusão:** Avaliação de falsos positivos/negativos utilizando *threshold* de $0.5$.
   * **Acurácia Global:** Proporção total de acertos das previsões na base de teste.
   * **Pseudo $R^2$ de McFadden:** Medida de qualidade do ajuste comparando o modelo ajustado contra o modelo nulo.
