# Documentação do Projeto — SIM (Sistema de Informação sobre Mortalidade)

## Corpus

- https://pt.wikipedia.org/wiki/Fortaleza_de_Itaipu
- https://pt.wikipedia.org/wiki/Bolsa_Oficial_de_Café
- https://pt.wikipedia.org/wiki/São_Vicente_(São_Paulo)

## Código Regressão Linear

```r
# 1) Leitura e preparo dos dados
dados <- read.csv2("DatasetSIM_tratado.csv", fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
dados$IDADE <- as.numeric(dados$IDADE)

mapa_escolaridade <- c(
  "Nenhuma" = 0, "1 a 3 anos" = 2, "4 a 7 anos" = 5.5,
  "8 a 11 anos" = 9.5, "12 anos ou mais" = 12
)
dados$ESCOLARIDADE_ANOS <- mapa_escolaridade[dados$ESCOLARIDADE]
dados$SEXO <- factor(dados$SEXO, levels = c("Feminino", "Masculino"))

dados_reg <- dados[!is.na(dados$ESCOLARIDADE_ANOS) & !is.na(dados$IDADE) &
                      !is.na(dados$ANO) & !is.na(dados$SEXO), ]

# 2) Ajuste do modelo múltiplo
m2 <- lm(IDADE ~ ESCOLARIDADE_ANOS + ANO + SEXO, data = dados_reg)

# 3) Previsões

novos_dados <- data.frame(
  ESCOLARIDADE_ANOS = c(0, 5.5, 12),
  ANO               = c(2023, 2023, 2023),
  SEXO              = factor(c("Masculino", "Masculino", "Feminino"),
                              levels = c("Feminino", "Masculino"))
)

previsoes <- predict(m2, novos_dados)
data.frame(novos_dados, IDADE_PREVISTA = round(previsoes, 1))
```

## Código Regressão Logística com escolaridade e idade como preditoras

```r
dados <- read.csv2("DatasetSIM_tratado.csv", encoding = "UTF-8",
                    na.strings = c("", "NA", "nan", "NaN"),
                    stringsAsFactors = FALSE)
names(dados)[1] <- "ANO"  # remove marca BOM que o R anexa ao 1o nome de coluna

cat("Dimensões originais:", dim(dados), "\n")
str(dados)
cat("\nDistribuição original de SEXO:\n")
print(table(dados$SEXO, useNA = "ifany"))
dados_limpos <- subset(dados, SEXO %in% c("Masculino", "Feminino") &
                               !is.na(IDADE) &
                               ESCOLARIDADE != "Ignorado" &
                               ESCOLARIDADE != "" &
                               !is.na(ESCOLARIDADE))

cat("\nLinhas após limpeza:", nrow(dados_limpos),
    "(de", nrow(dados), "originais)\n")
dados_limpos$SEXO_bin <- ifelse(dados_limpos$SEXO == "Feminino", 1, 0)
cat("\nDistribuição da variável binária SEXO_bin:\n")
print(table(dados_limpos$SEXO_bin))
dados_limpos$ESCOLARIDADE <- factor(
  dados_limpos$ESCOLARIDADE,
  levels = c("Nenhuma", "1 a 3 anos", "4 a 7 anos",
             "8 a 11 anos", "12 anos ou mais"),
  ordered = FALSE
)
dados_limpos$ESCOLARIDADE <- relevel(dados_limpos$ESCOLARIDADE, ref = "Nenhuma")
set.seed(42)
n <- nrow(dados_limpos)
indice_treino <- sample(seq_len(n), size = 0.7 * n)
treino <- dados_limpos[indice_treino, ]
teste  <- dados_limpos[-indice_treino, ]

cat("\nLinhas treino:", nrow(treino), " | Linhas teste:", nrow(teste), "\n")
modelo <- glm(SEXO_bin ~ IDADE + ESCOLARIDADE,
              data = treino,
              family = binomial(link = "logit"))

cat("\n===== RESUMO DO MODELO =====\n")
print(summary(modelo))
cat("\n===== ODDS RATIOS (exp dos coeficientes) =====\n")
odds_ratios <- exp(cbind(OR = coef(modelo), confint(modelo)))
print(round(odds_ratios, 3))
teste$prob_prevista <- predict(modelo, newdata = teste, type = "response")
teste$classe_prevista <- ifelse(teste$prob_prevista >= 0.5, 1, 0)

matriz_confusao <- table(Observado = teste$SEXO_bin,
                          Previsto = teste$classe_prevista)
cat("\n===== MATRIZ DE CONFUSÃO (conjunto de teste) =====\n")
print(matriz_confusao)

acuracia <- sum(diag(matriz_confusao)) / sum(matriz_confusao)
cat("\nAcurácia no teste:", round(acuracia, 4), "\n")
modelo_nulo <- glm(SEXO_bin ~ 1, data = treino, family = binomial)
pseudo_r2 <- 1 - (logLik(modelo) / logLik(modelo_nulo))
cat("Pseudo R² (McFadden):", round(as.numeric(pseudo_r2), 4), "\n")
```

## Código da regressão logística com a adição do Local da morte como preditora

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

dados_limpos$SEXO_bin <- ifelse(
  dados_limpos$SEXO == "Feminino",
  1,
  0
)
cat("\nDistribuição da variável SEXO_bin:\n")
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

set.seed(42)

n <- nrow(dados_limpos)

indice_treino <- sample(
  seq_len(n),
  size = 0.7 * n
)

treino <- dados_limpos[indice_treino, ]
teste  <- dados_limpos[-indice_treino, ]

cat(
  "\nLinhas treino:",
  nrow(treino),
  " | Linhas teste:",
  nrow(teste),
  "\n"
)

modelo <- glm(
  SEXO_bin ~ IDADE + ESCOLARIDADE + LOCAL_MORTE,
  data = treino,
  family = binomial(link = "logit")
)

cat("\n===== RESUMO DO MODELO =====\n")
print(summary(modelo))


cat("\n===== ODDS RATIOS =====\n")

odds_ratios <- exp(
  cbind(
    OR = coef(modelo),
    confint(modelo)
  )
)

print(round(odds_ratios, 3))
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
matriz_confusao <- table(
  Observado = teste$SEXO_bin,
  Previsto = teste$classe_prevista
)

cat("\n===== MATRIZ DE CONFUSÃO =====\n")
print(matriz_confusao)
acuracia <- sum(diag(matriz_confusao)) /
  sum(matriz_confusao)

cat(
  "\nAcurácia no teste:",
  round(acuracia, 4),
  "\n"
)
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
