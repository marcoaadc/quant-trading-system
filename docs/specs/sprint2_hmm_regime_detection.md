# Especificacao Matematica: Deteccao de Regime de Mercado via HMM Gaussiano

**Sprint 2 -- Quant Trading System**

**Autor**: Quant Researcher Senior
**Data**: 2026-05-05
**Status**: Aprovado para implementacao
**Ativos**: WDO (mini dolar futuro B3), WIN (mini indice futuro B3)
**Timeframes**: 1 minuto, 5 minutos

---

## Indice

1. [Transformacao de Features](#1-transformacao-de-features)
2. [Estacionariedade](#2-estacionariedade)
3. [Configuracao do HMM](#3-configuracao-do-hmm)
4. [Prevencao de Look-Ahead Bias](#4-prevencao-de-look-ahead-bias)
5. [Pipeline Recomendado](#5-pipeline-recomendado-fluxo-logico)
6. [Metricas de Avaliacao](#6-metricas-de-avaliacao)
7. [Riscos e Limitacoes](#7-riscos-e-limitacoes)
8. [Referencias](#8-referencias)

---

## 1. Transformacao de Features

O vetor de observacao para o HMM sera composto por **5 features** extraidas dos dados OHLCV. Cada feature foi selecionada para capturar uma dimensao distinta do regime de mercado: direcao (momento), dispersao (volatilidade), intensidade (volume), e microestrutura (range intra-candle).

### 1.1 Retornos Logaritmicos (Log Returns)

**Formula**:

$$r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)$$

onde $C_t$ e o preco de fechamento (close) no instante $t$.

**Justificativa teorica**: Os retornos logaritmicos sao a transformacao fundamental em financas quantitativas porque:
- Sao aditivos no tempo: $r_{t_1 \to t_n} = \sum_{i=1}^{n} r_{t_i}$, o que facilita a agregacao temporal.
- Sao aproximadamente simetricos para variacao pequena (ao contrario de retornos aritmeticos).
- Eliminam a nao-estacionariedade em nivel do preco, produzindo uma serie (aproximadamente) estacionaria.
- Capturam a **direcao instantanea** do mercado, que e a dimensao primaria para distinguir regimes de alta vs. baixa.

**Parametros recomendados**:
- Nenhum parametro de janela necessario (calculo ponto-a-ponto).
- Aplicavel identicamente para 1min e 5min.

**Nota de implementacao**: O primeiro valor da serie sera `NaN` (nao ha $C_{t-1}$ para $t=0$). Descartar esta observacao.

---

### 1.2 Realized Volatility (Volatilidade Realizada)

**Formula**:

$$\sigma_t^{(N)} = \sqrt{\sum_{i=0}^{N-1} r_{t-i}^2}$$

onde $N$ e o tamanho da janela e $r_{t-i}$ sao os retornos logaritmicos passados.

**Nota**: Usamos a soma dos retornos ao quadrado (sem subtrair a media) porque em alta frequencia a media dos retornos e estatisticamente indistinguivel de zero ($E[r_t] \approx 0$). Isto e consistente com a definicao standard de realized volatility em microestrutura de mercado (Andersen & Bollerslev, 1998).

**Justificativa teorica**: A volatilidade realizada captura o **segundo momento condicional** da distribuicao dos retornos. Regimes de mercado diferem fundamentalmente na sua volatilidade:
- **Regime de tendencia calma**: $\sigma$ baixo e estavel.
- **Regime de alta volatilidade / crise**: $\sigma$ elevado, frequentemente com clustering (efeito ARCH).
- **Regime lateral**: $\sigma$ intermediario, sem direcao clara.

A volatilidade e o principal discriminador entre regimes na literatura (Hamilton, 1989; Ang & Bekaert, 2002).

**Parametros recomendados**:

| Timeframe | Janela $N$ | Justificativa |
|-----------|-----------|---------------|
| 1 min | $N = 20$ | ~20 minutos de mercado; captura a volatilidade de curto prazo sem excesso de suavizacao |
| 5 min | $N = 20$ | ~100 minutos (~1.5 horas de pregao B3); janela adequada para regime intradiario |

**Transformacao adicional**: Aplicar logaritmo natural para estabilizacao de variancia (ver Seccao 2):

$$\tilde{\sigma}_t = \ln(\sigma_t^{(N)} + \epsilon)$$

onde $\epsilon = 10^{-10}$ para evitar $\ln(0)$.

---

### 1.3 Indicador Normalizado de Momento (Momentum Z-Score)

**Formula** (em dois passos):

**Passo 1** -- Retorno acumulado na janela $M$:

$$\text{mom}_t = \sum_{i=0}^{M-1} r_{t-i} = \ln\left(\frac{C_t}{C_{t-M}}\right)$$

**Passo 2** -- Normalizacao via z-score rolling com janela $L$:

$$z_t^{\text{mom}} = \frac{\text{mom}_t - \hat{\mu}_t^{(L)}}{\hat{\sigma}_t^{(L)}}$$

onde:
- $\hat{\mu}_t^{(L)} = \frac{1}{L}\sum_{i=0}^{L-1}\text{mom}_{t-i}$ e a media movel do momento.
- $\hat{\sigma}_t^{(L)} = \sqrt{\frac{1}{L-1}\sum_{i=0}^{L-1}(\text{mom}_{t-i} - \hat{\mu}_t^{(L)})^2}$ e o desvio padrao movel.

**Justificativa teorica**: O momento bruto captura a **tendencia direcional acumulada**. A normalizacao via z-score:
- Torna a feature comparavel entre periodos de alta e baixa volatilidade.
- Garante estacionariedade (media zero, variancia unitaria por construcao da janela rolling).
- Permite distinguir entre tendencia persistente ($|z| > 1.5$) e noise ($|z| < 0.5$).

**Parametros recomendados**:

| Timeframe | Janela momento $M$ | Janela z-score $L$ | Justificativa |
|-----------|-------------------|-------------------|---------------|
| 1 min | $M = 10$ | $L = 60$ | Momento de ~10min normalizado contra ~1 hora |
| 5 min | $M = 10$ | $L = 60$ | Momento de ~50min normalizado contra ~5 horas (aprox. 1 pregao B3) |

**Clipping**: Aplicar truncamento em $[-5, +5]$ para robustez contra outliers extremos:

$$z_t^{\text{mom, clip}} = \max(-5, \min(5, z_t^{\text{mom}}))$$

---

### 1.4 Volume Relativo (Relative Volume)

**Formula**:

$$\text{vrel}_t = \frac{V_t}{\hat{V}_t^{(K)}}$$

onde:
- $V_t$ e o volume no candle $t$.
- $\hat{V}_t^{(K)} = \frac{1}{K}\sum_{i=1}^{K} V_{t-i}$ e a media movel do volume nos $K$ candles **anteriores** (excluindo o candle atual para evitar look-ahead).

**Transformacao**: Aplicar logaritmo natural para normalizar a distribuicao (volumes tendem a ser log-normais):

$$\tilde{v}_t = \ln(\text{vrel}_t + \epsilon)$$

onde $\epsilon = 10^{-10}$.

**Justificativa teorica**: O volume e um indicador crucial de **confirmacao de regime**:
- **Breakouts reais** sao acompanhados por volume acima da media ($\tilde{v}_t > 0$).
- **Periodos laterais** tipicamente apresentam volume abaixo da media ($\tilde{v}_t < 0$).
- **Panic selling / squeezes** mostram spikes de volume extremos ($\tilde{v}_t \gg 0$).

A normalizacao pelo volume proprio do ativo torna a feature comparavel entre WDO e WIN, que possuem niveis absolutos de volume muito diferentes.

**Parametros recomendados**:

| Timeframe | Janela $K$ | Justificativa |
|-----------|-----------|---------------|
| 1 min | $K = 60$ | ~1 hora; captura o padrao intradiario de volume |
| 5 min | $K = 60$ | ~5 horas (~1 pregao B3); baseline de volume para o dia |

**Nota sobre volume em futuros B3**: O volume nos proxies (USDBRL=X, ^BVSP) pode nao refletir exatamente o volume dos contratos futuros mini. Se disponivel, usar volume dos contratos diretamente. Se usando proxy, a feature ainda e informativa (volume do ativo subjacente correlaciona com atividade do futuro), mas com menor poder discriminativo. Documentar qual fonte de volume esta sendo utilizada.

---

### 1.5 Range Normalizado (Normalized Range / Parkinson-like)

**Formula**:

$$\text{NR}_t = \ln\left(\frac{H_t}{L_t}\right)$$

onde $H_t$ e o preco maximo (high) e $L_t$ e o preco minimo (low) do candle $t$.

**Transformacao para z-score rolling** com janela $L$:

$$z_t^{\text{NR}} = \frac{\text{NR}_t - \hat{\mu}_t^{\text{NR}(L)}}{\hat{\sigma}_t^{\text{NR}(L)}}$$

com a mesma logica de media e desvio padrao movel descrita na Seccao 1.3.

**Justificativa teorica**: O range logaritmico e um estimador eficiente de volatilidade intra-candle (Parkinson, 1980). Diferentemente da realized volatility (que usa apenas closes), o range captura a **amplitude total do movimento** dentro de cada candle:
- Adiciona informacao complementar a $\sigma_t$: e possivel ter baixa realized volatility (closes similares) mas alto range (movimento intra-candle intenso que reverte).
- Detecta regimes de **expansao vs. contracao de range** que sao precursores de mudancas de regime.
- Capturas de "doji-like behavior" (alto range, low net return) que indicam indecisao/transicao de regime.

**Parametros recomendados**:

| Timeframe | Janela z-score $L$ | Justificativa |
|-----------|-------------------|---------------|
| 1 min | $L = 60$ | Mesma janela do z-score de momento para consistencia |
| 5 min | $L = 60$ | Consistencia com as demais features rolling |

**Clipping**: Aplicar truncamento em $[-5, +5]$ (mesmo criterio do z-score de momento).

---

### 1.6 Resumo do Vetor de Observacao

O vetor de observacao para o HMM no instante $t$ e:

$$\mathbf{x}_t = \begin{bmatrix} r_t \\ \tilde{\sigma}_t \\ z_t^{\text{mom, clip}} \\ \tilde{v}_t \\ z_t^{\text{NR, clip}} \end{bmatrix} \in \mathbb{R}^5$$

**Dimensionalidade**: 5 features. Este numero e deliberadamente baixo para:
- Evitar curse of dimensionality no EM algorithm do HMM.
- Manter interpretabilidade de cada estado.
- Reduzir risco de overfitting com dados limitados (max 60 dias de 5min, 7 dias de 1min).

**Warm-up period**: As primeiras $L_{\max} = \max(N, M, K, L)$ observacoes devem ser descartadas pois as janelas rolling nao estao completamente preenchidas. Com os parametros recomendados: **descartar as primeiras 60 observacoes** de cada sessao/dia.

---

## 2. Estacionariedade

### 2.1 Por que a Estacionariedade e Critica para o HMM

O HMM Gaussiano assume que as distribuicoes de emissao $b_j(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$ sao **estacionarias dentro de cada estado** $j$. Se as features exibirem tendencia temporal (nao-estacionariedade), o modelo confundira:
- **Drift secular** com mudanca de regime.
- **Heteroscedasticidade** temporal com transicoes entre estados.

Isto invalida os parametros estimados pelo algoritmo de Baum-Welch (EM), porque os parametros $(\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$ seriam uma mistura de valores de diferentes epocas, sem significado interpretavel.

**Requisito**: Todas as features do vetor $\mathbf{x}_t$ devem ser (pelo menos fracamente) estacionarias, ou seja, seus dois primeiros momentos devem ser constantes ao longo do tempo.

### 2.2 Analise de Estacionariedade por Feature

| Feature | Naturalmente estacionaria? | Transformacao necessaria |
|---------|---------------------------|------------------------|
| $r_t$ (log returns) | **Sim** (por construcao: diferenciacao de log-precos) | Nenhuma. Log returns sao I(0) por construcao. |
| $\tilde{\sigma}_t$ (log realized vol) | **Parcialmente**: apresenta clustering de volatilidade (ARCH effects), mas sem tendencia deterministica | A transformacao logaritmica ja aplicada na Seccao 1.2 estabiliza a variancia. O clustering e uma propriedade que o HMM deve capturar (nao eliminar). **Nenhuma transformacao adicional necessaria.** |
| $z_t^{\text{mom}}$ (momentum z-score) | **Sim** (por construcao: z-score rolling forca media zero e variancia unitaria) | Nenhuma adicional. O z-score rolling ja garante estacionariedade local. |
| $\tilde{v}_t$ (log relative volume) | **Parcialmente**: volume relativo pode ter sazonalidade intradiaria (abertura e fechamento do pregao B3 tem volume sistematicamente maior) | Ver tratamento de sazonalidade na Seccao 2.3. |
| $z_t^{\text{NR}}$ (range z-score) | **Sim** (por construcao: z-score rolling) | Nenhuma adicional. |

### 2.3 Tratamento de Sazonalidade Intradiaria do Volume

O mercado B3 possui um padrao de volume intradiario em "U" (volume alto na abertura ~9:00, queda no meio do dia, aumento no fechamento ~17:00). Para o volume relativo:

**Opcao recomendada**: Calcular a media movel $\hat{V}_t^{(K)}$ com $K$ suficientemente grande para suavizar o padrao intradiario. Com $K=60$ candles de 5min, cobrimos ~5 horas, o que e aproximadamente um pregao completo. Isto mitiga a sazonalidade sem necessidade de desazonalizacao explicita.

**Opcao avancada (se o padrao persistir)**: Desazonalizar o volume calculando o fator de sazonalidade intradiario:

$$S_h = \frac{\bar{V}_h}{\bar{V}_{\text{dia}}}$$

onde $\bar{V}_h$ e o volume medio historico na hora $h$ do dia, e $\bar{V}_{\text{dia}}$ e o volume medio diario. O volume desazonalizado seria:

$$V_t^{\text{adj}} = \frac{V_t}{S_{h(t)}}$$

**Importante**: O fator $S_h$ deve ser calculado **apenas com dados passados** (expanding window ou janela historica de $D$ dias anteriores, nunca incluindo dados futuros).

### 2.4 Testes Estatisticos de Estacionariedade

O engenheiro deve implementar os seguintes testes para validar a estacionariedade de cada feature antes do treino do HMM:

#### 2.4.1 Teste Augmented Dickey-Fuller (ADF)

- **Hipotese nula ($H_0$)**: A serie possui raiz unitaria (nao-estacionaria).
- **Hipotese alternativa ($H_1$)**: A serie e estacionaria.
- **Criterio de decisao**: Rejeitar $H_0$ se p-valor < 0.05.
- **Implementacao**: `statsmodels.tsa.stattools.adfuller`
- **Parametros**:
  - `maxlag`: Usar selecao automatica via AIC (`autolag='AIC'`).
  - `regression`: `'c'` (constante, sem tendencia) -- adequado para retornos e features normalizadas.

#### 2.4.2 Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

- **Hipotese nula ($H_0$)**: A serie e estacionaria (nivel ou tendencia).
- **Hipotese alternativa ($H_1$)**: A serie possui raiz unitaria.
- **Criterio de decisao**: **Nao** rejeitar $H_0$ se p-valor > 0.05.
- **Implementacao**: `statsmodels.tsa.stattools.kpss`
- **Parametros**:
  - `regression`: `'c'` (estacionariedade em nivel).
  - `nlags`: `'auto'` (selecao automatica).

**Nota critica**: Usar ADF e KPSS **conjuntamente** como confirmation strategy:

| ADF | KPSS | Conclusao |
|-----|------|-----------|
| Rejeita $H_0$ (p < 0.05) | Nao rejeita $H_0$ (p > 0.05) | **Estacionaria** (confirmado por ambos) |
| Nao rejeita $H_0$ | Rejeita $H_0$ | **Nao-estacionaria** (ambos concordam) |
| Rejeita $H_0$ | Rejeita $H_0$ | **Estacionaria com componente de tendencia** (diferenciacao pode ser necessaria) |
| Nao rejeita $H_0$ | Nao rejeita $H_0$ | **Inconclusivo** (aumentar amostra ou aplicar transformacao conservadora) |

#### 2.4.3 Protocolo de Validacao

1. Executar ADF e KPSS em **cada feature** usando os dados da janela de treino.
2. Se alguma feature falhar no teste de estacionariedade, aplicar diferenciacao de primeira ordem: $\Delta x_t = x_t - x_{t-1}$.
3. Repetir os testes na serie diferenciada.
4. **Logar os resultados** (estatistica de teste, p-valor, conclusao) para auditoria.
5. Se apos diferenciacao a serie continuar nao-estacionaria, alertar e investigar manualmente (possivel quebra estrutural nos dados).

---

## 3. Configuracao do HMM

### 3.1 Numero de Estados

**Recomendacao: $K = 3$ estados (baseline), testar $K \in \{2, 3, 4\}$.**

#### Justificativa para 3 estados

Tres estados correspondem aos regimes fundamentais observados empiricamente em mercados financeiros:

| Estado | Interpretacao esperada | Caracteristicas estatisticas esperadas |
|--------|----------------------|---------------------------------------|
| Estado 1: **Low Volatility / Lateral** | Mercado sem direcao definida, consolidacao | $\mu_{r} \approx 0$, $\sigma$ baixo, volume abaixo da media |
| Estado 2: **Trending (direcional)** | Tendencia sustentada (alta ou baixa) | $|\mu_{r}| > 0$ (positivo ou negativo), $\sigma$ moderado, momentum z-score elevado em valor absoluto |
| Estado 3: **High Volatility / Stress** | Volatilidade extrema, panic, squeezes | $\sigma$ muito elevado, range expandido, volume acima da media, retornos com dispersao alta |

**Por que nao 2 estados?** Dois estados (high vol / low vol) e demasiado simplista -- nao distingue entre tendencia direcional calma e lateralizacao, que requerem estrategias de trading diferentes.

**Por que nao 4+ estados?** Com os dados disponiveis (max 60 dias de 5min ~ 4.320 candles uteis), 4 estados implica estimar:
- 4 vetores de media ($\boldsymbol{\mu}_j \in \mathbb{R}^5$) = 20 parametros
- 4 matrizes de covariancia (diagonal: 20 parametros, full: 60 parametros)
- Matriz de transicao $4 \times 4$ = 12 parametros livres
- **Total (diagonal)**: ~52 parametros, **Total (full)**: ~92 parametros

Com ~4.000 observacoes e 92 parametros (full covariance), a razao dados/parametros e ~43:1 -- aceitavel mas sem margem. Com 4 estados full, a razao cai para ~43:1 que esta no limite. O principio de parcimonia (Occam's razor) favorece $K=3$.

**Procedimento**: Treinar modelos com $K \in \{2, 3, 4\}$ e selecionar via criterio de informacao (ver Seccao 3.4).

### 3.2 Tipo de Covariancia

**Recomendacao: `covariance_type = 'full'` como baseline, `'diagonal'` como fallback.**

#### Analise das opcoes

| Tipo | Parametros por estado (5 features) | Vantagens | Desvantagens |
|------|-----------------------------------|-----------|--------------|
| `full` | 15 (triangulo inferior de $5 \times 5$) | Captura correlacoes cruzadas entre features (ex: alta volatilidade correlacionada com volume elevado) | Mais parametros, risco de overfitting com poucos dados |
| `diagonal` | 5 | Robusto com poucos dados, menos parametros | Ignora correlacoes cruzadas entre features |
| `tied` | 15 (compartilhados entre estados) | Menos parametros total | Assume que a estrutura de correlacao e identica em todos os regimes -- irrealista para mercados financeiros |

**Justificativa para `full`**: As features sao **deliberadamente correlacionadas** entre si (retorno e momento, volatilidade e range sao intrinsecamente ligados). Ignorar estas correlacoes (`diagonal`) descartaria informacao discriminativa relevante. A covariancia `full` permite que cada estado tenha sua propria estrutura de dependencia, o que e empiricamente correto: em regimes de stress, a correlacao volume-volatilidade tende a ser mais forte.

**Fallback para `diagonal`**: Se o BIC indicar overfitting com `full`, ou se o treino nao convergir com `full`, usar `diagonal`.

**Nunca usar `tied`**: A hipotese de covariancia identica entre regimes contradiz a premissa fundamental do modelo (regimes tem propriedades estatisticas distintas).

### 3.3 Estrategia de Inicializacao

**Recomendacao: Inicializacao via K-Means seguida de estimacao de parametros.**

#### Protocolo de inicializacao

1. **K-Means clustering** no conjunto de treino (features ja transformadas e estacionarias) com $K$ clusters.
2. Para cada cluster $j$:
   - $\boldsymbol{\mu}_j^{(0)} = $ centroide do cluster $j$.
   - $\boldsymbol{\Sigma}_j^{(0)} = $ covariancia empirica das observacoes no cluster $j$.
3. **Matriz de transicao** inicial $\mathbf{A}^{(0)}$: Usar a sequencia temporal de labels do K-Means para estimar as probabilidades de transicao empiricas:

$$a_{ij}^{(0)} = \frac{\#\{t : s_t = i \text{ e } s_{t+1} = j\}}{\#\{t : s_t = i\}}$$

onde $s_t$ e o label do K-Means no instante $t$.

4. **Distribuicao inicial** $\boldsymbol{\pi}^{(0)}$: Proporcional a frequencia de cada cluster.

**Justificativa**: K-Means fornece uma inicializacao **informada** que esta proxima da solucao final, reduzindo significativamente:
- O numero de iteracoes necessarias para convergencia do EM (Baum-Welch).
- O risco de convergir para minimos locais subotimos (o EM e garantido convergir apenas para um otimo local, nunca global).

**Procedimento adicional para robustez**: Executar o HMM com **10 inicializacoes aleatorias** alem da inicializacao K-Means, e selecionar o modelo com maior log-likelihood na janela de treino. Isto mitiga o risco de K-Means forncer uma inicializacao subotima. Implementar via parametro `n_init` do `hmmlearn`.

### 3.4 Criterios de Convergencia

| Parametro | Valor recomendado | Justificativa |
|-----------|-------------------|---------------|
| Tolerancia (`tol`) | $10^{-4}$ | Equilibrio entre precisao e tempo de computacao. Valores menores ($10^{-6}$) raramente melhoram o fit e aumentam o tempo. |
| Max iteracoes (`n_iter`) | 200 | Suficiente para convergencia com boa inicializacao. Se nao convergir em 200, a inicializacao provavelmente e ruim. |
| `n_init` | 10 | Numero de restarts aleatorios (alem da inicializacao K-Means). |

**Criterio de parada**: O algoritmo EM para quando a variacao do log-likelihood entre iteracoes consecutivas e menor que `tol`:

$$|\mathcal{L}^{(k+1)} - \mathcal{L}^{(k)}| < \text{tol}$$

onde $\mathcal{L}^{(k)} = \ln P(\mathbf{X} | \boldsymbol{\theta}^{(k)})$ e o log-likelihood dos dados de treino no passo $k$.

### 3.5 Selecao do Numero Otimo de Estados

**Criterio primario: BIC (Bayesian Information Criterion).**

$$\text{BIC} = -2 \mathcal{L} + p \cdot \ln(T)$$

onde:
- $\mathcal{L}$ e o log-likelihood maximizado.
- $p$ e o numero total de parametros livres do modelo.
- $T$ e o numero de observacoes.

**Contagem de parametros** para HMM Gaussiano com $K$ estados e $D=5$ features:

| Componente | Parametros (`full` cov) | Parametros (`diag` cov) |
|------------|------------------------|------------------------|
| Medias $\boldsymbol{\mu}_j$ | $K \times D$ | $K \times D$ |
| Covariancias $\boldsymbol{\Sigma}_j$ | $K \times \frac{D(D+1)}{2}$ | $K \times D$ |
| Transicoes $\mathbf{A}$ | $K \times (K-1)$ | $K \times (K-1)$ |
| Iniciais $\boldsymbol{\pi}$ | $K - 1$ | $K - 1$ |
| **Total ($K=3$, `full`)** | **65** | **41** |

**Procedimento**:
1. Treinar modelos para $K \in \{2, 3, 4\}$ com covariancia `full` e `diagonal`.
2. Calcular BIC para cada combinacao $(K, \text{cov\_type})$.
3. Selecionar o modelo com **menor BIC**.
4. Se a diferenca de BIC entre dois modelos for < 10 (escala de Kass & Raftery, 1995), preferir o modelo mais simples (menor $K$).

**Criterio secundario: AIC (Akaike Information Criterion)** como referencia cruzada:

$$\text{AIC} = -2\mathcal{L} + 2p$$

O AIC tende a selecionar modelos com mais estados que o BIC. Se AIC e BIC discordarem, **preferir o BIC** (mais conservador, penaliza mais a complexidade).

**Cross-validation temporal** como criterio terciario (ver Seccao 4.2 para detalhes): calcular o log-likelihood out-of-sample em janelas walk-forward e verificar se modelos mais complexos generalizam melhor.

### 3.6 Interpretacao Esperada dos Estados

Apos o treino, o engenheiro deve verificar que os estados descobertos pelo HMM sao **interpretaveis** e consistentes com a intuicao financeira. Os estados NAO vem rotulados -- devem ser identificados pelas suas caracteristicas estatisticas.

**Procedimento de rotulagem**:

1. Calcular as medias $\boldsymbol{\mu}_j$ de cada estado.
2. Ordenar os estados por $\sigma_j$ (componente de volatilidade da media).
3. Atribuir rotulos provisorios:

| Rotulo | Criterio de identificacao |
|--------|--------------------------|
| **Low Vol / Lateral** | Menor $\tilde{\sigma}_j$ entre os estados; $\mu_{r_j} \approx 0$ |
| **Trending** | $\tilde{\sigma}_j$ intermediario; $|z_{\text{mom}_j}|$ mais alto |
| **High Vol / Stress** | Maior $\tilde{\sigma}_j$; $z_{\text{NR}_j}$ elevado; $\tilde{v}_j$ elevado |

4. Validar qualitativamente contra eventos conhecidos (ver Seccao 6.2).

**ATENCAO**: Se os estados nao forem interpretaveis (ex: dois estados com distribuicoes muito similares), isto e um **red flag** indicando que:
- O numero de estados esta excessivo (reduzir $K$).
- As features nao contem informacao discriminativa suficiente.
- O modelo esta capturando ruido em vez de regimes reais.

---

## 4. Prevencao de Look-Ahead Bias

Esta seccao e **critica**. Look-ahead bias e o erro mais insidioso em financas quantitativas e invalida completamente qualquer resultado de backtest ou avaliacao de modelo.

### 4.1 Principio Fundamental

**Em nenhum momento do pipeline, uma observacao no instante $t$ pode utilizar informacao de qualquer instante $t' > t$.**

Isto aplica-se a:
- Calculo de features.
- Normalizacao e transformacao de dados.
- Treino do modelo.
- Avaliacao e inferencia.

### 4.2 Esquema Walk-Forward

O modelo deve ser treinado e avaliado usando um esquema de **walk-forward expanding window** (ou sliding window):

```
Tempo:  |------- Treino -------|----- Validacao -----|--- Step ---|

Fold 1: |======= TREINO =======|===== VALID =========|
Fold 2: |=========== TREINO ===========|===== VALID =========|
Fold 3: |=============== TREINO ===============|===== VALID =========|
```

#### Parametros recomendados

| Parametro | Timeframe 1min | Timeframe 5min |
|-----------|---------------|---------------|
| Janela de treino minima | 1 dia (~390 candles uteis B3, 9:00-17:30) | 5 dias (~390 candles uteis) |
| Janela de validacao | 1 dia | 1 dia (~78 candles) |
| Step size (incremento) | 1 dia | 1 dia |
| Embargo period (gap entre treino e validacao) | 60 candles (~1 hora) | 12 candles (~1 hora) |
| Tipo de janela | **Expanding** (preferido inicialmente) | **Expanding** |

**Embargo period**: E um gap temporal entre o fim da janela de treino e o inicio da janela de validacao. Serve para evitar **data leakage** causado por:
- Features rolling que usam informacao de candles adjacentes ao fim do treino.
- Autocorrelacao serial dos retornos de curto prazo.

Recomendacao: embargo de pelo menos $L_{\max} = 60$ candles, que e o tamanho da maior janela rolling utilizada.

**Expanding vs. Sliding window**:
- **Expanding** (recomendado inicialmente): A janela de treino cresce a cada fold, incorporando todos os dados historicos disponiveis. Vantagem: mais dados de treino. Desvantagem: dados antigos podem nao ser representativos do regime atual.
- **Sliding** (considerar se resultados com expanding forem ruins): Janela de treino de tamanho fixo que "desliza" no tempo. Vantagem: adapta-se a mudancas estruturais. Desvantagem: menos dados de treino.

### 4.3 O que NUNCA Fazer (Lista de Armadilhas)

1. **NUNCA** normalizar features usando estatisticas de toda a serie temporal.
   - ERRADO: calcular z-score com media e desvio padrao de todo o dataset.
   - CORRETO: usar rolling z-score calculado apenas com dados passados (como especificado nas formulas da Seccao 1).

2. **NUNCA** treinar o HMM em todo o dataset e depois avaliar no mesmo dataset.
   - Isto garante overfitting e resultados enganosos.

3. **NUNCA** selecionar o numero de estados $K$ baseado na performance no dataset completo.
   - Selecionar $K$ apenas com dados da janela de treino (BIC calculado no treino).

4. **NUNCA** usar o futuro para preencher valores faltantes.
   - ERRADO: forward-fill que usa o proximo valor valido.
   - CORRETO: usar apenas backward-fill (ultimo valor conhecido) ou descartar a observacao.
   - **Nota**: `pandas.DataFrame.fillna(method='ffill')` faz forward-fill no tempo (usa valor anterior), o que e correto. O nome e confuso. O que e proibido e `fillna(method='bfill')` que usa valores futuros.

5. **NUNCA** calcular features que requerem dados futuros.
   - Exemplos proibidos: centered moving average, leading indicators que usam precos futuros.
   - Todas as janelas rolling devem ser **backward-looking** (apenas dados $\leq t$).

6. **NUNCA** usar retornos futuros para rotular ou validar estados.
   - A avaliacao do HMM deve ser baseada em estabilidade estatistica e interpretabilidade, nao em "qual estado gerou mais retorno" (isto seria target leakage).

7. **NUNCA** ajustar hiperparametros na janela de validacao.
   - Se necessario ajustar hiperparametros, usar um esquema de nested walk-forward (walk-forward dentro do walk-forward).

8. **NUNCA** ignorar os gaps de mercado (overnight, fim de semana).
   - O retorno do primeiro candle do dia ($r_{t_{\text{open}}}$) inclui o gap overnight, que tem natureza diferente dos retornos intradiarios.
   - Opcoes: (a) excluir o primeiro candle de cada dia, ou (b) tratar como missing e preencher com zero, ou (c) modelar separadamente.
   - **Recomendacao**: Excluir o primeiro candle de cada sessao e tratar cada dia como uma sequencia independente para o HMM.

### 4.4 Causalidade Temporal nas Features Rolling

Todas as features rolling devem usar janelas **estritamente causais**:

$$f_t = g(x_t, x_{t-1}, x_{t-2}, \ldots, x_{t-N+1})$$

e nunca:

$$f_t = g(\ldots, x_{t+1}, x_{t+2}, \ldots) \quad \text{(PROIBIDO)}$$

**Verificacao de implementacao**: Para cada feature rolling, o engenheiro deve confirmar que a funcao `pandas.DataFrame.rolling()` e chamada com os parametros corretos:
- `min_periods` deve ser igual ao tamanho da janela (para evitar calculos parciais no inicio da serie).
- O valor padrao de `center=False` deve ser mantido (nunca usar `center=True`, que centra a janela e usa dados futuros).

---

## 5. Pipeline Recomendado (Fluxo Logico)

### 5.1 Diagrama do Fluxo

```
OHLCV Raw (Parquet)
       |
       v
[1] PRE-PROCESSAMENTO
       |-- Remover candles do pre/pos mercado (se existirem)
       |-- Remover primeiro candle de cada sessao (gap overnight)
       |-- Validar integridade: sem NaN em OHLC, Volume >= 0
       |-- Verificar: High >= max(Open, Close), Low <= min(Open, Close)
       |
       v
[2] CALCULO DE FEATURES (estritamente causal)
       |-- r_t = ln(C_t / C_{t-1})                    [log returns]
       |-- sigma_t = sqrt(sum(r^2, N))                 [realized vol]
       |-- sigma_tilde_t = ln(sigma_t + eps)            [log realized vol]
       |-- mom_t = sum(r, M)                            [momentum]
       |-- z_mom_t = (mom_t - mu_L) / sigma_L           [momentum z-score]
       |-- z_mom_clip = clip(z_mom, -5, +5)             [clipped]
       |-- vrel_t = V_t / mean(V, K candles anteriores) [relative volume]
       |-- v_tilde_t = ln(vrel_t + eps)                 [log relative vol]
       |-- NR_t = ln(H_t / L_t)                         [normalized range]
       |-- z_NR_t = (NR_t - mu_L) / sigma_L             [range z-score]
       |-- z_NR_clip = clip(z_NR, -5, +5)               [clipped]
       |
       v
[3] DESCARTE DO WARM-UP PERIOD
       |-- Remover as primeiras L_max = 60 observacoes
       |
       v
[4] VALIDACAO DE ESTACIONARIEDADE
       |-- Executar ADF + KPSS em cada feature
       |-- Se falhar: aplicar diferenciacao e re-testar
       |-- Logar resultados
       |
       v
[5] WALK-FORWARD SPLIT
       |-- Definir janela de treino (expanding)
       |-- Definir janela de validacao
       |-- Aplicar embargo period entre treino e validacao
       |
       v
[6] TREINO DO HMM (por fold)
       |-- Inicializacao via K-Means nos dados de treino
       |-- Fit do HMM Gaussiano (Baum-Welch / EM)
       |-- Registar: log-likelihood, BIC, AIC, parametros
       |
       v
[7] SELECAO DE MODELO
       |-- Comparar K in {2, 3, 4} via BIC no treino
       |-- Comparar covariance_type in {full, diagonal}
       |-- Selecionar modelo com menor BIC (ou mais parcimonioso se BIC similar)
       |
       v
[8] INFERENCIA E DECODIFICACAO
       |-- Aplicar modelo treinado na janela de validacao
       |-- Decodificar estados via algoritmo de Viterbi: s* = argmax P(S | X)
       |-- Obter probabilidades posteriores via Forward-Backward: P(s_t = j | X)
       |
       v
[9] AVALIACAO E ROTULAGEM
       |-- Calcular metricas de estabilidade (Seccao 6)
       |-- Rotular estados por caracteristicas estatisticas
       |-- Sanity checks qualitativos
       |
       v
[10] OUTPUT: Serie temporal de regimes rotulados
       |-- Coluna: regime_label (int: 0, 1, 2)
       |-- Coluna: regime_name (str: 'low_vol', 'trending', 'high_vol')
       |-- Coluna: regime_prob_0, regime_prob_1, regime_prob_2 (probabilidades posteriores)
       |-- Salvar em Parquet em data/processed/
```

### 5.2 Ordem de Operacoes -- Regras Criticas

1. **Features ANTES de split**: As features sao calculadas em toda a serie temporal, MAS usando apenas janelas backward-looking. Isto e permitido porque cada feature no instante $t$ depende apenas de dados $\leq t$.

2. **Testes de estacionariedade APENAS na janela de treino**: Os testes ADF/KPSS devem ser executados nos dados de treino de cada fold, nao no dataset completo.

3. **Treino do HMM APENAS na janela de treino**: Nunca expor dados de validacao ao algoritmo EM.

4. **Inferencia na validacao SEM re-treino**: Usar o modelo treinado "como esta" para decodificar os estados na janela de validacao.

### 5.3 Pontos de Validacao Intermediarios

Em cada etapa do pipeline, o engenheiro deve implementar assertions e logs:

| Etapa | Validacao |
|-------|----------|
| [1] Pre-processamento | Nenhuma linha com NaN em OHLC; Volume nao-negativo; $H \geq L$; timestamps monotonicos crescentes |
| [2] Features | Dimensao do vetor de features = 5; sem NaN apos warm-up; retornos $|r_t| < 0.5$ (retornos intradiarios acima de 50% sao suspeitos) |
| [3] Warm-up | Numero de observacoes restantes >= 100 (minimo para treino do HMM) |
| [4] Estacionariedade | Todas as 5 features passam no teste combinado ADF+KPSS |
| [5] Split | Janela de treino >= 200 observacoes; janela de validacao >= 50 observacoes; embargo period aplicado |
| [6] Treino | EM convergiu (num iteracoes < max_iter); log-likelihood e finito e nao-NaN; covariancias sao positivas definidas |
| [7] Selecao | BIC calculado corretamente; pelo menos 2 modelos comparados |
| [8] Inferencia | Probabilidades posteriores somam ~1.0 (tolerancia $10^{-6}$) para cada $t$; estados atribuidos estao em $\{0, 1, \ldots, K-1\}$ |

---

## 6. Metricas de Avaliacao

### 6.1 Metricas Quantitativas

#### 6.1.1 Estabilidade dos Estados (Regime Persistence)

A duração media de permanencia em cada estado e uma medida critica. Se o modelo troca de estado a cada candle, esta capturando ruido, nao regimes.

**Duracao media de um estado $j$** (derivada da matriz de transicao):

$$\bar{d}_j = \frac{1}{1 - a_{jj}}$$

onde $a_{jj}$ e a probabilidade de permanecer no estado $j$.

**Criterio de aceitacao**:
- Para 1min: $\bar{d}_j \geq 10$ candles (permanencia media de pelo menos ~10 minutos).
- Para 5min: $\bar{d}_j \geq 5$ candles (permanencia media de pelo menos ~25 minutos).
- Se $\bar{d}_j < 5$ candles para qualquer estado, o modelo esta **instavel** e provavelmente captura ruido.

**Duracao media empirica** (verificacao alternativa):

$$\bar{d}_j^{\text{emp}} = \frac{\text{numero total de candles no estado } j}{\text{numero de segmentos contiguos no estado } j}$$

#### 6.1.2 Frequencia de Transicao

**Numero medio de transicoes por dia**:

$$\text{trans\_rate} = \frac{\#\{t : s_t \neq s_{t-1}\}}{D}$$

onde $D$ e o numero de dias de trading.

**Criterio de aceitacao**:
- Para 1min: $\leq 15$ transicoes por dia.
- Para 5min: $\leq 8$ transicoes por dia.
- Transicoes excessivas indicam instabilidade do modelo.

#### 6.1.3 Distribuicao dos Estados

**Proporcao temporal de cada estado**:

$$p_j = \frac{\#\{t : s_t = j\}}{T}$$

**Criterio**: Nenhum estado deve ter $p_j < 0.05$ (menos de 5% do tempo). Um estado raramente visitado e estatisticamente irrelevante e sugere que $K$ esta sobre-especificado.

#### 6.1.4 Separacao dos Estados (Discriminability)

Para verificar se os estados sao estatisticamente distintos, calcular:

**Divergencia de Kullback-Leibler entre distribuicoes de emissao**:

$$D_{\text{KL}}(\mathcal{N}_i \| \mathcal{N}_j) = \frac{1}{2}\left[\text{tr}(\boldsymbol{\Sigma}_j^{-1}\boldsymbol{\Sigma}_i) + (\boldsymbol{\mu}_j - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_j^{-1}(\boldsymbol{\mu}_j - \boldsymbol{\mu}_i) - D + \ln\frac{|\boldsymbol{\Sigma}_j|}{|\boldsymbol{\Sigma}_i|}\right]$$

**Criterio**: $D_{\text{KL}} > 0.5$ entre todos os pares de estados. Se dois estados tiverem $D_{\text{KL}} < 0.1$, sao praticamente indistinguiveis e devem ser fundidos (reduzir $K$).

**Alternativa pragmatica**: Se a divergencia KL for complexa de implementar, verificar a **separabilidade visual**: plotar as distribuicoes 2D (scatter plots de pares de features) coloridas por estado. Os clusters devem ser visualmente distinguiveis.

#### 6.1.5 Log-Likelihood Out-of-Sample

Calcular o log-likelihood medio por observacao na janela de validacao:

$$\bar{\mathcal{L}}_{\text{val}} = \frac{1}{T_{\text{val}}} \ln P(\mathbf{X}_{\text{val}} | \boldsymbol{\theta}_{\text{train}})$$

**Criterios**:
- $\bar{\mathcal{L}}_{\text{val}}$ deve ser **estavel** entre folds walk-forward (variancia baixa).
- $\bar{\mathcal{L}}_{\text{val}}$ nao deve ser **muito inferior** a $\bar{\mathcal{L}}_{\text{train}}$. Um gap grande indica overfitting.
- Comparar com um modelo baseline de 1 estado (Gaussiana unica): se o HMM com $K > 1$ nao superar significativamente o baseline em log-likelihood out-of-sample, os regimes nao sao detectaveis nos dados.

### 6.2 Sanity Checks Qualitativos

O engenheiro deve verificar visualmente que os regimes detectados correspondem a intuicao financeira:

1. **Overlay de regimes sobre o grafico de precos**: Colorir o fundo do grafico de precos com a cor do regime detectado. Os periodos de "High Vol / Stress" devem coincidir visualmente com movimentos abruptos e aumento de amplitude.

2. **Verificacao contra eventos conhecidos** (para WDO/USDBRL):
   - Dias de decisao de SELIC (Copom): espera-se aumento de volatilidade pre e pos anuncio.
   - Divulgacao de dados de emprego (payroll EUA, CAGED Brasil): volatilidade na USDBRL.
   - Intervencoes do Banco Central no cambio: movimentos direcionais abruptos.

3. **Verificacao contra eventos conhecidos** (para WIN/IBOVESPA):
   - Circuit breakers historicos devem coincidir com estado "High Vol / Stress".
   - Periodos de feriado prolongado: volume reduzido, esperado estado "Low Vol / Lateral".

4. **Padrao intradiario**: Verificar se o modelo nao esta sistematicamente atribuindo regimes com base na hora do dia (ex: abertura sempre = "High Vol"). Isto indicaria que a desazonalizacao do volume e insuficiente ou que features adicionais de controle sao necessarias.

### 6.3 Resumo dos Criterios de Aceitacao / Rejeicao

| Criterio | Aceitar | Rejeitar |
|----------|---------|----------|
| Duracao media dos estados (1min) | $\geq 10$ candles cada | Qualquer estado com $< 5$ candles |
| Duracao media dos estados (5min) | $\geq 5$ candles cada | Qualquer estado com $< 3$ candles |
| Transicoes por dia (1min) | $\leq 15$ | $> 30$ |
| Transicoes por dia (5min) | $\leq 8$ | $> 15$ |
| Proporcao minima por estado | $\geq 5\%$ | $< 5\%$ |
| Separacao KL entre estados | $> 0.5$ todos os pares | $< 0.1$ qualquer par |
| Log-likelihood out-of-sample | Estavel entre folds, superior ao baseline | Inferior ao baseline ou alta variancia |
| Interpretabilidade | Estados correspondem a regimes financeiros reconheciveis | Estados indistinguiveis ou sem sentido financeiro |

---

## 7. Riscos e Limitacoes

### 7.1 Limitacoes do HMM Gaussiano

#### 7.1.1 Caudas Leves da Gaussiana

A distribuicao Normal subestima a probabilidade de eventos extremos (fat tails). Retornos financeiros tipicamente seguem distribuicoes leptocurticas com excess kurtosis $\kappa > 0$ (frequentemente $\kappa \in [3, 50]$ para dados intradiarios).

**Consequencia pratica**: O HMM Gaussiano pode:
- Subestimar a probabilidade de transicao para o estado "High Vol / Stress" durante eventos de cauda.
- Atribuir incorretamente retornos extremos ao estado mais volatil, mesmo quando a dinamica subjacente e diferente (ex: flash crash vs. volatilidade sustentada).

**Mitigacao parcial**: A feature de log realized volatility ja captura parte da variacao de cauda. Adicionalmente, o clipping dos z-scores em $[-5, +5]$ limita a influencia de outliers na estimacao de parametros.

#### 7.1.2 Independencia Condicional das Emissoes

O HMM assume que, dado o estado oculto $s_t$, a observacao $\mathbf{x}_t$ e independente das observacoes anteriores $\mathbf{x}_{t-1}, \mathbf{x}_{t-2}, \ldots$. Isto ignora a autocorrelacao dos retornos ao quadrado (efeito ARCH/GARCH), que e uma propriedade empirica robusta dos dados financeiros.

**Consequencia**: O modelo pode nao capturar adequadamente a **persistencia** da volatilidade dentro de um regime.

**Mitigacao parcial**: A feature de realized volatility (janela rolling) codifica parcialmente a memoria da volatilidade.

#### 7.1.3 Cadeia de Markov de Primeira Ordem

O HMM assume que o estado atual $s_t$ depende apenas do estado anterior $s_{t-1}$ (propriedade de Markov). Na realidade, regimes de mercado podem ter memoria mais longa (ex: a probabilidade de sair de um estado de crise depende de quanto tempo se esta em crise).

**Consequencia**: O modelo pode subestimar a persistencia de regimes longos.

#### 7.1.4 Numero Fixo de Estados

O HMM assume um numero fixo $K$ de regimes. Na realidade, novos regimes podem emergir (ex: COVID-19 criou dinamicas de mercado sem precedente historico).

### 7.2 Cenarios de Falha do Modelo

| Cenario | Por que o modelo falha | Sinal de alerta |
|---------|----------------------|-----------------|
| **Mudanca estrutural** (ex: nova politica monetaria do BC) | Parametros estimados em dados historicos nao representam o novo regime | Log-likelihood out-of-sample cai abruptamente |
| **Flash crash / evento de cauda extrema** | Observacao muito fora da distribuicao de qualquer estado | Probabilidade posterior maxima < 0.5 (modelo "indeciso") |
| **Mercado sem liquidez** (pre-feriado, pos-horario) | Volume proximo de zero distorce a feature de volume relativo | Volume relativo com valores extremos ($\tilde{v}_t \ll -3$) |
| **Dados corrompidos / gaps** | Missing data ou erros no feed criam features artificialmente anomalas | NaN nas features ou retornos $|r_t| > 0.1$ no 1min |
| **Overfitting no treino** | Modelo memoriza padroes espurios da janela de treino | $\bar{\mathcal{L}}_{\text{train}} \gg \bar{\mathcal{L}}_{\text{val}}$ |

### 7.3 Alternativas Futuras (Roadmap)

Para sprints futuros, considerar as seguintes extensoes por ordem de prioridade:

#### 7.3.1 HMM com Distribuicao t-Student (Prioridade Alta)

Substituir a emissao Gaussiana por uma t-Student multivariada:

$$b_j(\mathbf{x}) = \text{MVT}(\mathbf{x}; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j, \nu_j)$$

onde $\nu_j$ sao os graus de liberdade (estimados por estado). Isto modela explicitamente caudas pesadas.

**Beneficio**: Melhor captura de eventos extremos sem atribuir tudo ao estado "High Vol".

#### 7.3.2 Markov-Switching Autoregressive Model (MS-AR) (Prioridade Media)

Modelo de Hamilton (1989) que incorpora dependencia autorregressiva nos retornos:

$$r_t = \mu_{s_t} + \sum_{k=1}^{p} \phi_{k, s_t} r_{t-k} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_{s_t}^2)$$

**Beneficio**: Captura autocorrelacao de retornos dentro de cada regime, melhorando a modelacao de tendencias.

#### 7.3.3 Sticky HDP-HMM (Prioridade Baixa)

Hierarchical Dirichlet Process HMM (Fox et al., 2011): um HMM nao-parametrico que infere automaticamente o numero de estados a partir dos dados.

**Beneficio**: Elimina a necessidade de selecao manual de $K$.

**Custo**: Significativamente mais complexo de implementar e computar; requer sampling MCMC.

#### 7.3.4 Features Adicionais a Investigar (Sprint 3+)

- **Order Flow Imbalance** (se dados de book estiverem disponiveis): $\text{OFI}_t = V_t^{\text{bid}} - V_t^{\text{ask}}$
- **VWAP Deviation**: $\text{dev}_t = \frac{C_t - \text{VWAP}_t}{\text{VWAP}_t}$
- **Realized Skewness** (terceiro momento): indicador de assimetria nas distribuicoes de retornos

---

## 8. Referencias

1. **Hamilton, J.D. (1989)**. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384. -- Trabalho fundacional sobre regime-switching models.

2. **Ang, A. & Bekaert, G. (2002)**. "International Asset Allocation With Regime Shifts." *Review of Financial Studies*, 15(4), 1137-1187. -- Aplicacao de HMM a alocacao de ativos com regimes.

3. **Andersen, T.G. & Bollerslev, T. (1998)**. "Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts." *International Economic Review*, 39(4), 885-905. -- Realized volatility em alta frequencia.

4. **Parkinson, M. (1980)**. "The Extreme Value Method for Estimating the Variance of the Rate of Return." *Journal of Business*, 53(1), 61-65. -- Estimacao de volatilidade via range High-Low.

5. **Rabiner, L.R. (1989)**. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2), 257-286. -- Tutorial classico sobre HMMs.

6. **Fox, E.B., Sudderth, E.B., Jordan, M.I. & Willsky, A.S. (2011)**. "A Sticky HDP-HMM With Application to Speaker Diarization." *Annals of Applied Statistics*, 5(2A), 1020-1056. -- HMM nao-parametrico com persistencia.

7. **Kass, R.E. & Raftery, A.E. (1995)**. "Bayes Factors." *Journal of the American Statistical Association*, 90(430), 773-795. -- Criterios de selecao de modelos (BIC).

8. **de Prado, M.L. (2018)**. *Advances in Financial Machine Learning*. Wiley. -- Purged walk-forward cross-validation, prevencao de look-ahead bias em ML financeiro.

---

## Apendice A: Tabela Resumo de Parametros

| Parametro | Valor (1min) | Valor (5min) | Descricao |
|-----------|-------------|-------------|-----------|
| $N$ (realized vol window) | 20 | 20 | Janela para calculo de realized volatility |
| $M$ (momentum window) | 10 | 10 | Janela para calculo de momento |
| $K$ (volume rel window) | 60 | 60 | Janela para media movel de volume |
| $L$ (z-score window) | 60 | 60 | Janela para normalizacao z-score rolling |
| $\epsilon$ (log safety) | $10^{-10}$ | $10^{-10}$ | Constante para evitar log(0) |
| Clipping z-score | $[-5, +5]$ | $[-5, +5]$ | Limites de truncamento |
| Warm-up descartado | 60 candles | 60 candles | Periodo de aquecimento das janelas |
| $K_{\text{estados}}$ (baseline) | 3 | 3 | Numero de estados do HMM |
| $K_{\text{estados}}$ (testar) | {2, 3, 4} | {2, 3, 4} | Range para selecao via BIC |
| Covariancia | full (fallback: diag) | full (fallback: diag) | Tipo de covariancia do HMM |
| EM tolerancia | $10^{-4}$ | $10^{-4}$ | Criterio de parada do EM |
| EM max iteracoes | 200 | 200 | Limite de iteracoes |
| `n_init` | 10 | 10 | Restarts aleatorios |
| Treino minimo | 1 dia (~390 candles) | 5 dias (~390 candles) | Janela minima de treino |
| Validacao | 1 dia | 1 dia (~78 candles) | Janela de validacao |
| Embargo | 60 candles | 12 candles | Gap treino-validacao |
| Walk-forward step | 1 dia | 1 dia | Incremento por fold |
| Walk-forward type | Expanding | Expanding | Tipo de janela |

---

## Apendice B: Checklist de Implementacao

- [ ] Features calculadas com formulas exatas da Seccao 1
- [ ] Todas as janelas rolling sao backward-looking (`center=False`)
- [ ] `min_periods` igual ao tamanho da janela em todos os `.rolling()`
- [ ] Primeiro candle de cada sessao descartado
- [ ] Warm-up period de 60 candles descartado
- [ ] Testes ADF + KPSS executados e logados para cada feature
- [ ] Walk-forward com embargo period implementado
- [ ] HMM treinado apenas na janela de treino (nunca validacao)
- [ ] BIC calculado para K in {2, 3, 4} com cov in {full, diagonal}
- [ ] Modelo selecionado por menor BIC (parcimonia em caso de empate)
- [ ] Inicializacao K-Means + 10 restarts aleatorios
- [ ] Metricas de estabilidade calculadas (duracao media, transicoes/dia)
- [ ] Separacao entre estados verificada (KL divergence > 0.5)
- [ ] Sanity check visual contra eventos de mercado conhecidos
- [ ] Log-likelihood out-of-sample comparado com baseline (1 estado)
- [ ] Resultados salvos em Parquet com colunas de regime e probabilidades
- [ ] Nenhum look-ahead bias em nenhuma etapa (verificar com o Quant Researcher)
