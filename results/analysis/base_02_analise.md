# Análise da Base 02

## Resultado principal

Random Forest obteve o menor MAE: 185.62 veículos/h. O segundo foi Holt-Winters, com 186.83. A redução relativa foi 0.6%. A comparação usa 4805 horas com alvo observado entre 2018-03-14 04:00:00 e 2018-09-30 23:00:00.

## Leitura por modelo

- Random Forest: MAE 185.62, RMSE 319.87, viés -11.54 (superestimação média), tempo de teste 64.8 s. O MAE na validação foi 199.64.
- Holt-Winters: MAE 186.83, RMSE 296.54, viés -1.31 (superestimação média), tempo de teste 0.6 s. O MAE na validação foi 223.65.
- Decision Tree: MAE 249.94, RMSE 451.98, viés -12.55 (superestimação média), tempo de teste 1.9 s. O MAE na validação foi 259.18.
- SARIMAX: MAE 291.97, RMSE 417.90, viés -5.32 (superestimação média), tempo de teste 575.1 s. O MAE na validação foi 341.34.
- SVR: MAE 314.04, RMSE 494.93, viés 46.72 (subestimação média), tempo de teste 5.3 s. O MAE na validação foi 403.42.

A diferença entre RMSE e MAE indica quanto erros grandes pesam no resultado. A diferença entre validação e teste também depende da época do ano e da dificuldade dos períodos; ela sozinha não comprova sobreajuste.

## Estabilidade e custo

O menor tempo foi de Holt-Winters: 0.6 s. Os tempos dependem da máquina e incluem os reajustes semanais; a busca não está incluída.
Vitórias por mês do teste: Holt-Winters: 4, Random Forest: 3. Meses extremos podem ser parciais.
Para o vencedor, a hora de maior MAE foi 7h (320.81 veículos/h).

## Resíduos

ACF e Ljung-Box usam 1588 horas contínuas, de 2018-06-02 03:00:00 até 2018-08-07 06:00:00.
- SVR: evidência de autocorrelação nas defasagens 24, 48, 168 horas.
- Decision Tree: evidência de autocorrelação nas defasagens 24, 48, 168 horas.
- Random Forest: evidência de autocorrelação nas defasagens 24, 48, 168 horas.
- SARIMAX: evidência de autocorrelação nas defasagens 24, 48, 168 horas.
- Holt-Winters: evidência de autocorrelação nas defasagens 24, 48, 168 horas.

## O que os modelos aproveitaram

- SVR: maiores aumentos de MAE na permutação: volume_lag168 (288.5), volume_lag1 (228.0), variacao_1h (185.6).
- Decision Tree: maiores aumentos de MAE na permutação: volume_lag1 (762.7), hora_cos (473.0), volume_lag168 (208.7).
- Random Forest: maiores aumentos de MAE na permutação: volume_lag1 (867.6), volume_lag168 (369.1), hora_cos (218.0).

As árvores representam relações não lineares; a floresta combina árvores para reduzir variabilidade. O SVR usa uma margem de tolerância e distâncias após padronização. SARIMAX combina dependência temporal e exógenas; Holt-Winters usa somente o histórico do alvo. Estas diferenças ajudam a formular hipóteses sobre os resultados, mas a importância de uma feature não demonstra causalidade.

## Limitações e recomendação

Usar Random Forest como referência inicial para este horizonte, acompanhando erro por horário e mudanças ao longo do tempo. A recomendação vale para a busca limitada e a janela móvel de oito semanas deste experimento.
- O teste tem observações ausentes. Elas não foram imputadas para calcular erros.
- Holt-Winters usa preenchimento causal apenas na cópia de ajuste. SARIMAX recebe NaN; modelos tabulares usam alvos observados.
- O resultado pressupõe acesso ao volume e ao clima da hora anterior. Atrasos reais de publicação exigem rever os lags.
- Há medições meteorológicas extremas. A transformação logarítmica reduz escala, mas não comprova a qualidade dessas medições.
- A busca usa duas semanas de validação, insuficientes para cobrir todas as estações. Novas decisões exigem outra validação, mantendo o teste protegido.
- A sazonalidade e os coeficientes podem variar ao longo do tempo; conclusões são restritas à Base 02.
- Os dados são horas locais sem fuso. A regularização existente não permite resolver ambiguidades de horário de verão.
- O diagnóstico de resíduos usa um único trecho contínuo e não todo o teste.
- Não foi calculado intervalo de confiança da diferença de MAE; uma diferença pequena pode não ser conclusiva.
- SARIMAX: convergência confirmada nos 29 ajustes do teste.

## Referências

- UCI: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- statsmodels, SARIMAXResults.extend: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.extend.html
- statsmodels, Holt-Winters: https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
- scikit-learn, SVR: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
- scikit-learn, permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html