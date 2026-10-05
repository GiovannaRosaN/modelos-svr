# Análise da Base 02

## Resultado principal

Random Forest obteve o menor MAE: 134.74 veículos/h. O segundo foi SVR, com 143.51. A redução relativa foi 6.1%. A comparação usa 4805 horas com alvo observado entre 2018-03-14 04:00:00 e 2018-09-30 23:00:00.

## Leitura por modelo

- Random Forest: MAE 134.74, RMSE 214.03, viés -9.67 (superestimação média), tempo de teste 1766.9 s. O MAE na validação foi 177.15.
- SVR: MAE 143.51, RMSE 223.66, viés -0.65 (superestimação média), tempo de teste 1838.9 s. O MAE na validação foi 160.53.
- Decision Tree: MAE 173.15, RMSE 275.41, viés -8.60 (superestimação média), tempo de teste 26.8 s. O MAE na validação foi 228.05.
- Holt-Winters: MAE 199.14, RMSE 317.28, viés -2.33 (superestimação média), tempo de teste 1.3 s. O MAE na validação foi 225.89.
- SARIMAX: MAE 290.41, RMSE 418.08, viés 1.48 (subestimação média), tempo de teste 3615.7 s. O MAE na validação foi 339.30.

A diferença entre RMSE e MAE indica quanto erros grandes pesam no resultado. A diferença entre validação e teste também depende da época do ano e da dificuldade dos períodos; ela sozinha não comprova sobreajuste.

## Estabilidade e custo

O menor tempo foi de Holt-Winters: 1.3 s. Os tempos dependem da máquina e incluem os reajustes semanais; a busca não está incluída.
Vitórias por mês do teste: Random Forest: 6, SVR: 1. Meses extremos podem ser parciais.
Para o vencedor, a hora de maior MAE foi 23h (227.58 veículos/h).

## Resíduos

ACF e Ljung-Box usam 1588 horas contínuas, de 2018-06-02 03:00:00 até 2018-08-07 06:00:00.
- SVR: evidência de autocorrelação nas defasagens 24, 48, 168 horas.
- Decision Tree: evidência de autocorrelação nas defasagens 24, 48, 168 horas.
- Random Forest: evidência de autocorrelação nas defasagens 24, 48, 168 horas.
- SARIMAX: evidência de autocorrelação nas defasagens 24, 48, 168 horas.
- Holt-Winters: evidência de autocorrelação nas defasagens 24, 48, 168 horas.

## O que os modelos aproveitaram

- SVR: maiores aumentos de MAE na permutação: volume_lag1 (329.1), volume_lag24 (305.2), hora_cos (248.1).
- Decision Tree: maiores aumentos de MAE na permutação: volume_lag1 (785.5), hora_cos (578.2), volume_lag168 (356.9).
- Random Forest: maiores aumentos de MAE na permutação: volume_lag1 (747.4), hora_cos (454.5), volume_lag168 (201.5).

## Experimento adicional

Decision Tree: MAE 173.15 veículos/h. Está fora do ranking oficial.

## Limitações e causalidade

A sequência segue os 16 passos da Base 01, adaptados à previsão horária. O recorte é cronológico 80/20, com janela expansiva desde a primeira hora e reajuste semanal. Clima entra defasado; o alvo da hora prevista não entra nas features. Imputação e escala são aprendidas apenas no treino de cada janela.
A busca considera uma grade pequena e somente duas semanas de validação. O melhor candidato não é um ótimo global, e pode não representar outras estações do ano.
ACF/Ljung-Box cobrem 33.0% das horas observadas no teste; o trecho contínuo não representa automaticamente todo o período.
As horas sem alvo real recebem previsão, mas não participam das métricas. Valores meteorológicos extremos requerem investigação da fonte; IQR não justifica remoção automática.

## Benchmark semanal

                 modelo  mae_mesmas_datas    n
          Random Forest            134.86 4788
                    SVR            143.47 4788
          Decision Tree            172.92 4788
           Holt-Winters            199.41 4788
                SARIMAX            290.49 4788
Sazonal ingênuo (168 h)            293.65 4788