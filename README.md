# Trabalho de Series Temporais - Grupo 2

Projeto organizado em cinco pipelines independentes, um para cada base. Cada notebook deve executar SARIMAX, Holt-Winters, Random Forest e SVR usando o mesmo protocolo walk-forward.

## Responsabilidades

- Integrantes 1 a 5: uma base completa por pessoa.
- Integrante 6: padronizacao metodologica, revisao contra vazamento, consolidacao dos resultados e relatorio final.

## Notebooks

- `01_base_01.ipynb` ate `05_base_05.ipynb`: analise completa de cada base.
- `06_comparacao_geral.ipynb`: consolidacao de MAE, rankings, vitorias, Ljung-Box e conclusoes.

## Contrato de saida de cada notebook

Cada pipeline deve salvar:

- `results/predictions/base_XX_predictions.csv`
- `results/metrics/base_XX_metrics.csv`
- `results/residuals/base_XX_residuals.csv`
- `results/feature_importance/base_XX_feature_importance.csv`
- graficos em `results/figures/base_XX/`
- base tratada em `data/processed/base_XX.parquet`

As previsoes devem conter: `base`, `modelo`, `data_origem`, `data_prevista`, `horizonte`, `valor_real`, `valor_previsto` e `residuo`.

## Ordem obrigatoria em cada pipeline

1. Identificacao e documentacao da base.
2. Carregamento e auditoria temporal.
3. Limpeza e regularizacao.
4. Analise exploratoria.
5. Decomposicao STL e forca da sazonalidade.
6. Feature engineering sem vazamento.
7. Definicao do walk-forward.
8. Otimizacao apenas com treino/validacao.
9. SARIMAX.
10. Holt-Winters.
11. Random Forest.
12. SVR com padronizacao dentro de cada treino.
13. MAE e ranking.
14. Residuos, ACF e Ljung-Box.
15. Importancia das features.
16. Exportacao dos resultados.

## Execucao

Instale as dependencias de `requirements.txt`, preencha a configuracao no inicio de cada notebook e execute todas as celulas em ordem. O notebook de comparacao deve ser executado somente depois dos cinco pipelines.
