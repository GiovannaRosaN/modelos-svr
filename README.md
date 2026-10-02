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

## Pipeline da Base 02

O notebook `notebooks/02_base_02.ipynb` contém o pipeline Metro Interstate Traffic Volume com SVR, Decision Tree, Random Forest, SARIMAX e Holt-Winters. Decision Tree é uma comparação adicional aos quatro modelos obrigatórios do Grupo 2.

A organização segue as mesmas 16 etapas da Pipeline 01, adaptadas ao tráfego horário: identificação, auditoria, limpeza, exploração, STL, features, divisão temporal, seleção de parâmetros, previsões por modelo, comparação, resíduos, interpretação e exportação. Cada modelo tem sua própria seção de previsão. O ranking oficial é calculado somente para os quatro modelos obrigatórios; Decision Tree é identificada como extra.

- Horizonte: uma hora, com grade temporal regular e métricas nas mesmas horas com alvo observado.
- Recorte: 80% das horas para desenvolvimento e 20% para teste final.
- Validação: duas semanas anteriores ao teste; hiperparâmetros escolhidos por MAE.
- Walk-forward: janela expansiva desde a primeira hora, usando todo o histórico anterior à origem em cada ajuste; reajuste semanal e atualização de histórico a cada hora.
- Random state: 67. Imputação, categorias e escala são ajustadas somente no treino de cada janela.
- Resultados: previsões, MAE/RMSE/viés, tempos, resíduos, ACF, Ljung-Box e importância das features. A análise textual é salva em `results/analysis/base_02_analise.md` e exibida no próprio notebook.

Execute da raiz do repositório:

```powershell
python -m nbconvert --to notebook --execute notebooks/02_base_02.ipynb --output base_02_execucao_teste.ipynb --output-dir results/test_runs --ExecutePreprocessor.timeout=-1
python -m unittest discover -s tests -v
```

`data/processed/base_02.parquet` preserva a grade horária e as lacunas do alvo. Para modelos tabulares, as linhas sem alvo são excluídas somente do ajuste; para avaliação, todos os modelos compartilham a mesma máscara de alvos observados. O manifesto e os registros de busca, avisos e convergência ficam em `results/tuning/`.

O notebook de código pode ser mantido sem saídas embutidas. A suíte confere o notebook executado em `results/test_runs/` e verifica se a evidência corresponde ao código atual antes de auditar as métricas.

A janela expansiva substitui o limite anterior de oito semanas para todos os cinco modelos. Resultados produzidos com o protocolo antigo não representam este código e precisam ser regenerados; não reutilize seus MAEs na análise atual. No Jupyter, reinicie o kernel e execute todas as células, incluindo a seleção dos hiperparâmetros. A execução integral pode ser consideravelmente mais demorada, por isso o comando não impõe limite de tempo por célula.

Para verificar código, dados e causalidade sem executar a busca e sem reutilizar resultados antigos:

```powershell
python -m unittest discover -s tests -v -k Base02Protocol -k Base02Data
```

A auditoria completa das saídas (`Base02Exports`) só será aprovada depois da nova execução integral.
