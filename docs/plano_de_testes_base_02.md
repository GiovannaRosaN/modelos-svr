# Plano de testes — Base 02 (Metro Interstate Traffic Volume)

Este documento **planeja** os testes do notebook `notebooks/02_base_02.ipynb`; não afirma que todos já estejam automatizados. O mesmo contrato de saída pode ser reaproveitado nas outras quatro bases e no notebook `06_comparacao_geral.ipynb`.

## Objetivo e referência atual

Garantir que os **quatro modelos obrigatórios do Grupo 2** (SARIMAX, Holt-Winters, Random Forest e SVR) sejam comparados sem vazamento temporal, no mesmo horizonte de **uma hora**, com treino e validação anteriores ao teste e métricas calculadas nas mesmas horas observadas. A Decision Tree permanece como experimento adicional solicitado pelo grupo; seus resultados devem ser identificados como extras e não alterar o ranking oficial exigido pela atividade.

| Item | Referência da execução atual |
| --- | --- |
| Entrada | 24.096 horas de 2016-01-01 a 2018-09-30; 1.012 alvos ausentes |
| Recorte | 80% desenvolvimento / 20% teste sobre a grade horária |
| Teste | 4.820 horas de 2018-03-14 04:00 a 2018-09-30 23:00; 4.805 alvos observados |
| Validação | Dois folds consecutivos de 168 horas antes do teste |
| Previsão | `data_prevista = data_origem + 1 hora`; janela de ajuste de 1.344 horas; reajuste a cada 168 horas |
| Aleatoriedade | `random_state = 67` nos modelos que a utilizam |
| Registro de entrada | SHA-256 e versões em `results/tuning/base_02_manifest.json` |

Esses números são uma **fotografia da versão atual dos dados**, não uma regra para bases futuras. Uma mudança intencional na entrada exige atualizar a referência e documentar o motivo. O tempo de execução e a posição exata entre Random Forest e Holt-Winters não são critérios de aprovação: seus MAEs atuais são próximos (185,62 e 186,83 veículos/h).

## Casos de teste

`P0` bloqueia a entrega do **escopo correspondente**. `I01` e `G01`–`G03` se aplicam à entrega final do grupo; enquanto as outras bases não estiverem prontas, seu estado é **pendente**, não falha da Base 02. `P1` exige revisão antes da apresentação final. Os testes negativos devem usar uma cópia pequena dos dados, nunca alterar o CSV original.

| ID | Prioridade | Verificação | Critério de aprovação |
| --- | --- | --- | --- |
| D01 | P0 | Ler o CSV e conferir as colunas usadas (`date_time`, `traffic_volume`, clima e condição do tempo). | Leitura sem erro; datas válidas; alvo observado numérico e não negativo; nenhuma coluna obrigatória ausente. |
| D02 | P0 | Auditar a grade horária e as lacunas após a regularização. | Datas únicas e crescentes, intervalo de exatamente 1 hora e nenhuma hora criada silenciosamente por `asfreq`; `NaN` do alvo preservado, sem imputação para calcular erros. |
| D03 | P1 | Revisar documentação e exploração da base. | Fonte, cobertura, frequência, unidade do alvo e dicionário das variáveis externas explícitos; ausentes, duplicidades, irregularidades, atípicos, gráficos do alvo e das exógenas e decisões de limpeza discutidos. |
| D04 | P0 na entrega final | Confirmar o congelamento da base com a turma e eventuais aprovações. | O hash da versão usada coincide com o registro comum das cinco bases; qualquer alteração de base, horizonte, protocolo ou conjunto oficial de modelos tem aprovação documentada. O hash local, sozinho, prova reprodutibilidade, não igualdade com a versão da turma. |
| S01 | P1 | Revisar STL de tendência, sazonalidade, resíduo e força sazonal. | Além dos gráficos e cálculos, o texto interpreta períodos de crescimento/queda, padrão sazonal, resíduos e o valor da força calculada, usando somente dados anteriores ao teste. |
| T01 | P0 | Recalcular o corte 80/20 e os dois folds. | Desenvolvimento e teste não se sobrepõem; ambos os folds e toda seleção de hiperparâmetros terminam antes do teste; cada ajuste termina antes da primeira previsão do bloco. |
| T02 | P0 | Inspecionar cada linha do log de ajustes. | Janela com no máximo 1.344 horas, datas em ordem e início de previsão posterior ao fim do ajuste; reajustes do teste a cada 168 horas, salvo o bloco final parcial. |
| F01 | P0 | Verificar alinhamento de `lag1`, `lag24`, `lag168`, médias móveis e clima defasado em datas normais e perto de lacunas. | A feature da hora `t` não usa `traffic_volume[t]` nem clima medido em `t`; `lag1` corresponde a `t-1`. Lacunas mantêm sua posição no relógio. |
| F02 | P0 | Teste de causalidade: modificar alvo e clima em `t` e depois recalcular as features. | Features e previsão emitida para `t` não mudam; previsões posteriores **podem** mudar depois que novas observações forem incorporadas. |
| F03 | P0 | Instrumentar imputação, codificação e escala dentro de cada bloco. | `fit` recebe apenas linhas anteriores à previsão; validação/teste usam somente `transform`. Categoria climática inédita não causa falha nem reajuste com dados futuros. |
| F04 | P1 | Conferir disponibilidade operacional das exógenas. | Cada variável externa tem definição e momento de disponibilidade documentados; clima observado entra apenas defasado, calendário pode ser conhecido antecipadamente e nenhum valor futuro observado entra como entrada. |
| M01 | P0 | Conferir previsão sequencial dos quatro modelos obrigatórios e da Decision Tree extra em um bloco curto. | Uma previsão por hora e por modelo; nenhuma previsão usa o alvo da própria hora antes de emiti-la; previsões finitas e não negativas. Horas sem alvo ainda recebem previsão. |
| M02 | P0 | Repetir as verificações específicas já presentes no notebook. | SARIMAX em lote coincide com previsão/atualização hora a hora; Holt-Winters recursivo coincide com `statsmodels` e sua primeira previsão independe do primeiro alvo futuro. |
| M03 | P1 | Conferir condições de ajuste por modelo. | SARIMAX recebe exógenas alinhadas e conhecidas na origem; Holt-Winters dispõe de pelo menos dois ciclos sazonais completos; avisos e convergência ficam no log. |
| H01 | P1 | Conferir busca e justificativa de hiperparâmetros. | Espaço, método, configurações testadas e escolhidas estão documentados; SARIMAX discute `p,d,q,P,D,Q,m`, AIC/BIC e exógenas; os outros três modelos têm os parâmetros mínimos do enunciado examinados. |
| O01 | P0 | Conferir `results/predictions/base_02_predictions.csv`. | Os quatro modelos obrigatórios estão presentes; Decision Tree pode constar como extra. Datas iguais e sem duplicatas por modelo; `horizonte = 1`; `data_prevista - data_origem = 1 hora`; na entrada atual, 4.820 previsões por modelo. |
| O02 | P0 | Recalcular `residuo = valor_real - valor_previsto`, MAE, RMSE e viés diretamente do CSV de previsões. | Resíduo vazio quando o alvo falta; mesmas 4.805 datas observadas para os quatro modelos obrigatórios e também para o extra, se incluído; métricas recalculadas coincidem com o CSV de métricas até tolerância de `1e-6`. |
| O03 | P1 | Comparar com o sazonal ingênuo de 168 horas e recalcular ranking. | Benchmark e modelos avaliados somente nas **mesmas** datas disponíveis; ranking **oficial** ordena apenas os quatro modelos exigidos pelo MAE e mostra Decision Tree separadamente. Ao menos o melhor modelo deve superar o benchmark; se não, investigar e registrar. |
| A01 | P1 | Revisar resíduos, ACF, Ljung-Box e interpretação. | Resíduos são calculados fora da amostra para cada modelo. Se ACF/Ljung-Box usarem apenas um trecho horário contínuo, informar datas, quantidade, percentual de cobertura e limitação de representatividade; a tabela consolidada de Ljung-Box aparece no corpo do relatório. Um p-valor baixo é achado a relatar, não falha de execução. |
| A02 | P1 | Revisar interpretação dos modelos. | RF e SVR têm importância de features; exógenas e disponibilidade temporal são destacadas; SARIMAX discute sinal, magnitude e significância de coeficientes; Holt-Winters discute nível, tendência e sazonalidade. |
| E01 | P0 | Executar o notebook do zero em ambiente com as dependências instaladas. | Todas as células terminam sem exceção; arquivos de saída e manifesto são gerados; hash de entrada, `random_state`, datas e versões ficam rastreáveis. |
| E02 | P1 | Repetir a execução sem mudar dados, código ou versões. | Contagens, datas, parâmetros selecionados e métricas são reproduzidos dentro de tolerância numérica. Não comparar tempo de execução como valor fixo. |
| I01 | P0 | Depois que as cinco bases estiverem prontas, executar `06_comparacao_geral.ipynb`. | Lê o contrato das cinco bases sem ajustes manuais; calcula ranking dos **quatro modelos oficiais** em cada base, quantidade de vitórias e posição média, sem somar ou tirar média direta de MAEs em escalas diferentes. |
| G01 | P1 | Revisar estudo didático de especialização do Grupo 2. | O relatório explica SVR (intuição, hipóteses, preparação, hiperparâmetros, busca, importância, limites e resultados nas cinco bases), além de citar bibliotecas e fontes. |
| G02 | P0 | Conferir os dois formatos do relatório e a apresentação. | HTML paginado e autocontido e PDF possuem o mesmo conteúdo mínimo das 14 seções; a apresentação usa o HTML e todos os integrantes têm participação prevista. |
| G03 | P0 | Conferir gestão e pacote de entrega. | Há registro diário de demandas de todos os integrantes, com carga e complexidade separadas; o ZIP único contém os relatórios, fonte, códigos, bases/fontes, MAEs consolidados e registro. O prazo deve ser conferido na Plataforma Odete. |

## Testes negativos e casos de borda

1. **Timestamp duplicado ou faltando:** a auditoria deve falhar com mensagem clara; não agregar ou inventar horas automaticamente. Se a entrada vier fora de ordem, a ordenação precisa ser registrada e verificada antes da modelagem.
2. **Coluna obrigatória ausente, data inválida ou alvo negativo:** interromper antes do ajuste.
3. **Alvo ausente no teste:** gerar previsão, mas não preencher o valor real nem incluí-lo em MAE/RMSE/viés.
4. **Categoria climática nova ou clima ausente:** manter o alinhamento horário; a codificação deve tratar a categoria nova e a imputação deve usar só o treino.
5. **Futuro adulterado:** mudar valores em `t` ou depois não pode alterar a previsão já emitida para `t`.
6. **Janela curta para sazonalidade:** falhar explicitamente ou escolher configuração válida; não ajustar Holt-Winters sazonal com menos de dois ciclos.

## Execução e evidências

1. **Smoke rápido:** validar notebook/entrada, um fold e um bloco curto por modelo em uma cópia de teste. Os cenários pequenos devem rodar antes da execução integral; não substituem a avaliação final.
2. **Execução integral:** da raiz do repositório, executar `python -m nbconvert --to notebook --execute notebooks/02_base_02.ipynb --output base_02_execucao_teste.ipynb --output-dir results/test_runs --ExecutePreprocessor.timeout=1800` e conferir os casos P0 e P1 sem sobrescrever o notebook de trabalho.
3. **Conferência independente:** recalcular métricas a partir de `results/predictions/base_02_predictions.csv`, comparar com `results/metrics/base_02_metrics.csv` e guardar manifesto, log de ajustes, relatório de resíduos e análise textual como evidências.

O responsável pela Base 02 executa e registra os resultados; o sexto integrante revisa os testes de vazamento, as métricas e a integração com as demais bases. Uma falha P0 bloqueia a conclusão da base. Uma falha P1 deve ser corrigida ou aparecer explicitamente como limitação justificada no relatório. A implementação sugerida é uma suíte `tests/test_base_02.py` para as verificações rápidas e um teste de integração separado para a execução completa do notebook.
