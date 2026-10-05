"""Auditoria independente do código, dos dados e dos artefatos da Base 02.

Execute da raiz do repositório: python -m unittest discover -s tests -v
Só código/dados: python -m unittest discover -s tests -v -k Base02Protocol -k Base02Data
A execução integral do notebook é uma etapa separada do plano de testes.
As classes Base02Protocol e Base02Data não dependem de previsões antigas.
Base02Exports exige evidência nova, com janela expansiva e código atualizado.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import unittest
import warnings
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02_base_02.ipynb"
RAW = ROOT / "data/raw/base_02/Metro_Interstate_Traffic_Volume_2016_2018_regularizada_sem_imputacao.csv"
RESULTS = ROOT / "results"
FROZEN_SHA256 = "3c105848c040bbbcf3deb07da26a5ab3a1487543c342dc0501ba97c63679bca4"
OFFICIAL_MODELS = {"SVR", "Random Forest", "SARIMAX", "Holt-Winters"}
OPTIONAL_MODELS = {"Decision Tree"}
EXPECTED_HOURS = 24_096
EXPECTED_MISSING_TARGETS = 1_012
TRAIN_RATIO = 0.80
VALIDATION_HOURS = 168
N_FOLDS = 2
TRAIN_START_POS = 0
REFIT_HOURS = 168
HOUR = pd.Timedelta(hours=1)


def notebook_cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def notebook_function(name: str, namespace: dict):
    """Compila só a função solicitada; não executa o notebook inteiro."""
    for cell in notebook_cells():
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
                exec(compile(module, str(NOTEBOOK), "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"Função {name} não encontrada no notebook")


def ingestion_code():
    """Célula de carga auditada isoladamente, com entrada sintética em memória."""
    for cell in notebook_cells():
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "raw = pd.read_csv(DATA_PATH" in source:
                return compile(source, str(NOTEBOOK), "exec")
    raise AssertionError("Célula de auditoria de entrada não encontrada")


class Base02Protocol(unittest.TestCase):
    """Contrato de treino verificado sem executar buscas ou ler resultados."""

    def test_p01_full_history_configuration(self):
        """P01: nenhum limite móvel; corte e reajuste semanal preservados."""
        assignments = {}
        for cell in notebook_cells():
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignments[target.id] = node.value
        expected = {"TRAIN_START_POS": TRAIN_START_POS, "TRAIN_RATIO": TRAIN_RATIO,
                    "REFIT_HOURS": REFIT_HOURS, "VALIDATION_HOURS": VALIDATION_HOURS,
                    "N_FOLDS": N_FOLDS, "RANDOM_STATE": 67, "HORIZON": 1}
        for name, value in expected.items():
            with self.subTest(configuracao=name):
                self.assertIn(name, assignments)
                self.assertEqual(ast.literal_eval(assignments[name]), value)
        self.assertNotIn("TRAIN_WINDOW", assignments,
                         "O limite de oito semanas não pertence ao protocolo expansivo.")

    def test_p02_all_models_receive_the_entire_available_prefix(self):
        """P02: cinco modelos partem da hora zero, incluindo as primeiras 168 h."""
        dates = pd.date_range("2020-01-01", periods=2_400, freq="h")
        target = pd.Series(np.arange(2_400, dtype=float), index=dates)
        target.iloc[10] = np.nan  # faltante preservado no prefixo
        features = pd.DataFrame({"exog": np.arange(2_400, dtype=float)}, index=dates)
        received = []

        def tabular(name, params, ytr, Xtr, Xfuture):
            received.append((name, ytr.copy(), Xtr.copy(), Xfuture.copy()))
            return np.ones(len(Xfuture)), None, {}

        def sarimax(params, ytr, yfuture, Xtr, Xfuture):
            received.append(("SARIMAX", ytr.copy(), Xtr.copy(), Xfuture.copy()))
            pd.testing.assert_index_equal(yfuture.index, Xfuture.index)
            return np.ones(len(yfuture)), None, {}

        def holt_winters(params, start, first_train_hour, yfuture):
            self.assertEqual(first_train_hour, TRAIN_START_POS)
            received.append(("Holt-Winters", target.iloc[first_train_hour:start].copy(),
                             features.iloc[first_train_hour:start].copy(),
                             features.loc[yfuture.index].copy()))
            return np.ones(len(yfuture)), None, {}

        namespace = {"np": np, "y": target, "X": features,
                     "TRAIN_START_POS": TRAIN_START_POS, "warnings": warnings,
                     "perf_counter": perf_counter, "fit_logs": [],
                     "forecast_tabular": tabular, "forecast_sarimax": sarimax,
                     "forecast_holt_winters": holt_winters}
        forecast = notebook_function("forecast_block", namespace)
        for name in sorted(OFFICIAL_MODELS | OPTIONAL_MODELS):
            for start in (400, 2_000):
                with self.subTest(modelo=name, origem=start):
                    pred = forecast(name, {}, start, start + 3, "teste_unitario")
                    recorded_name, ytr, Xtr, Xfuture = received[-1]
                    self.assertEqual(recorded_name, name)
                    pd.testing.assert_series_equal(ytr, target.iloc[:start])
                    pd.testing.assert_frame_equal(Xtr, features.iloc[:start])
                    pd.testing.assert_frame_equal(Xfuture, features.iloc[start:start+3])
                    self.assertEqual(len(pred), 3)
                    log = namespace["fit_logs"][-1]
                    self.assertEqual(log["treino_inicio"], dates[0])
                    self.assertEqual(log["treino_fim"], dates[start-1])
                    self.assertEqual(log["n_treino_horas"], start)
                    self.assertEqual(log["n_treino_observado"], start - 1)
                    self.assertEqual(log["janela_tipo"], "expansiva")

    def test_p03_manifest_source_describes_the_expanding_protocol(self):
        """P03: manifesto futuro deve distinguir as saídas novas das móveis antigas."""
        manifest_node = None
        for cell in notebook_cells():
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "manifest"
                    for target in node.targets
                ):
                    manifest_node = node.value
        self.assertIsInstance(manifest_node, ast.Dict)
        fields = {ast.literal_eval(key): value
                  for key, value in zip(manifest_node.keys, manifest_node.values)}
        expected = {"janela_tipo": "expansiva", "janela_ajuste_horas": None,
                    "treino_inicio_posicao": TRAIN_START_POS}
        for name, value in expected.items():
            with self.subTest(campo=name):
                self.assertIn(name, fields)
                expression = ast.fix_missing_locations(ast.Expression(fields[name]))
                actual = eval(compile(expression, str(NOTEBOOK), "eval"),
                              {"TRAIN_START_POS": TRAIN_START_POS})
                self.assertEqual(actual, value)
        self.assertIn("one-step", ast.literal_eval(fields["protocolo"]))
        self.assertIn("janela expansiva", ast.literal_eval(fields["protocolo"]))


class Base02Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(RAW, parse_dates=["date_time"])
        cls.frame = cls.raw.set_index("date_time")
        cls.y = cls.frame["traffic_volume"].astype(float)
        cls.split = int(len(cls.raw) * TRAIN_RATIO)

    def test_d01_schema_target_and_frozen_input(self):
        """D01: entrada, schema, domínio do alvo e SHA independente."""
        self.assertEqual(hashlib.sha256(RAW.read_bytes()).hexdigest(), FROZEN_SHA256)
        needed = {"date_time", "traffic_volume", "traffic_volume_original",
                  "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main"}
        self.assertFalse(needed - set(self.raw), f"Colunas ausentes: {needed - set(self.raw)}")
        self.assertFalse(self.raw.date_time.isna().any())
        parsed_target = pd.to_numeric(self.raw.traffic_volume, errors="coerce")
        self.assertTrue(parsed_target.notna().eq(self.raw.traffic_volume.notna()).all())
        self.assertTrue(parsed_target.dropna().ge(0).all())
        pd.testing.assert_series_equal(self.raw.traffic_volume, self.raw.traffic_volume_original,
                                       check_names=False)

    def test_d02_regular_grid_and_preserved_missing_values(self):
        """D02: a entrada congelada já é horária; as lacunas de alvo continuam NaN."""
        self.assertEqual(len(self.raw), EXPECTED_HOURS)
        self.assertEqual(int(self.y.isna().sum()), EXPECTED_MISSING_TARGETS)
        self.assertTrue(self.raw.date_time.is_monotonic_increasing)
        self.assertFalse(self.raw.date_time.duplicated().any())
        self.assertTrue(self.raw.date_time.diff().dropna().eq(HOUR).all())
        self.assertEqual(len(self.frame.asfreq("h")), len(self.frame))
        self.assertEqual(self.raw.date_time.iloc[0], pd.Timestamp("2016-01-01 00:00:00"))
        self.assertEqual(self.raw.date_time.iloc[-1], pd.Timestamp("2018-09-30 23:00:00"))

    def test_d01_d02_reject_invalid_synthetic_inputs(self):
        """Casos negativos: duplicata, hora ausente e alvo negativo falham na carga."""
        base = pd.DataFrame({
            "date_time": pd.date_range("2020-01-01", periods=5, freq="h"),
            "traffic_volume": [1.0, 2.0, 3.0, 4.0, 5.0],
            "traffic_volume_original": [1.0, 2.0, 3.0, 4.0, 5.0],
            "temp": [273.0] * 5,
            "rain_1h": [0.0] * 5,
            "snow_1h": [0.0] * 5,
            "clouds_all": [0.0] * 5,
            "weather_main": ["Clear"] * 5,
        })
        duplicate = base.copy()
        duplicate.loc[2, "date_time"] = duplicate.loc[1, "date_time"]
        missing_hour = base.drop(index=2).reset_index(drop=True)
        negative = base.copy()
        negative.loc[3, ["traffic_volume", "traffic_volume_original"]] = -1.0
        code = ingestion_code()
        for label, bad in (("duplicata", duplicate),
                           ("hora ausente", missing_hour), ("alvo negativo", negative)):
            with self.subTest(caso=label):
                namespace = {"pd": pd, "DATA_PATH": RAW, "DATETIME_COL": "date_time",
                             "FREQ": "h", "TARGET": "traffic_volume"}
                with patch.object(pd, "read_csv", return_value=bad):
                    with self.assertRaises(AssertionError):
                        exec(code, namespace)

    def test_t01_temporal_cut_and_validation_folds(self):
        """T01: recorte 80/20 e dois folds estritamente anteriores ao teste."""
        self.assertEqual(self.split, 19_276)
        self.assertEqual(len(self.raw) - self.split, 4_820)
        self.assertEqual(self.raw.date_time.iloc[self.split], pd.Timestamp("2018-03-14 04:00:00"))
        self.assertEqual(int(self.y.iloc[self.split:].notna().sum()), 4_805)
        folds = [(self.split - (N_FOLDS - k) * VALIDATION_HOURS,
                  self.split - (N_FOLDS - k - 1) * VALIDATION_HOURS)
                 for k in range(N_FOLDS)]
        self.assertEqual(folds[-1][1], self.split)
        self.assertEqual(folds[0][1], folds[1][0])
        for start, stop in folds:
            self.assertGreater(start, TRAIN_START_POS)
            self.assertLess(start, stop)
            self.assertLessEqual(stop, self.split)
            self.assertEqual(stop - start, VALIDATION_HOURS)

    def test_f01_lags_rolling_and_missing_hour_alignment(self):
        """F01: defasagens e médias usam relógio real, inclusive perto de NaN."""
        make_features = notebook_function("make_features",
                                          {"np": np, "pd": pd, "TARGET": "traffic_volume"})
        missing = np.flatnonzero(self.y.isna().to_numpy())
        gap = int(next(i for i in missing if i >= 200))
        for sample in [self.frame.iloc[:400], self.frame.iloc[gap-200:gap+4]]:
            features = make_features(sample)
            target = sample.traffic_volume
            for lag in (1, 24, 168):
                pd.testing.assert_series_equal(features[f"volume_lag{lag}"],
                                               target.shift(lag), check_names=False)
            for pos in (199, 200):
                expected = target.iloc[max(0, pos-24):pos].mean()
                actual = features.iloc[pos]["media_24h"]
                if np.isfinite(expected):
                    self.assertAlmostEqual(actual, expected, places=8)
            weather = sample.temp.where(sample.temp.ge(0)).ffill(limit=6).shift(1)
            pd.testing.assert_series_equal(features.temp_lag1, weather, check_names=False)
        sample = self.frame.iloc[gap-200:gap+4]
        features = make_features(sample)
        self.assertTrue(math.isnan(features.iloc[201].volume_lag1))

    def test_f02_feature_and_first_forecast_causality(self):
        """F02/F03: adulterar t e o futuro não altera a previsão já emitida para t."""
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor

        make_features = notebook_function("make_features",
                                          {"np": np, "pd": pd, "TARGET": "traffic_volume"})
        sample = self.frame.iloc[:550].copy()
        t = 400
        baseline = make_features(sample)
        changed = sample.copy()
        changed.loc[changed.index[t]:, "traffic_volume"] = 999_999.0
        changed.loc[changed.index[t]:, ["temp", "rain_1h", "snow_1h", "clouds_all"]] = 999.0
        changed.loc[changed.index[t]:, "weather_main"] = "CATEGORIA_FUTURA"
        altered = make_features(changed)
        pd.testing.assert_frame_equal(baseline.iloc[:t+1], altered.iloc[:t+1])

        cat_cols = ["weather_main_lag1"]
        num_cols = [col for col in baseline if col not in cat_cols]
        namespace = {"Pipeline": Pipeline, "SimpleImputer": SimpleImputer,
                     "StandardScaler": StandardScaler, "ColumnTransformer": ColumnTransformer,
                     "OneHotEncoder": OneHotEncoder, "SVR": SVR,
                     "DecisionTreeRegressor": DecisionTreeRegressor,
                     "RandomForestRegressor": RandomForestRegressor,
                     "NUM_COLS": num_cols, "CAT_COLS": cat_cols, "RANDOM_STATE": 67}
        notebook_function("tabular_model", namespace)
        forecast = notebook_function("forecast_tabular", namespace)
        train_y = sample.traffic_volume.iloc[:t]
        train_x = baseline.iloc[:t]
        original_pred, fitted, _ = forecast("Decision Tree", {"max_depth": 3},
                                           train_y, train_x, baseline.iloc[t:t+1])
        altered_pred, _, _ = forecast("Decision Tree", {"max_depth": 3},
                                      train_y, train_x, altered.iloc[t:t+1])
        np.testing.assert_array_equal(original_pred, altered_pred)
        prep = fitted.named_steps["prep"]
        train_median = train_x.loc[train_y.notna(), "temp_lag1"].median()
        temp_idx = num_cols.index("temp_lag1")
        self.assertAlmostEqual(prep.named_transformers_["num"].named_steps["imputer"]
                               .statistics_[temp_idx], train_median, places=8)
        unseen = baseline.iloc[t:t+1].copy()
        unseen["weather_main_lag1"] = "CATEGORIA_NUNCA_VISTA"
        self.assertNotIn("CATEGORIA_NUNCA_VISTA",
                         prep.named_transformers_["cat"].categories_[0])
        self.assertTrue(np.isfinite(fitted.predict(unseen)).all())
        missing_weather = unseen.copy()
        missing_weather["temp_lag1"] = np.nan
        missing_weather["weather_main_lag1"] = "Desconhecido"
        self.assertTrue(np.isfinite(fitted.predict(missing_weather)).all())


class Base02Exports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest_path = RESULTS / "tuning/base_02_manifest.json"
        if not manifest_path.is_file():
            raise AssertionError("Execute novamente o notebook: falta o manifesto expansivo.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not (manifest.get("janela_tipo") == "expansiva"
                and manifest.get("janela_ajuste_horas", "ausente") is None
                and manifest.get("treino_inicio_posicao") == TRAIN_START_POS):
            raise AssertionError(
                "Resultados incompatíveis com o código atual: ainda não há evidência "
                "de execução com todo o histórico (janela expansiva desde a posição 0). "
                "Execute novamente o notebook antes de auditar as saídas.")
        executed_path = RESULTS / "test_runs/base_02_execucao_teste.ipynb"
        if not executed_path.is_file():
            raise AssertionError("Falta a evidência executada do notebook expansivo.")
        executed = json.loads(executed_path.read_text(encoding="utf-8"))
        source_codes = ["".join(c["source"]).strip() for c in notebook_cells()
                        if c["cell_type"] == "code"]
        executed_codes = ["".join(c["source"]).strip() for c in executed["cells"]
                          if c["cell_type"] == "code"]
        if source_codes != executed_codes:
            raise AssertionError("A evidência de execução não corresponde ao código expansivo atual.")
        cls.raw = pd.read_csv(RAW, parse_dates=["date_time"])
        cls.y = cls.raw.set_index("date_time").traffic_volume.astype(float)
        cls.test_start = int(len(cls.y) * TRAIN_RATIO)
        cls.expected_dates = pd.DatetimeIndex(cls.raw.date_time.iloc[cls.test_start:])
        cls.pred = pd.read_csv(RESULTS / "predictions/base_02_predictions.csv",
                              parse_dates=["data_origem", "data_prevista"])
        cls.metrics = pd.read_csv(RESULTS / "metrics/base_02_metrics.csv")

    def test_d02_processed_grid_preserves_target(self):
        """D02: o parquet de features não preenche nem comprime o alvo."""
        processed = pd.read_parquet(ROOT / "data/processed/base_02.parquet")
        self.assertEqual(len(processed), len(self.raw))
        self.assertEqual(processed.index.name, "date_time")
        pd.testing.assert_index_equal(processed.index, pd.DatetimeIndex(self.raw.date_time,
                                                                        name="date_time"))
        pd.testing.assert_series_equal(processed.traffic_volume, self.y)

    def test_e01_executed_notebook_without_errors(self):
        """E01/M02: a evidência executada corresponde ao código atual do notebook."""
        source_codes = [cell for cell in notebook_cells() if cell["cell_type"] == "code"]
        executed_path = RESULTS / "test_runs/base_02_execucao_teste.ipynb"
        self.assertTrue(executed_path.is_file(), "Execute o notebook antes de auditar as saídas.")
        executed = json.loads(executed_path.read_text(encoding="utf-8"))
        codes = [cell for cell in executed["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(["".join(c["source"]).strip() for c in source_codes],
                         ["".join(c["source"]).strip() for c in codes],
                         "A evidência de execução está desatualizada em relação ao código.")
        self.assertTrue(codes)
        self.assertTrue(all(cell.get("execution_count") is not None for cell in codes))
        errors = [(i, output.get("ename"), output.get("evalue"))
                  for i, cell in enumerate(codes) for output in cell.get("outputs", [])
                  if output.get("output_type") == "error"]
        self.assertFalse(errors, errors)
        verification = next("".join(c["source"]) for c in codes
                            if "batch, artifact = forecast_block('SARIMAX'" in "".join(c["source"]))
        self.assertIn("np.testing.assert_allclose(batch", verification)
        self.assertIn("np.testing.assert_allclose(hw_online", verification)

    def test_e01_required_artifacts_exist(self):
        """E01: os arquivos principais e os gráficos da Base 02 foram gerados."""
        expected = [
            ROOT / "data/processed/base_02.parquet",
            RESULTS / "predictions/base_02_predictions.csv",
            RESULTS / "metrics/base_02_metrics.csv",
            RESULTS / "metrics/base_02_baseline.csv",
            RESULTS / "residuals/base_02_residuals.csv",
            RESULTS / "residuals/base_02_ljung_box.csv",
            RESULTS / "feature_importance/base_02_feature_importance.csv",
            RESULTS / "feature_importance/base_02_sarimax_coefficients.csv",
            RESULTS / "tuning/base_02_search.csv",
            RESULTS / "tuning/base_02_fit_log.csv",
            RESULTS / "tuning/base_02_manifest.json",
            RESULTS / "analysis/base_02_analise.md",
        ]
        for path in expected:
            with self.subTest(arquivo=path.name):
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)
        self.assertTrue(list((RESULTS / "figures/base_02").glob("*.png")))

    def test_o01_same_hours_and_one_step_forecasts(self):
        """O01/M01: quatro modelos oficiais e, se presente, Decision Tree extra."""
        models = set(self.pred.modelo)
        self.assertTrue(OFFICIAL_MODELS <= models, OFFICIAL_MODELS - models)
        self.assertTrue(models <= OFFICIAL_MODELS | OPTIONAL_MODELS, models)
        self.assertFalse(self.pred.duplicated(["modelo", "data_prevista"]).any())
        self.assertEqual(set(self.pred.base), {"base_02"})
        for name, group in self.pred.groupby("modelo"):
            self.assertEqual(len(group), 4_820, name)
            pd.testing.assert_index_equal(pd.DatetimeIndex(group.data_prevista),
                                          self.expected_dates, check_names=False, obj=name)
            self.assertTrue(group.horizonte.eq(1).all(), name)
            self.assertTrue((group.data_prevista - group.data_origem).eq(HOUR).all(), name)
            self.assertTrue(np.isfinite(group.valor_previsto.to_numpy(dtype=float)).all(), name)
            self.assertTrue(group.valor_previsto.ge(0).all(), name)
            np.testing.assert_allclose(group.valor_real.to_numpy(dtype=float),
                                       self.y.iloc[self.test_start:].to_numpy(), equal_nan=True)

    def test_o02_residuals_and_recomputed_metrics(self):
        """O02: erros recalculados do CSV de previsões, sem imputar o alvo."""
        self.assertEqual(set(self.metrics.modelo), set(self.pred.modelo))
        actual_observed_dates = set(self.expected_dates[self.y.iloc[self.test_start:].notna()])
        for name, group in self.pred.groupby("modelo"):
            valid = group.valor_real.notna()
            self.assertEqual(int(valid.sum()), 4_805, name)
            self.assertEqual(set(group.loc[valid, "data_prevista"]), actual_observed_dates)
            self.assertTrue(group.alvo_observado.astype(str).str.lower().eq("true").eq(valid).all())
            np.testing.assert_allclose(group.loc[valid, "residuo"],
                                       group.loc[valid, "valor_real"] - group.loc[valid, "valor_previsto"],
                                       atol=1e-9)
            self.assertTrue(group.loc[~valid, "residuo"].isna().all(), name)
            err = group.loc[valid, "valor_real"].to_numpy() - group.loc[valid, "valor_previsto"].to_numpy()
            row = self.metrics.set_index("modelo").loc[name]
            self.assertAlmostEqual(float(row.mae), float(np.mean(np.abs(err))), delta=1e-6)
            self.assertAlmostEqual(float(row.rmse), float(np.sqrt(np.mean(err ** 2))), delta=1e-6)
            self.assertAlmostEqual(float(row.vies), float(np.mean(err)), delta=1e-6)
            self.assertEqual(int(row.n_previsoes), 4_820)
            self.assertEqual(int(row.n_avaliadas), 4_805)
        expected_ranking = self.metrics.mae.rank(method="min").astype(int)
        pd.testing.assert_series_equal(self.metrics.ranking, expected_ranking, check_names=False)
        self.assertTrue(self.metrics.mae.is_monotonic_increasing)
        official = self.metrics.modelo.isin(OFFICIAL_MODELS)
        self.assertTrue(self.metrics.loc[~official, "ranking_oficial"].isna().all())
        np.testing.assert_array_equal(
            self.metrics.loc[official, "ranking_oficial"].to_numpy(),
            self.metrics.loc[official, "mae"].rank(method="min").to_numpy())
        self.assertTrue(self.metrics.loc[official, "escopo"].eq("oficial").all())

    def test_o03_seasonal_baseline_on_common_dates(self):
        """O03: ingênuo de 168 h e modelos medidos no mesmo subconjunto."""
        baseline = pd.read_csv(RESULTS / "metrics/base_02_baseline.csv")
        naive = self.y.shift(168).iloc[self.test_start:]
        actual = self.y.iloc[self.test_start:]
        common = actual.notna() & naive.notna()
        expected_n = int(common.sum())
        self.assertEqual(set(baseline.modelo), set(self.pred.modelo) | {"Sazonal ingênuo (168 h)"})
        self.assertTrue(baseline.n.eq(expected_n).all())
        expected = float(np.mean(np.abs(actual[common].to_numpy() - naive[common].to_numpy())))
        recorded = float(baseline.set_index("modelo").loc["Sazonal ingênuo (168 h)", "mae_mesmas_datas"])
        self.assertAlmostEqual(recorded, expected, delta=1e-6)
        for name, group in self.pred.groupby("modelo"):
            pred = group.set_index("data_prevista").valor_previsto.reindex(actual.index)
            expected = float(np.mean(np.abs(actual[common].to_numpy() - pred[common].to_numpy())))
            recorded = float(baseline.set_index("modelo").loc[name, "mae_mesmas_datas"])
            self.assertAlmostEqual(recorded, expected, delta=1e-6)
        self.assertLess(baseline.loc[baseline.modelo.ne("Sazonal ingênuo (168 h)"),
                                     "mae_mesmas_datas"].min(),
                        float(baseline.set_index("modelo").loc["Sazonal ingênuo (168 h)",
                                                               "mae_mesmas_datas"]))

    def test_t02_fit_log_windows_and_weekly_test_refits(self):
        """T02/M03: treino cresce desde hora zero; reajustes continuam semanais."""
        logs = pd.read_csv(RESULTS / "tuning/base_02_fit_log.csv",
                           parse_dates=["treino_inicio", "treino_fim", "previsao_inicio", "previsao_fim"])
        self.assertTrue(set(self.pred.modelo) <= set(logs.modelo))
        self.assertTrue(logs.treino_inicio.le(logs.treino_fim).all())
        self.assertTrue(logs.treino_fim.lt(logs.previsao_inicio).all())
        self.assertTrue(logs.previsao_inicio.le(logs.previsao_fim).all())
        train_hours = (logs.treino_fim - logs.treino_inicio) / HOUR + 1
        self.assertTrue(logs.treino_inicio.eq(self.y.index[TRAIN_START_POS]).all())
        self.assertTrue(logs.janela_tipo.eq("expansiva").all())
        self.assertTrue((logs.previsao_inicio - logs.treino_fim).eq(HOUR).all())
        self.assertTrue(logs.n_treino_horas.eq(train_hours).all())
        for row in logs.itertuples():
            start = self.y.index.get_loc(row.previsao_inicio)
            self.assertEqual(int(row.n_treino_horas), start - TRAIN_START_POS)
            self.assertEqual(int(row.n_treino_observado),
                             int(self.y.iloc[TRAIN_START_POS:start].notna().sum()))
        self.assertTrue(logs.n_treino_observado.gt(0).all())
        non_test = logs.loc[logs.etapa.ne("teste")]
        self.assertTrue(non_test.previsao_fim.lt(self.expected_dates[0]).all())
        expected_starts = list(self.expected_dates[::REFIT_HOURS])
        for name in set(self.pred.modelo):
            test_logs = logs.loc[logs.modelo.eq(name) & logs.etapa.eq("teste")]
            self.assertEqual(list(test_logs.previsao_inicio), expected_starts, name)
            expected_ends = [self.expected_dates[min((i+1)*REFIT_HOURS, len(self.expected_dates))-1]
                             for i in range(len(expected_starts))]
            self.assertEqual(list(test_logs.previsao_fim), expected_ends, name)
            self.assertTrue(test_logs.n_treino_horas.is_monotonic_increasing, name)
            self.assertEqual(int(test_logs.n_treino_horas.iloc[0]), self.test_start, name)
        sar = logs.loc[logs.modelo.eq("SARIMAX") & logs.etapa.eq("teste")]
        self.assertTrue(sar.convergiu.notna().all())
        self.assertIn("avisos", logs)

    def test_e01_manifest_and_pretest_parameter_selection(self):
        """E01/T01: configuração e seleção permanecem rastreáveis e pré-teste."""
        manifest = json.loads((RESULTS / "tuning/base_02_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sha256_entrada"], FROZEN_SHA256)
        self.assertEqual(manifest["random_state"], 67)
        self.assertEqual(manifest["horizonte"], 1)
        self.assertEqual(manifest["frequencia"], "h")
        self.assertEqual(manifest["treino_proporcao_grade"], TRAIN_RATIO)
        self.assertEqual(manifest["janela_tipo"], "expansiva")
        self.assertIsNone(manifest["janela_ajuste_horas"])
        self.assertEqual(manifest["treino_inicio_posicao"], TRAIN_START_POS)
        self.assertIn("janela expansiva", manifest["protocolo"])
        self.assertIn("one-step", manifest["protocolo"])
        self.assertEqual(manifest["refit_horas"], REFIT_HOURS)
        self.assertEqual(pd.Timestamp(manifest["teste_inicio"]), self.expected_dates[0])
        self.assertEqual(pd.Timestamp(manifest["teste_fim"]), self.expected_dates[-1])
        self.assertEqual(set(manifest["parametros"]), set(self.pred.modelo))
        self.assertEqual(len(manifest["validacao_folds"]), N_FOLDS)
        for start, stop in manifest["validacao_folds"]:
            self.assertEqual(stop - start, VALIDATION_HOURS)
            self.assertLessEqual(stop, self.test_start)
        self.assertNotIn("traffic_volume", manifest["feature_cols"])
        self.assertTrue(set(manifest["sarimax_exog"]) <= set(manifest["feature_cols"]))
        for package in ("python", "pandas", "sklearn", "statsmodels"):
            self.assertTrue(manifest["versoes"].get(package))
        search = pd.read_csv(RESULTS / "tuning/base_02_search.csv")
        for name, params in manifest["parametros"].items():
            candidates = search.loc[search.modelo.eq(name) & search.status.eq("ok")]
            self.assertFalse(candidates.empty, name)
            winner = candidates.loc[candidates.mae_validacao.idxmin()]
            self.assertEqual(json.loads(winner.parametros), params)
            recorded_validation = float(self.metrics.set_index("modelo").loc[name, "mae_validacao"])
            self.assertAlmostEqual(recorded_validation, float(winner.mae_validacao), delta=1e-6)

    def test_a01_residual_export_and_contiguous_ljung_box(self):
        """A01: diagnóstico usa apenas observações em um trecho contínuo."""
        residuals = pd.read_csv(RESULTS / "residuals/base_02_residuals.csv",
                                parse_dates=["data_prevista"])
        self.assertEqual(len(residuals), len(self.pred))
        joined = residuals.merge(self.pred, on=["base", "modelo", "data_prevista"],
                                 suffixes=("_res", "_pred"), validate="one_to_one")
        self.assertEqual(len(joined), len(self.pred))
        np.testing.assert_allclose(joined.residuo_res, joined.residuo_pred, equal_nan=True)
        lb = pd.read_csv(RESULTS / "residuals/base_02_ljung_box.csv",
                         parse_dates=["inicio", "fim"])
        self.assertEqual(set(lb.modelo), set(self.pred.modelo))
        self.assertTrue(lb.p_valor.between(0, 1).all())
        for row in lb.itertuples():
            segment = self.y.loc[row.inicio:row.fim]
            self.assertEqual(len(segment), row.n_trecho)
            self.assertTrue(segment.notna().all())
            self.assertEqual((row.fim - row.inicio) / HOUR + 1, row.n_trecho)
            self.assertLess(row.lag, row.n_trecho / 2)
        analysis = (RESULTS / "analysis/base_02_analise.md").read_text(encoding="utf-8").lower()
        self.assertIn("causalidade", analysis)
        self.assertIn("limita", analysis)


if __name__ == "__main__":
    unittest.main()
