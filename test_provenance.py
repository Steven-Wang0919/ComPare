import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import common_utils
import evaluate_inverse_opening_holdout
import run_utils
import train_mlp


class ProvenanceTests(unittest.TestCase):
    def test_load_data_with_metadata_assigns_sample_tracking(self):
        df = pd.DataFrame({
            "排肥口开度（mm）": [20.0, 35.0],
            "排肥轴转速（r/min）": [25.0, 30.0],
            "实际排肥质量（g/min）": [100.0, 200.0],
        })

        with mock.patch.object(common_utils.pd, "read_excel", return_value=df):
            X, y, sample_meta = common_utils.load_data_with_metadata("ignored.xlsx")

        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2,))
        self.assertEqual(sample_meta["sample_id"].tolist(), [0, 1])
        self.assertEqual(sample_meta["source_row_number"].tolist(), [2, 3])

    def test_write_manifest_records_command_and_split_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "runs", "20260322T000000_dummy")
            os.makedirs(run_dir, exist_ok=True)

            data_path = os.path.join(tmpdir, "dataset.xlsx")
            script_path = os.path.join(tmpdir, "dummy.py")
            with open(data_path, "wb") as f:
                f.write(b"dataset")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write("print('dummy')\n")

            split_payload = run_utils.build_single_split_artifact_payload(
                [0, 1],
                [2],
                [3],
                n_samples=4,
            )
            manifest_path = run_utils.write_manifest(
                run_dir,
                script_name=script_path,
                data_path=data_path,
                seed=7,
                params={"demo": True},
                source_files=[script_path],
                split_payload=split_payload,
            )

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            split_summary = manifest["artifacts"]["split_indices"]
            split_path = os.path.join(run_dir, split_summary["path"])
            self.assertTrue(os.path.exists(split_path))
            self.assertEqual(manifest["command"]["python_executable"], os.path.abspath(os.sys.executable).replace("\\", "/"))
            self.assertEqual(split_summary["kind"], "single_split")
            self.assertEqual(split_summary["fold_count"], 1)
            self.assertEqual(split_summary["n_samples"], 4)
            self.assertEqual(split_summary["sha256"], run_utils.sha256_of_file(split_path))
            self.assertIn("sample_id_definition", manifest["sample_tracking"])

            with open(split_path, "r", encoding="utf-8") as f:
                split_payload_on_disk = json.load(f)
            self.assertEqual(split_payload_on_disk["schema_version"], 1)
            self.assertEqual(split_payload_on_disk["kind"], "single_split")

    def test_train_and_eval_mlp_writes_sample_tracking_columns(self):
        X = np.array(
            [
                [20.0, 20.0],
                [20.0, 25.0],
                [35.0, 30.0],
                [35.0, 35.0],
                [50.0, 40.0],
                [50.0, 45.0],
            ],
            dtype=np.float32,
        )
        y = np.array([100.0, 120.0, 140.0, 160.0, 180.0, 200.0], dtype=np.float32)
        sample_meta = pd.DataFrame({
            "sample_id": np.arange(len(X), dtype=int),
            "source_row_number": np.arange(len(X), dtype=int) + 2,
        })

        def fake_fit(X_train_raw, y_train_raw, X_eval_raw, **kwargs):
            del kwargs
            mean_value = float(np.mean(y_train_raw))
            return {
                "model": "dummy-mlp",
                "y_pred_eval": np.full(len(X_eval_raw), mean_value, dtype=float),
                "norm_stats": {
                    "X_min": np.min(X_train_raw, axis=0, keepdims=True),
                    "X_max": np.max(X_train_raw, axis=0, keepdims=True),
                    "y_min": float(np.min(y_train_raw)),
                    "y_max": float(np.max(y_train_raw)),
                },
            }

        fake_tuning_result = {
            "best_config": {"hidden_layer_sizes": (2,), "alpha": 1e-4},
            "best_candidate_idx": 0,
            "candidate_summaries": [{"mean_val_r2": 0.5}],
            "inner_fold_count": 1,
            "inner_split_strategy": "repeated_random",
            "inner_split_meta": {},
            "tuning_records": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results_mlp.csv")
            with mock.patch.object(train_mlp, "load_data_with_metadata", return_value=(X, y, sample_meta)):
                with mock.patch.object(train_mlp, "prepare_inner_cv", return_value=([], "repeated_random", {})):
                    with mock.patch.object(train_mlp, "run_fair_tuning", return_value=fake_tuning_result):
                        with mock.patch.object(train_mlp, "_fit_predict_forward_mlp", side_effect=fake_fit):
                            result = train_mlp.train_and_eval_mlp(
                                data_path="ignored.xlsx",
                                hidden_layer_candidates=[(2,)],
                                alpha_candidates=[1e-4],
                                save_csv_path=csv_path,
                                split_indices=(np.array([0, 1, 2]), np.array([3]), np.array([4, 5])),
                            )

            df = pd.read_csv(csv_path)
            self.assertEqual(df["sample_id"].tolist(), [4, 5])
            self.assertEqual(df["source_row_number"].tolist(), [6, 7])
            self.assertEqual(result["split_indices_payload"]["kind"], "single_split")
            self.assertEqual(result["split_indices_payload"]["fold_count"], 1)

    def test_inverse_opening_holdout_outputs_sample_tracking_columns(self):
        X_raw = np.array(
            [
                [20.0, 21.0],
                [20.0, 22.0],
                [35.0, 31.0],
                [35.0, 32.0],
            ],
            dtype=np.float32,
        )
        y_raw = np.array([100.0, 110.0, 200.0, 210.0], dtype=np.float32)
        sample_meta = pd.DataFrame({
            "sample_id": np.arange(len(X_raw), dtype=int),
            "source_row_number": np.arange(len(X_raw), dtype=int) + 2,
        })
        folds = [
            {
                "fold_id": 1,
                "test_opening_mm": 20.0,
                "train_val_openings_mm": [35.0],
                "train_val_openings_label": "35",
                "train_size": 1,
                "val_size": 1,
                "test_size": 2,
                "split_info": {
                    "protocol": "leave_one_opening_out",
                    "idx_train": np.array([2]),
                    "idx_val": np.array([3]),
                    "idx_test": np.array([0, 1]),
                },
            },
            {
                "fold_id": 2,
                "test_opening_mm": 35.0,
                "train_val_openings_mm": [20.0],
                "train_val_openings_label": "20",
                "train_size": 1,
                "val_size": 1,
                "test_size": 2,
                "split_info": {
                    "protocol": "leave_one_opening_out",
                    "idx_train": np.array([0]),
                    "idx_val": np.array([1]),
                    "idx_test": np.array([2, 3]),
                },
            },
        ]

        def make_stub(model_name, pred_offset):
            def _stub(*, split_indices=None, **kwargs):
                del kwargs
                idx_test = np.asarray(split_indices[2], dtype=int)
                y_true = X_raw[idx_test, 1].astype(float)
                mass = y_raw[idx_test].astype(float)
                opening = X_raw[idx_test, 0].astype(float)
                pred = y_true + pred_offset
                return {
                    "r2_main": 0.9,
                    "are_main": 1.0,
                    "n_main": len(idx_test),
                    "main_ratio": 1.0,
                    "r2_all": 0.9,
                    "are_all": 1.0,
                    "n_all": len(idx_test),
                    "best_hidden": (8,),
                    "best_alpha": 1e-4,
                    "best_sigma": 0.5,
                    "best_hidden_dim": 16,
                    "best_lr": 1e-3,
                    "best_weight_decay": 1e-5,
                    "y_true_all": y_true,
                    "y_pred_all": pred,
                    "mass_all": mass,
                    "opening_all": opening,
                    "strategy_opening_all": opening.copy(),
                    "y_true_main": y_true,
                    "y_pred_main": pred,
                    "mass_main": mass,
                    "opening_main": opening,
                    "strategy_opening_main": opening.copy(),
                    "policy_mask": np.ones(len(idx_test), dtype=bool),
                    "tuning_records": [],
                    "artifact_model_path": None,
                    "artifact_meta_path": None,
                    "artifact_test_inputs_path": None,
                    "artifact_test_targets_path": None,
                }

            return _stub

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                evaluate_inverse_opening_holdout,
                "train_and_eval_inverse_mlp",
                side_effect=make_stub("inverse_MLP", 0.1),
            ):
                with mock.patch.object(
                    evaluate_inverse_opening_holdout,
                    "train_and_eval_inverse_grnn",
                    side_effect=make_stub("inverse_GRNN", 0.2),
                ):
                    with mock.patch.object(
                        evaluate_inverse_opening_holdout,
                        "train_and_eval_inverse_kan_v2",
                        side_effect=make_stub("inverse_KAN", 0.3),
                    ):
                        outputs = evaluate_inverse_opening_holdout.run_inverse_opening_holdout_compare(
                            tmpdir,
                            data_path="ignored.xlsx",
                            seed=42,
                            X_raw=X_raw,
                            y_raw=y_raw,
                            sample_meta=sample_meta,
                            folds=folds,
                        )

            df_all = pd.read_csv(outputs["all_path"])
            self.assertEqual(df_all["sample_id"].tolist(), [0, 1, 2, 3])
            self.assertEqual(df_all["sample_index"].tolist(), [0, 1, 2, 3])
            self.assertEqual(df_all["source_row_number"].tolist(), [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
