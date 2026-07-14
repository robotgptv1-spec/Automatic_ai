"""
AutoAI ML Engine
----------------
Fixed + generalized version of the original SimpleClassifier /
SimpleSoftmaxClassifier / ChatGptNetwork prototype.

Bugs fixed from the original notebook code:
  1. SimpleSoftmaxClassifier was being used for BINARY classification
     with BCELoss + `(outputs > 0.5)` thresholding, but Softmax output
     sums to 1 across classes and was instantiated with only
     `input_size` (missing num_classes) -> would crash.
  2. Sigmoid/Softmax were baked into the network and combined with
     BCELoss/CrossEntropyLoss, which already expect raw logits.
     Applying an activation before those losses is mathematically
     wrong (and numerically unstable). Logits are now returned by the
     network; activations are applied only at inference time.
  3. `input()` calls for column selection can't run in a web backend -
     replaced with structured JSON config coming from the UI.
  4. Encoded target (`y_train_labeled`) was computed but never used -
     the raw unencoded y was fed to the tensors instead.
  5. No support for regression, no feature scaling, no categorical
     encoding, no handling of unseen categories at predict time -
     all added below.
"""

import io
import json
import pickle
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

# Keep torch's internal thread pool small. Letting it grab all CPU cores
# inside a Flask dev server worker (especially on Windows) is a common
# cause of the whole process hanging or dying mid-request under load,
# which the browser reports as an opaque "Failed to fetch".
torch.set_num_threads(1)


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------

class BinaryClassifierNet(nn.Module):
    """Outputs a single raw logit. Use with BCEWithLogitsLoss."""

    def __init__(self, input_size, hidden=(64, 32)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.network(x)


class MultiClassClassifierNet(nn.Module):
    """Outputs raw logits (num_classes,). Use with CrossEntropyLoss."""

    def __init__(self, input_size, num_classes, hidden=(64, 32)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], num_classes),
        )

    def forward(self, x):
        return self.network(x)


class RegressionNet(nn.Module):
    def __init__(self, input_size, hidden=(64, 32)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------------------------------
# Session (one per uploaded dataset / trained model)
# --------------------------------------------------------------------------

class AutoMLSession:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.df = None
        self.filename = None

        # config
        self.feature_columns = []
        self.target_column = None
        self.task_type = None          # 'classification' | 'regression'
        self.problem_mode = None       # 'binary' | 'multiclass' | 'regression'

        # fitted preprocessing artifacts
        self.numeric_features = []
        self.categorical_features = []
        self.cat_encoders = {}         # col -> LabelEncoder
        self.scaler = None             # StandardScaler over final numeric feature matrix
        self.target_encoder = None     # LabelEncoder for classification targets
        self.class_names = None

        # model
        self.model = None
        self.input_size = None
        self.num_classes = None
        self.train_log = []
        self.final_metrics = {}

    # ---------------- dataset ----------------

    def load_csv(self, file_storage):
        self.filename = file_storage.filename
        self.df = pd.read_csv(file_storage)
        self.df.columns = [str(c).strip() for c in self.df.columns]
        return self.summary()

    def summary(self):
        df = self.df
        cols = []
        for c in df.columns:
            dtype = str(df[c].dtype)
            nunique = int(df[c].nunique(dropna=True))
            is_numeric = pd.api.types.is_numeric_dtype(df[c])
            cols.append({
                "name": c,
                "dtype": dtype,
                "nunique": nunique,
                "is_numeric": bool(is_numeric),
                "null_count": int(df[c].isna().sum()),
                "sample_values": [self._jsonable(v) for v in df[c].dropna().unique()[:5]],
            })
        preview = json.loads(df.head(8).to_json(orient="records"))
        return {
            "session_id": self.id,
            "filename": self.filename,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": cols,
            "preview": preview,
        }

    @staticmethod
    def _jsonable(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    # ---------------- detection ----------------

    def _detect_task_type(self, y: pd.Series):
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        nunique = y.nunique(dropna=True)
        looks_categorical = nunique <= 15 and (y.dropna() % 1 == 0).all()
        return "classification" if looks_categorical else "regression"

    # ---------------- preprocessing ----------------

    def _build_feature_matrix(self, df_slice, fit: bool):
        """Encode categoricals, return a float32 numpy matrix. If fit=True,
        (re)fits encoders/scaler; otherwise reuses fitted ones (predict time)."""
        cols = []
        for col in self.feature_columns:
            series = df_slice[col]
            if pd.api.types.is_numeric_dtype(series) and col not in self.categorical_features:
                cols.append(series.astype(float).fillna(series.astype(float).mean() if fit else 0).values.reshape(-1, 1))
            else:
                if fit:
                    enc = LabelEncoder()
                    filled = series.astype(str).fillna("__missing__")
                    encoded = enc.fit_transform(filled)
                    self.cat_encoders[col] = enc
                else:
                    enc = self.cat_encoders[col]
                    filled = series.astype(str).fillna("__missing__")
                    # map unseen categories to a safe fallback (0) instead of crashing
                    known = set(enc.classes_)
                    mapped = filled.apply(lambda v: v if v in known else enc.classes_[0])
                    encoded = enc.transform(mapped)
                cols.append(np.asarray(encoded).astype(float).reshape(-1, 1))
        X = np.hstack(cols).astype(np.float32)
        return X

    def configure(self, feature_columns, target_column, task_type="auto",
                  test_size=0.2, epochs=30, lr=0.001, batch_size=32):
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        missing = [c for c in feature_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        if target_column in feature_columns:
            raise ValueError("Target column cannot also be a feature column.")
        if not feature_columns:
            raise ValueError("Select at least one feature column.")

        self.feature_columns = feature_columns
        self.target_column = target_column
        self.categorical_features = [
            c for c in feature_columns if not pd.api.types.is_numeric_dtype(self.df[c])
        ]
        self.numeric_features = [c for c in feature_columns if c not in self.categorical_features]

        y_raw = self.df[target_column]
        self.task_type = self._detect_task_type(y_raw) if task_type == "auto" else task_type

        return {
            "task_type": self.task_type,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
        }

    # ---------------- training ----------------

    def train(self, test_size=0.2, epochs=30, lr=0.001, batch_size=32):
        df = self.df.dropna(subset=[self.target_column])
        X = self._build_feature_matrix(df, fit=True)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X).astype(np.float32)

        y_raw = df[self.target_column]

        if self.task_type == "classification":
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y_raw.astype(str))
            self.class_names = [str(c) for c in self.target_encoder.classes_]
            self.num_classes = len(self.class_names)
            self.problem_mode = "binary" if self.num_classes == 2 else "multiclass"
        else:
            y = y_raw.astype(float).values
            self.problem_mode = "regression"
            self.class_names = None

        stratify = y if self.task_type == "classification" and self.num_classes > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        self.input_size = X.shape[1]

        if self.problem_mode == "binary":
            self.model = BinaryClassifierNet(self.input_size)
            criterion = nn.BCEWithLogitsLoss()
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        elif self.problem_mode == "multiclass":
            self.model = MultiClassClassifierNet(self.input_size, self.num_classes)
            criterion = nn.CrossEntropyLoss()
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            y_test_t = torch.tensor(y_test, dtype=torch.long)
        else:  # regression
            self.model = RegressionNet(self.input_size)
            criterion = nn.MSELoss()
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_log = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            train_loss = running_loss / max(n_batches, 1)

            self.model.eval()
            with torch.no_grad():
                test_out = self.model(X_test_t)
                test_loss = criterion(test_out, y_test_t).item()
                metric_name, metric_value = self._eval_metric(test_out, y_test_t)

            self.train_log.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "test_loss": round(test_loss, 5),
                "metric_name": metric_name,
                "metric_value": round(metric_value, 4),
            })

        self.final_metrics = self.train_log[-1] if self.train_log else {}
        return {"log": self.train_log, "final": self.final_metrics, "problem_mode": self.problem_mode}

    def _eval_metric(self, out, y_true_t):
        if self.problem_mode == "binary":
            probs = torch.sigmoid(out).numpy().ravel()
            preds = (probs > 0.5).astype(int)
            acc = accuracy_score(y_true_t.numpy().ravel(), preds)
            return "accuracy", float(acc)
        elif self.problem_mode == "multiclass":
            preds = torch.argmax(out, dim=1).numpy()
            acc = accuracy_score(y_true_t.numpy(), preds)
            return "accuracy", float(acc)
        else:
            preds = out.numpy().ravel()
            r2 = r2_score(y_true_t.numpy().ravel(), preds)
            return "r2_score", float(r2)

    # ---------------- prediction ----------------

    def predict_one(self, feature_dict):
        row = {col: feature_dict.get(col) for col in self.feature_columns}
        df_row = pd.DataFrame([row])
        for col in self.numeric_features:
            df_row[col] = pd.to_numeric(df_row[col], errors="coerce")
        X = self._build_feature_matrix(df_row, fit=False)
        X = self.scaler.transform(X).astype(np.float32)
        X_t = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            out = self.model(X_t)

        if self.problem_mode == "binary":
            prob = torch.sigmoid(out).item()
            pred_idx = int(prob > 0.5)
            label = self.target_encoder.inverse_transform([pred_idx])[0]
            return {
                "prediction": str(label),
                "confidence": round(prob if pred_idx == 1 else 1 - prob, 4),
                "probabilities": {
                    self.class_names[0]: round(1 - prob, 4),
                    self.class_names[1]: round(prob, 4),
                },
            }
        elif self.problem_mode == "multiclass":
            probs = torch.softmax(out, dim=1).numpy().ravel()
            pred_idx = int(np.argmax(probs))
            label = self.target_encoder.inverse_transform([pred_idx])[0]
            return {
                "prediction": str(label),
                "confidence": round(float(probs[pred_idx]), 4),
                "probabilities": {self.class_names[i]: round(float(p), 4) for i, p in enumerate(probs)},
            }
        else:
            value = out.item()
            return {"prediction": round(float(value), 4)}

    # ---------------- persistence ----------------

    def to_bytes(self):
        buf = io.BytesIO()
        pickle.dump({
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "problem_mode": self.problem_mode,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "cat_encoders": self.cat_encoders,
            "scaler": self.scaler,
            "target_encoder": self.target_encoder,
            "class_names": self.class_names,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "model_state_dict": self.model.state_dict(),
            "model_class": type(self.model).__name__,
        }, buf)
        buf.seek(0)
        return buf