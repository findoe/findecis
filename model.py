import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data_processed.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

FEATURES = [f"x{i}" for i in range(1, 36)]
REG_TARGETS = ["reg", "kredit", "teh", "market", "staff", "psich", "ability", "turn", "finn", "Z25", "Z35"]
CLS_TARGET = "Bankrupt"
META_COLS = ["Отрасль экономики", "ИНН", "Год"]
GROUP_COL = "ИНН"
YEAR_COL = "Год"
SEED = 42


def split_data(df, mode):
    if mode == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_val_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COL]))
        train_val, test = df.iloc[train_val_idx].copy(), df.iloc[test_idx].copy()

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
        train_idx, val_idx = next(splitter.split(train_val, groups=train_val[GROUP_COL]))
        return train_val.iloc[train_idx].copy(), train_val.iloc[val_idx].copy(), test

    if mode == "time":
        years = sorted(df[YEAR_COL].unique())
        if len(years) < 3:
            raise ValueError("Для split-mode=time нужно минимум 3 разных года")
        return (
            df[df[YEAR_COL] < years[-2]].copy(),
            df[df[YEAR_COL] == years[-2]].copy(),
            df[df[YEAR_COL] == years[-1]].copy(),
        )

    if mode == "random":
        train_val, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[CLS_TARGET])
        train, val = train_test_split(train_val, test_size=0.2, random_state=SEED + 1, stratify=train_val[CLS_TARGET])
        return train.copy(), val.copy(), test.copy()

    raise ValueError("split-mode должен быть: group, time или random")


def make_xy(df):
    return (
        df[FEATURES].astype("float32").values,
        df[REG_TARGETS].astype("float32").values,
        df[CLS_TARGET].astype("float32").values.reshape(-1, 1),
    )


def build_model():
    inputs = Input(shape=(len(FEATURES),))

    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    reg_output = Dense(len(REG_TARGETS), activation="sigmoid", name="regression_output")(x)
    cls_output = Dense(1, activation="sigmoid", name="classification_output")(x)

    model = Model(inputs, [reg_output, cls_output])
    model.compile(
        optimizer=Adam(1e-3),
        loss={"regression_output": "mse", "classification_output": "binary_crossentropy"},
        loss_weights={"regression_output": 0.35, "classification_output": 0.65},
        metrics={"classification_output": ["accuracy"]},
    )
    return model


def target_distribution(df):
    return {str(k): int(v) for k, v in df[CLS_TARGET].value_counts().sort_index().items()}


def evaluate(y_cls_test, y_reg_test, pred_cls, pred_reg, threshold):
    y_true = y_cls_test.ravel().astype(int)
    y_pred = (pred_cls.ravel() >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "classification": {
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": cm.tolist(),
        },
        "regression": {
            "mae_macro": float(mean_absolute_error(y_reg_test, pred_reg)),
            "rmse_macro": float(np.sqrt(mean_squared_error(y_reg_test, pred_reg))),
        },
    }


def make_predictions(test_df, pred_cls, pred_reg, threshold):
    result = test_df[META_COLS + [CLS_TARGET]].copy().reset_index(drop=True)
    result["bankruptcy_probability"] = pred_cls.ravel()
    result["bankruptcy_predicted"] = (result["bankruptcy_probability"] >= threshold).astype(int)

    for i, col in enumerate(REG_TARGETS):
        result[f"pred_{col}"] = pred_reg[:, i]
        result[f"true_{col}"] = test_df[col].values

    return result


def main(args):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    artifacts_dir = Path(args.artifacts_dir)
    metrics_dir = artifacts_dir / "metrics"
    predictions_dir = artifacts_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, sep=";")
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    required = META_COLS + FEATURES + REG_TARGETS + [CLS_TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"В датасете нет столбцов: {missing}")

    train_df, val_df, test_df = split_data(df, args.split_mode)
    x_train, y_reg_train, y_cls_train = make_xy(train_df)
    x_val, y_reg_val, y_cls_val = make_xy(val_df)
    x_test, y_reg_test, y_cls_test = make_xy(test_df)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    model = build_model()
    model.fit(
        x_train,
        {"regression_output": y_reg_train, "classification_output": y_cls_train},
        validation_data=(x_val, {"regression_output": y_reg_val, "classification_output": y_cls_val}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)],
        verbose=1,
    )

    pred_reg, pred_cls = model.predict(x_test, verbose=0)
    metrics = evaluate(y_cls_test, y_reg_test, pred_cls, pred_reg, args.threshold)
    predictions = make_predictions(test_df, pred_cls, pred_reg, args.threshold)

    model.save(artifacts_dir / "financial_agent_model.keras")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    predictions.to_csv(predictions_dir / "test_predictions.csv", index=False, sep=";")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(args.data),
        "split_mode": args.split_mode,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_validation": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "feature_columns": FEATURES,
        "regression_columns": REG_TARGETS,
        "classification_column": CLS_TARGET,
        "target_distribution_total": target_distribution(df),
        "target_distribution_train": target_distribution(train_df),
        "target_distribution_validation": target_distribution(val_df),
        "target_distribution_test": target_distribution(test_df),
    }

    with open(metrics_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = np.array(metrics["classification"]["confusion_matrix"])
    tn, fp, fn, tp = cm.ravel()

    print("\nКлассификация:")
    print(f"  Accuracy:  {metrics['classification']['accuracy']:.3f}")
    print(f"  Precision: {metrics['classification']['precision']:.3f}")
    print(f"  Recall:    {metrics['classification']['recall']:.3f}")
    print(f"  F1:        {metrics['classification']['f1']:.3f}")
    print("\nConfusion matrix:")
    print(f"  {tn} — правильно найденные небанкроты")
    print(f"  {fp} — ложные тревоги")
    print(f"  {fn} — пропущенные банкроты")
    print(f"  {tp} — правильно найденные банкроты")
    print("\nРегрессия:")
    print(f"  MAE macro:  {metrics['regression']['mae_macro']:.3f}")
    print(f"  RMSE macro: {metrics['regression']['rmse_macro']:.3f}")
    print(f"\nАртефакты сохранены в: {artifacts_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--split-mode", choices=["group", "time", "random"], default="group")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
