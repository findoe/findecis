import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
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
SEED = 42


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
        loss={
            "regression_output": "mse",
            "classification_output": "binary_crossentropy",
        },
        loss_weights={
            "regression_output": 0.35,
            "classification_output": 0.65,
        },
        metrics={
            "classification_output": ["accuracy"],
        },
    )
    return model


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    x = df[FEATURES].astype("float32").values
    y_reg = df[REG_TARGETS].astype("float32").values
    y_cls = df[CLS_TARGET].astype("int32").values.reshape(-1, 1)

    x_train, x_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        x,
        y_reg,
        y_cls,
        test_size=0.2,
        random_state=SEED,
        stratify=y_cls,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = build_model()
    model.fit(
        x_train,
        {
            "regression_output": y_reg_train,
            "classification_output": y_cls_train,
        },
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
        verbose=1,
    )

    pred_reg, pred_cls = model.predict(x_test, verbose=0)
    y_true = y_cls_test.ravel()
    y_pred = (pred_cls.ravel() >= 0.5).astype(int)

    print("\nКлассификация:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("\nConfusion matrix:")
    print(f"  {tn} — правильно найденные небанкроты")
    print(f"  {fp} — ложные тревоги")
    print(f"  {fn} — пропущенные банкроты")
    print(f"  {tp} — правильно найденные банкроты")

    mae = mean_absolute_error(y_reg_test, pred_reg)
    rmse = np.sqrt(mean_squared_error(y_reg_test, pred_reg))

    print("\nРегрессия:")
    print(f"  MAE macro:  {mae:.3f}")
    print(f"  RMSE macro: {rmse:.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS_DIR / "financial_agent_model.keras")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")


if __name__ == "__main__":
    main()
