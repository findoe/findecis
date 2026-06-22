from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# =========================
# =       МОДЕЛЬ          =
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "data_processed.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


#Фиксированное значение для повторяемого обучения
RANDOM_STATE = 25

#Входные признаки модели
FEATURE_COLUMNS: List[str] = [f"x{i}" for i in range(1, 36)]
#Выходные регрессионные показатели
REGRESSION_COLUMNS: List[str] = [
    "reg", "kredit", "teh", "market", "staff", "psich",
    "ability", "turn", "finn", "Z25", "Z35",
]

#Целевая переменная бинарной классификации
CLASSIFICATION_COLUMN = "Bankrupt"

#Служебные столбцы для группировки и описания наблюдений
GROUP_COLUMN = "ИНН"
YEAR_COLUMN = "Год"
META_COLUMNS = ["Отрасль экономики", "ИНН", "Год"]

#Воспроизводимость
def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


#Загрузка и проверка данных
def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")

    df = pd.read_csv(data_path, sep=";")
    df.columns = [str(col).strip() for col in df.columns]

    #В файле может быть служебный индексный столбец - убираем
    unnamed_columns = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    required_columns = META_COLUMNS + FEATURE_COLUMNS + REGRESSION_COLUMNS + [CLASSIFICATION_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют необходимые столбцы: {missing_columns}")

    numeric_columns = FEATURE_COLUMNS + REGRESSION_COLUMNS + [CLASSIFICATION_COLUMN, YEAR_COLUMN]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_columns].isna().any().any():
        bad = df[numeric_columns].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"В числовых столбцах есть пропуски или некорректные значения: {bad}")

    df[GROUP_COLUMN] = df[GROUP_COLUMN].astype(str)
    df[CLASSIFICATION_COLUMN] = df[CLASSIFICATION_COLUMN].astype(int)

    unique_targets = set(df[CLASSIFICATION_COLUMN].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Столбец {CLASSIFICATION_COLUMN} должен содержать только 0 и 1")

    return df



# Разделение выборки
def split_by_groups(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Основной вариант разбиения
    Один ИНН не должен попасть одновременно в обучение и тест, иначе возникает утечка данных
    """
    y = df[CLASSIFICATION_COLUMN].values
    groups = df[GROUP_COLUMN].values

    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(sgkf.split(df, y, groups))
    except Exception:
        #Резервный вариант, если версия sklearn не поддерживает нужное поведение.
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(gss.split(df, y, groups))

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    y_train_val = train_val[CLASSIFICATION_COLUMN].values
    groups_train_val = train_val[GROUP_COLUMN].values

    try:
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
        train_idx_rel, val_idx_rel = next(sgkf_val.split(train_val, y_train_val, groups_train_val))
        train = train_val.iloc[train_idx_rel].copy()
        val = train_val.iloc[val_idx_rel].copy()
    except Exception:
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
        train_idx_rel, val_idx_rel = next(gss_val.split(train_val, y_train_val, groups_train_val))
        train = train_val.iloc[train_idx_rel].copy()
        val = train_val.iloc[val_idx_rel].copy()

    return train, val, test



#Временное разбиение выборки
def split_by_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Альтернативный вариант для прогностической постановки:
    обучение на ранних годах, тестирование на самом позднем году
    """
    years = sorted(df[YEAR_COLUMN].unique())
    if len(years) < 3:
        raise ValueError("Для временного разбиения нужно минимум 3 разных года наблюдений!")

    test_year = years[-1]
    val_year = years[-2]

    train = df[df[YEAR_COLUMN] < val_year].copy()
    val = df[df[YEAR_COLUMN] == val_year].copy()
    test = df[df[YEAR_COLUMN] == test_year].copy()

    return train, val, test



#Случайное разбиение выборки
def split_random(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[CLASSIFICATION_COLUMN],
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=seed + 1,
        stratify=train_val[CLASSIFICATION_COLUMN],
    )
    return train.copy(), val.copy(), test.copy()



#Выбор режима разбиения данных
def make_split(df: pd.DataFrame, split_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_mode == "group":
        return split_by_groups(df)
    if split_mode == "time":
        return split_by_time(df)
    if split_mode == "random":
        return split_random(df)
    raise ValueError(f"Неизвестный режим разбиения: {split_mode}")



#Подготовка матриц
def make_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_reg = df[REGRESSION_COLUMNS].to_numpy(dtype=np.float32)
    y_cls = df[CLASSIFICATION_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)
    return x, y_reg, y_cls



#Масштабирование входных признаков
def scale_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


# =========================
# АРХИТЕКУТРА МОДЕЛИ
# =========================

def build_multitask_model(input_dim: int) -> Model:
    inputs = Input(shape=(input_dim,), name="financial_features")

    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4), name="dense_128")(inputs)
    x = BatchNormalization(name="bn_128")(x)
    x = Dropout(0.30, name="dropout_128")(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4), name="dense_64")(x)
    x = BatchNormalization(name="bn_64")(x)
    x = Dropout(0.25, name="dropout_64")(x)

    x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4), name="dense_32")(x)
    x = BatchNormalization(name="bn_32")(x)

    #Отдельная ветка регрессии: классификационный выход ниже не меняется.
    #Так регрессия получает больше емкости и меньше конкурирует с классификационной задачей.
    regression_branch = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(5e-5),
        name="regression_dense_64",
    )(x)
    regression_branch = BatchNormalization(name="regression_bn_64")(regression_branch)
    regression_branch = Dropout(0.10, name="regression_dropout_64")(regression_branch)

    regression_branch = Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(5e-5),
        name="regression_dense_32",
    )(regression_branch)
    regression_branch = BatchNormalization(name="regression_bn_32")(regression_branch)

    #Регрессионные цели лежат в диапазоне [0; 1], поэтому sigmoid ограничивает выход модели.
    regression_output = Dense(
        len(REGRESSION_COLUMNS),
        activation="sigmoid",
        name="regression_output",
    )(regression_branch)

    classification_output = Dense(
        1,
        activation="sigmoid",
        name="classification_output",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=[regression_output, classification_output],
        name="financial_state_multitask_mlp",
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            #Оптимизируем именно MAE, потому что этот показатель выводится как основная ошибка регрессии.
            "regression_output": "mae",
            "classification_output": "binary_crossentropy",
        },
        loss_weights={
            #Усиливаем только регрессионную задачу; классификационный вес оставлен как в исходной модели.
            "regression_output": 0.60,
            "classification_output": 0.65,
        },
        metrics={
            "regression_output": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "classification_output": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        },
    )
    return model



#Оценка качества

#Расчет ROC-AUC
def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))

#Расчет PR-AUC
def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))

#Расчет метрик качества модели
def evaluate_model(
    model: Model,
    x_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_cls: np.ndarray,
    threshold: float,
) -> Dict:
    pred_reg, pred_cls = model.predict(x_test, verbose=0)

    y_true = y_test_cls.ravel().astype(int)
    y_score = pred_cls.ravel()
    y_pred = (y_score >= threshold).astype(int)

    cls_metrics = {
        "THRESHOLD": threshold,
        "ACCURACY": float(accuracy_score(y_true, y_pred)),
        "PRECISION": float(precision_score(y_true, y_pred, zero_division=0)),
        "RECALL": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": safe_roc_auc(y_true, y_score),
        "PR-AUC": safe_average_precision(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    reg_metrics_by_target = {}
    for i, col in enumerate(REGRESSION_COLUMNS):
        mae = mean_absolute_error(y_test_reg[:, i], pred_reg[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_reg[:, i], pred_reg[:, i]))
        reg_metrics_by_target[col] = {
            "mae": float(mae),
            "rmse": float(rmse),
        }

    overall_regression = {
        "mae_macro": float(np.mean([m["mae"] for m in reg_metrics_by_target.values()])),
        "rmse_macro": float(np.mean([m["rmse"] for m in reg_metrics_by_target.values()])),
    }

    return {
        "classification": cls_metrics,
        "regression": {
            "overall": overall_regression,
            "by_target": reg_metrics_by_target,
        },
    }



#Формирование таблицы с предсказаниями на тестовой выборке
def make_predictions_frame(
    test_df: pd.DataFrame,
    model: Model,
    x_test_scaled: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    pred_reg, pred_cls = model.predict(x_test_scaled, verbose=0)

    result = test_df[META_COLUMNS + [CLASSIFICATION_COLUMN]].copy().reset_index(drop=True)
    result["bankruptcy_probability"] = pred_cls.ravel()
    result["bankruptcy_predicted"] = (result["bankruptcy_probability"] >= threshold).astype(int)

    for i, col in enumerate(REGRESSION_COLUMNS):
        result[f"pred_{col}"] = pred_reg[:, i]
        result[f"true_{col}"] = test_df[col].values

    return result



#Визуализация обучения
def save_training_plots(
    history: tf.keras.callbacks.History,
    plots_dir: Path,
    reports_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(reports_dir / "training_history.csv", index=False)

    #Сохранение одного графика обучения
    def plot_metric(metric_name: str, file_name: str, title: str) -> None:
        if metric_name not in history_df.columns:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(history_df[metric_name], label=metric_name)
        val_metric = f"val_{metric_name}"
        if val_metric in history_df.columns:
            plt.plot(history_df[val_metric], label=val_metric)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / file_name, dpi=150)
        plt.close()

    plot_metric("loss", "loss.png", "Total loss")
    plot_metric("classification_output_accuracy", "classification_accuracy.png", "Classification accuracy")
    plot_metric("classification_output_roc_auc", "classification_roc_auc.png", "Classification ROC-AUC")
    plot_metric("regression_output_mae", "regression_mae.png", "Regression MAE")


# =========================
#     ОБУЧЕНИЕ МОДЕЛИ
# =========================

def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    #Пути берутся из аргументов cmd
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    metrics_dir = model_dir / "metrics"
    plots_dir = model_dir / "plots"
    predictions_dir = model_dir / "predictions"
    reports_dir = model_dir / "reports"

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    train_df, val_df, test_df = make_split(df, args.split_mode)

    x_train, y_train_reg, y_train_cls = make_xy(train_df)
    x_val, y_val_reg, y_val_cls = make_xy(val_df)
    x_test, y_test_reg, y_test_cls = make_xy(test_df)

    x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_features(x_train, x_val, x_test)

    model = build_multitask_model(input_dim=len(FEATURE_COLUMNS))

    #Колбэки контролируют раннюю остановку, скорость обучения и сохранение лучшей модели.
    #Теперь лучшая эпоха выбирается по регрессии, иначе веса откатываются к эпохе,
    #где классификация уже хорошая, а регрессия еще не дообучена.
    regression_monitor = "val_regression_output_mae"
    callbacks = [
        EarlyStopping(
            monitor=regression_monitor,
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=regression_monitor,
            mode="min",
            factor=0.5,
            patience=max(3, args.patience // 2),
            min_lr=1e-5,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor=regression_monitor,
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    #Запуск обучения модели
    history = model.fit(
        x_train_scaled,
        {
            "regression_output": y_train_reg,
            "classification_output": y_train_cls,
        },
        validation_data=(
            x_val_scaled,
            {
                "regression_output": y_val_reg,
                "classification_output": y_val_cls,
            },
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    #Оценка модели на тестовой выборке
    metrics = evaluate_model(
        model=model,
        x_test=x_test_scaled,
        y_test_reg=y_test_reg,
        y_test_cls=y_test_cls,
        threshold=args.threshold,
    )

    predictions = make_predictions_frame(
        test_df=test_df,
        model=model,
        x_test_scaled=x_test_scaled,
        threshold=args.threshold,
    )

    #Сохранение обученной модели и скейлера
    model.save(model_dir / "financial_agent_model.keras")
    joblib.dump(scaler, model_dir / "scaler.pkl")



    #метадата и метрики
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "split_mode": args.split_mode,
        "random_state": args.seed,
        "feature_columns": FEATURE_COLUMNS,
        "regression_columns": REGRESSION_COLUMNS,
        "classification_column": CLASSIFICATION_COLUMN,
        "group_column": GROUP_COLUMN,
        "year_column": YEAR_COLUMN,
        "threshold": args.threshold,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_validation": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "target_distribution_total": df[CLASSIFICATION_COLUMN].value_counts().sort_index().to_dict(),
        "target_distribution_train": train_df[CLASSIFICATION_COLUMN].value_counts().sort_index().to_dict(),
        "target_distribution_validation": val_df[CLASSIFICATION_COLUMN].value_counts().sort_index().to_dict(),
        "target_distribution_test": test_df[CLASSIFICATION_COLUMN].value_counts().sort_index().to_dict(),
    }

    with open(metrics_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    predictions.to_csv(predictions_dir / "test_predictions.csv", index=False, sep=";")
    save_training_plots(history, plots_dir, reports_dir)



    print("\nОбучение завершено!")
    print(f"Артефакты сохранены в: {model_dir.resolve()}")

    print("\nКлассификационные метрики на тестовой выборке:")
    for key, value in metrics["classification"].items():
        #Интерпретация матрицы ошибок классификации
        if key == "confusion_matrix":
            print(f"  {key}: {value}")

            cm = np.array(value)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                print(f"    {tn} — правильно найденные небанкроты")
                print(f"    {fp} — ложные тревоги")
                print(f"    {fn} — пропущенные банкроты")
                print(f"    {tp} — правильно найденные банкроты")
        elif value is None:
            print(f"  {key}: None")
        else:
            print(f"  {key}: {value:.3f}")

    print("\nРегрессия:")
    print(f"  MAE macro:  {metrics['regression']['overall']['mae_macro']:.3f}")
    print(f"  RMSE macro: {metrics['regression']['overall']['rmse_macro']:.3f}")



# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели финансового состояния предприятий")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Путь к CSV-файлу с данными")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Папка для сохранения артефактов модели")
    parser.add_argument("--split-mode", choices=["group", "time", "random"], default="group", help="Способ разбиения данных")
    parser.add_argument("--epochs", type=int, default=200, help="Максимальное количество эпох")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер mini-batch")
    parser.add_argument("--patience", type=int, default=20, help="Patience для EarlyStopping")
    parser.add_argument("--threshold", type=float, default=0.50, help="Порог классификации банкротства")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Seed для воспроизводимости")
    return parser.parse_args()



if __name__ == "__main__":
    train_model(parse_args())