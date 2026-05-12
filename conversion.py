from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
excel_file_path = PROJECT_ROOT / "data" / "data_raw.xlsx"
csv_file_path = PROJECT_ROOT / "data" / "data_processed.csv"

#Диапазон столбцов, который переносится
columns_to_import = "A:AX"

df = pd.read_excel(excel_file_path, usecols=columns_to_import)

#Удаление строк с пропущенными значениями
df.dropna(inplace=True)

#Сохранение подготовленного датасета
df.to_csv(csv_file_path, index=False, sep=";")

print(f"Успешно экспортировано в {csv_file_path}")