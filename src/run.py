from __future__ import annotations

import sys
from pathlib import Path



#Определение корневой папки
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#Добавление корневой папки проекта в путь поиска модулей
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main

if __name__ == "__main__":
    raise SystemExit(main())