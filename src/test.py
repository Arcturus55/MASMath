import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv

PROJECT_PATH = "D:\DevCodes\MASMath"

load_dotenv(os.path.join(PROJECT_PATH, ".env"))

from workflow import WorkFlow

mas = WorkFlow()
info = {
    "subject": "数列",
    "difficulty": 0.5,
    "type": "简答"
}

print(mas.generate(info))
