import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
PROJECT_PATH = "D:\DevCodes\MASMath"
load_dotenv(os.path.join(PROJECT_PATH, ".env"))

from workflow import WorkFlow
from agents.type_agent import TypeAgent

type_agent = TypeAgent(model_name=os.getenv("Deepseek-R1"))
resp = type_agent.ask()
data = type_agent.extract(resp)
print(data)

mas = WorkFlow(redund=False)

for i, info in enumerate(data):
    print()
    print(f"{i+1}.", mas.generate(info))

