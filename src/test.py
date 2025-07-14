import os

os.environ["OPENAI_API_KEY"] = "sk-k0j35XYUqIbm03151d354f010dBc4347953d8fC3Dd2e3200"
os.environ["OPENAI_API_URL"] = "https://api.shubiaobiao.cn/v1/"

from agents.knowledge_agent import KnowledgeAgent
from agents.structure_agent import StructureAgent

k_agent = KnowledgeAgent(model_name="gemini-2.0-flash")
s_agent = StructureAgent(model_name="gemini-2.0-flash")

prompt = "请命制一道高中数学简答题，内容有关三角函数。"

knowledge = k_agent.ask(prompt)

structure = s_agent.ask(prompt, knowledge)

print(knowledge)

print()

print(structure)