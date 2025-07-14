import os

os.environ["OPENAI_API_KEY"] = "sk-k0j35XYUqIbm03151d354f010dBc4347953d8fC3Dd2e3200"
os.environ["OPENAI_API_URL"] = "https://api.shubiaobiao.cn/v1/"

from agents.knowledge_agent import KnowledgeAgent
from agents.structure_agent import StructureAgent
from agents.generation_agent import GenerationAgent
from agents.solver_agent import SolverAgent
from agents.judger_agent import JudgerAgent

k_agent = KnowledgeAgent(model_name="gemini-2.0-flash")
s_agent = StructureAgent(model_name="gemini-2.0-flash")
g_agent = GenerationAgent(model_name="gemini-2.0-flash")
solver_agent = SolverAgent(model_name="gemini-2.0-flash")
j_agent = JudgerAgent(model_name="gemini-2.0-flash")

# prompt = "请命制一道高考数学简答题，内容有关导数压轴题。"

info = {
    "subject": "数列",
    "difficulty": 0.5,
    "type": "简答"
}

knowledge = k_agent.ask(info)
print(knowledge)

structure = s_agent.ask(info, knowledge)
print('-'*50+'\n')
print(structure)

question = g_agent.ask_structure(info, structure)
print('-'*50+'\n')
print(question)

ans = solver_agent.ask(question)
print('-'*50+'\n')
print(ans)

score = j_agent.ask(question, ans)
print('-'*50+'\n')
print(score)