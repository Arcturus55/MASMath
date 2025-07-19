import warnings
warnings.filterwarnings("ignore")

import os

from agents.knowledge_agent import KnowledgeLLMAgent, KnowledgeGraphAgent
from agents.structure_agent import StructureAgent
from agents.generation_agent import GenerationAgent
from agents.solver_agent import SolverAgent
from agents.judger_agent import JudgerAgent
from agents.checking_agent import CheckingAgent

class WorkFlow:
    def __init__(self, max_iter: int=5, eps: float=0.1, redund=True):
        # self.knowledge_agent = KnowledgeLLMAgent(model_name=os.getenv("Gemini-2.0"))
        self.knowledge_agent = KnowledgeGraphAgent(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            kb_path=os.path.join(os.getenv("PROJECT_PATH"), "kbase/kbase.json"),
            embed_model_name=os.getenv("Embedding_Model_ID")
        )
        self.structure_agent = StructureAgent(model_name=os.getenv("Gemini-2.0"))
        self.generation_agent = GenerationAgent(model_name=os.getenv("Gemini-2.0"))
        self.solver_agent = SolverAgent(model_name=os.getenv("GPT-4o"))
        self.judger_agent = JudgerAgent(model_name=os.getenv("Gemini-2.0"))
        self.checking_agent = CheckingAgent(
            model_name=os.getenv("Gemini-2.0"),
            data_file_path=os.path.join(os.getenv("PROJECT_PATH"), "data/jsons")
        )

        self.max_iter = max_iter
        self.eps = eps
        self.redund = redund
    
    def generate(self, info: dict) -> str:
        knowledge = self.knowledge_agent.ask(info["subject"])
        if self.redund:
            print('-'*50+'\n')
            print(knowledge)
        
        structure = self.structure_agent.ask(info, knowledge)
        if self.redund:
            print('-'*50+'\n')
            print(structure)

        question = None
        direct = None
        for i in range(self.max_iter):  
            if i == 0:  
                question = self.generation_agent.ask_structure(info, structure)
            else:
                question = self.generation_agent.modify(info, structure, question, direct)
            if self.redund:
                print('-'*50+'\n')
                print(question)
            
            answer = self.solver_agent.ask(question)
            if self.redund:
                print('-'*50+'\n')
                print(answer)

            score_text = self.judger_agent.ask(question, answer)
            score = self.judger_agent.extract(score_text)
            if self.redund:
                print('-'*50+'\n')
                print(score_text)
            
            examples, check_text = self.checking_agent.ask(question)
            check = self.checking_agent.extract(check_text)
            if self.redund:
                print('-'*50+'\n')
                print(check_text)
            
            if abs(score - info["difficulty"]) <= self.eps and not check:
                break
            direct = "上升" if score < info["difficulty"] else "下降"
            
        return question