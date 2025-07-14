import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent

from .base_agent import BaseAgent

class SolverAgent(BaseAgent):
    def __init__(self, model_name: str, temperature: float=0.4):

        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                    model_type=model_name,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    url=os.environ.get("OPENAI_API_URL"),
                    model_config_dict={"temperature": temperature},
        )

        self.agent = ChatAgent(
            model=model,
        )

    def ask(self, question: str) -> str:

        sys_prompt = "以下是一道高中数学题，请为其提供详细的解答。请注意，在回答中应该给出详细的解题思路和步骤，除此以外不需要包含任何额外表述。"
        
        input = f"{sys_prompt} \n\n题目： {question}"

        resp = self.agent.step(input)

        return resp.msgs[0].content
    