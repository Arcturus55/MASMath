import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent

from .base_agent import BaseAgent

class JudgerAgent(BaseAgent):
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

    def ask(self, question: str, answer: str) -> str:

        sys_prompt = "以下是一道高中数学题及其解答，请仔细分析这道题，并为它进行难度打分。请注意，这里难度打分的取值范围为0到1，值越大代表题目越难。你应该给出详细的分析，并将最终难度得分的值放在<score>和</score>之间。"
        
        input = f"{sys_prompt} \n\n题目： {question} \n\n解答： {answer}"

        resp = self.agent.step(input)

        return resp.msgs[0].content
    