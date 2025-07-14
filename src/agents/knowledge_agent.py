import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent

from .base_agent import BaseAgent

class KnowledgeAgent(BaseAgent):
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

    def ask(self, info: dict) -> str:

        query = f"假设你现在要命制一道有关高中数学{info.get('subject')}的综合难度难度为{info.get('difficulty')}的{info.get('type')}题（这里难度的取值范围为0到1，值越大代表题目越难）。"

        sys_prompt = "请根据上述目标，首先确定要求中的题目主题，然后根据这个主题，给出3个可以考察的知识点。回答时，对每个知识点，请分别回答这个知识点的名称、认知层级、基准难度以及可以组合的知识点。请注意，回答中不需要包含例题或答案。"

        input = f"目标： {query} \n\n{sys_prompt}"

        resp = self.agent.step(input)

        return resp.msgs[0].content