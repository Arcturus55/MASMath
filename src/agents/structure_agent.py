import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent

from .base_agent import BaseAgent

class StructureAgent(BaseAgent):
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

    def ask(self, info: dict, knowledge: str) -> str:

        query = f"假设你现在要命制一道有关高中数学{info.get('subject')}的综合难度难度为{info.get('difficulty')}的{info.get('type')}题（这里难度的取值范围为0到1，值越大代表题目越难）。"

        sys_prompt = "请根据上述目标以及给出的知识点，确定题目的结构。你需要给出题目共设置几个小问，每个小问的难度，以及每个小问考察的知识点。请注意，你给出的题目结构应该确保题目难度随小问逐级递增，并且在回答中不需要包含题目的具体内容或答案。"

        input = f"目标：{query} \n\n知识点： {knowledge} \n\n{sys_prompt}"

        resp = self.agent.step(input)

        return resp.msgs[0].content