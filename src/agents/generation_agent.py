import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent

from .base_agent import BaseAgent

class GenerationAgent(BaseAgent):
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

    def ask_structure(self, info: dict, structure: str) -> str:

        query = f"假设你现在要命制一道有关高中数学{info.get('subject')}的难度为{info.get('difficulty')}的{info.get('type')}题（这里难度的取值范围为0到1，值越大代表题目越难）。"

        sys_prompt = "请根据上述给出的目标和题目结构，命制一道高中数学题。你需要给出题目的具体题面，其中公式使用latex格式，在题面中使用（1）、（2）等来标识各小问。请注意，你给出的题目应该确保题目难度符合给出的题目结构中的难度要求，并且随小问逐级递增。此外，题面中请尽量不要包含图像信息，如果一定要包含图像，请尽可能详细地描述这个图像。最后，在回答中只需要给出题目，不需要包含题目的答案或其他额外表述。"

        input = f"目标：{query} \n\n题目结构： {structure} \n\n{sys_prompt}"

        resp = self.agent.step(input)

        return resp.msgs[0].content
    
    def modify(self, info: dict, structure: str, prev_question: str, direct: str) -> str:
        query = f"假设你现在要命制一道有关高中数学{info.get('subject')}的难度为{info.get('difficulty')}的{info.get('type')}题（这里难度的取值范围为0到1，值越大代表题目越难）。"

        sys_prompt = f"请根据上述给出的目标和题目结构和给出的例题，对这道例题进行修改，使其难度{direct}。你需要给出题目的具体题面，其中公式使用latex格式，题面中请尽量不要包含图像信息，如果一定要包含图像，请尽可能详细地描述这个图像。最后，在回答中只需要给出题目，不需要包含题目的答案或其他额外表述。"

        input = f"目标：{query} \n\n题目结构： {structure} \n\n待修改的例题： {prev_question} \n\n{sys_prompt}"

        resp = self.agent.step(input)

        return resp.msgs[0].content

    def ask_query(self) -> str:
        pass