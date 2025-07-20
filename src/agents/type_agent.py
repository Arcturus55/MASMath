import os
import re
import json
from typing import List
from openai import OpenAI
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent

from .base_agent import BaseAgent

class TypeAgent(BaseAgent):
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

    def ask(self) -> str:

        input = '''请用json格式生成一份全国卷高考数学的题型设置，其中每一条数据都是一道题的设置，需要包含这道题目考察的主题、难度以及题型（这里难度的取值范围为0到1，值越大代表题目越难），以下是一个示例：
{
    "subject": "数列",
    "difficulty": 0.2,
    "type": "选择"
}
你需要把生成的json文件内容放在<json>和</json>之间。'''

        resp = self.agent.step(input)

        return resp.msgs[0].content
    
    def extract(self, answer: str) -> List[dict]:
        pattern = re.compile(r"<json>(.*?)</json>", re.DOTALL)
        matches = pattern.findall(answer)
        if matches:
            return json.loads(matches[-1])
        else:
            return None
