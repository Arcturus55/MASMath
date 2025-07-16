import os
import json
from tqdm import tqdm

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.embeddings import OpenAICompatibleEmbedding
from camel.storages import QdrantStorage
from camel.retrievers import VectorRetriever

from .base_agent import BaseAgent

class CheckingAgent(BaseAgent):
    def __init__(self, 
                 model_name: str="gemini-2.0-flash",  
                 temperature: float=0.4,
                 embed_model_name: str="text-embedding-3-large",
                 data_file_path: str=None
                ):

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

        embedding_instance = OpenAICompatibleEmbedding(
            model_type=embed_model_name, 
            api_key=os.environ.get("OPENAI_API_KEY"),
            url=os.environ.get("OPENAI_API_URL"),
        )

        storage_instance = QdrantStorage(
            vector_dim=3072,
            path="local_data",
            collection_name="test",
        )
        storage_instance.clear()

        self.vector_retriever = VectorRetriever(
            embedding_model=embedding_instance,
            storage=storage_instance
        )

        files = [os.path.join(data_file_path, file) for file in os.listdir(data_file_path)]
        datas = []
        for file in files:
            with open(file, 'r') as f:
                datas.extend(json.loads(f.read()))

        print("正在构建数据库...")
        for d in tqdm(datas):
            self.vector_retriever.process(
                content=d,
                should_chunk=False,
                max_characters=1000
            )
        print("数据库构建完成")

    def ask(self, question: str) -> str:
        rag_result = self.vector_retriever.query(question, top_k=3, similarity_threshold=0.0)
        examples = [r['text'] for r in rag_result]

        sys_prompt = "上面是一道高中数学题，以及三道与之相关的例题。请判断目标题目是否和例题存在过度雷同的现象。你需要进行详细的分析，并将最终结果（是、否）放在<answer>和</answer>之间。"
        
        input = f"目标题目： {question} \n\n例题一： {examples[0]} \n\n例题二： {examples[1]} \n\n例题三： {examples[2]} \n\n{sys_prompt} \n"

        resp = self.agent.step(input)

        return examples, resp.msgs[0].content
    