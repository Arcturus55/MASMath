import os
import json
import random
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from openai import OpenAI
from py2neo import Graph, Node, Relationship, Subgraph
from sentence_transformers import util

from .base_agent import BaseAgent

class KnowledgeLLMAgent(BaseAgent):
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

class KnowledgeGraphAgent(BaseAgent):
    def __init__(self, uri: str, user: str, password: str, kb_path: str, embed_model_name: str):
        self.graph = Graph(uri, auth=(user, password))
        self.graph.delete_all()
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chapters = []
        print("正在构建知识图谱...")
        self._batch_build_graph(data)
        print("知识图谱构建完成")
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_URL")
        )
        self.embed_model_name = embed_model_name
    
    def _batch_build_graph(self, json_data):
        knowledge_point_map = {}
        nodes, relationships = [], []
        
        for subject in json_data:
            chapter_node = Node("Chapter", name=subject["chapter"])
            self.chapters.append(subject["chapter"])
            nodes.append(chapter_node)
            
            for topic in subject["topics"]:
                topic_node = Node("Topic", name=topic["topic"])
                nodes.append(topic_node)
                relationships.append(Relationship(chapter_node, "HAS_TOPIC", topic_node))
                
                for kp in topic["knowledge_points"]:
                    kp_node = Node("KnowledgePoint", 
                                name=kp["name"],
                                cognitive=kp["cognitive"],
                                difficulty=kp["difficulty"])
                    nodes.append(kp_node)
                    knowledge_point_map[kp["name"]] = kp_node
                    relationships.append(Relationship(topic_node, "CONTAINS_POINT", kp_node))
        
        # 批量创建节点和层级关系
        subgraph = Subgraph(nodes, relationships)
        self.graph.create(subgraph)
        
        # 批量创建组合关系
        combine_rels = []
        for subject in json_data:
            for topic in subject["topics"]:
                for kp in topic["knowledge_points"]:
                    source = knowledge_point_map[kp["name"]]
                    for target_name in kp["combinations"]:
                        if target_name in knowledge_point_map:
                            target = knowledge_point_map[target_name]
                            combine_rels.append(Relationship(source, "COMBINES_WITH", target))
        
        self.graph.create(Subgraph(relationships=combine_rels))

    def get_kp_by_chapter(self, chapter_name):
        """
        根据章节名称获取所有知识点
        """
        query = """
        MATCH (c:Chapter {name: $chapter_name})-[:HAS_TOPIC]->(t:Topic)-[:CONTAINS_POINT]->(kp:KnowledgePoint)
        RETURN kp.name AS name, kp.cognitive AS cognitive, kp.difficulty AS difficulty
        """
        return self.graph.run(query, chapter_name=chapter_name).data()

    def get_combined_kp(self, point_name):
        """
        获取与指定知识点组合的其他知识点
        """
        query = """
        MATCH (kp:KnowledgePoint {name: $point_name})-[:COMBINES_WITH]->(related:KnowledgePoint)
        RETURN related.name AS name
        """
        results = self.graph.run(query, point_name=point_name).data()
        return [item['name'] for item in results]

    def get_random_kp(self, chapter_name):
        """
        主函数：随机选择三个知识点并返回详细信息
        """
        # 1. 获取该章节所有知识点
        all_points = self.get_kp_by_chapter(chapter_name)
        
        if not all_points:
            return {"error": f"未找到章节 '{chapter_name}' 或该章节下无知识点"}
        
        # 2. 随机选择三个知识点
        if len(all_points) <= 3:
            selected_points = all_points
        else:
            selected_points = random.sample(all_points, 3)
        
        # 3. 为每个知识点获取组合知识点
        result = []
        for point in selected_points:
            combined_points = self.get_combined_kp(point['name'])
            
            result.append({
                "知识点名称": point['name'],
                "认知层级": point['cognitive'],
                "难度": point['difficulty'],
                "可组合知识点": combined_points
            })
        
        return result
    
    def find_most_similar(self, query, candidates):
        q_res = self.client.embeddings.create(input=[query], model=self.embed_model_name)
        q_emb = [r.embedding for r in q_res.data]
        c_res = self.client.embeddings.create(input=candidates, model=self.embed_model_name)
        c_emb = [r.embedding for r in c_res.data]
        cos_scores = util.cos_sim(q_emb, c_emb)[0]
        best_idx = cos_scores.argmax().item()
        return candidates[best_idx], cos_scores[best_idx].item()

    def ask(self, topic: str) -> str:
        chap, score = self.find_most_similar(topic, self.chapters)
        knowledge_points = self.get_random_kp(chap)
        answer = ""
        for i, kp in enumerate(knowledge_points):
            answer += f'''{i+1}. **知识点名称：** {kp["知识点名称"]}
    *   **认知层级：** {kp["认知层级"]}
    *   **基准难度：** {kp["难度"]}
    *   **可以组合的知识点：** {"、".join(kp["可组合知识点"])}
'''
        return answer