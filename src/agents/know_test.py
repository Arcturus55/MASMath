from py2neo import Graph, Node, Relationship, Subgraph
import json
import random

# 连接 Neo4j 数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "masmath"))
graph.delete_all()  # 清空现有数据（可选）


def batch_build_graph(json_data):
    knowledge_point_map = {}
    nodes, relationships = [], []
    
    for subject in json_data:
        chapter_node = Node("Chapter", name=subject["chapter"])
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
    graph.create(subgraph)
    
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
    
    graph.create(Subgraph(relationships=combine_rels))

# 加载 JSON 数据
with open("D:\DevCodes\MASMath\kbase\kbase.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 构建知识图谱
batch_build_graph(data)

def get_knowledge_points_by_chapter(chapter_name):
    """
    根据章节名称获取所有知识点
    """
    query = """
    MATCH (c:Chapter {name: $chapter_name})-[:HAS_TOPIC]->(t:Topic)-[:CONTAINS_POINT]->(kp:KnowledgePoint)
    RETURN kp.name AS name, kp.cognitive AS cognitive, kp.difficulty AS difficulty
    """
    return graph.run(query, chapter_name=chapter_name).data()

def get_combined_knowledge_points(point_name):
    """
    获取与指定知识点组合的其他知识点
    """
    query = """
    MATCH (kp:KnowledgePoint {name: $point_name})-[:COMBINES_WITH]->(related:KnowledgePoint)
    RETURN related.name AS name
    """
    results = graph.run(query, point_name=point_name).data()
    return [item['name'] for item in results]

def get_random_knowledge_points(chapter_name):
    """
    主函数：随机选择三个知识点并返回详细信息
    """
    # 1. 获取该章节所有知识点
    all_points = get_knowledge_points_by_chapter(chapter_name)
    
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
        combined_points = get_combined_knowledge_points(point['name'])
        
        result.append({
            "知识点名称": point['name'],
            "认知层级": point['cognitive'],
            "难度": point['difficulty'],
            "可组合知识点": combined_points
        })
    
    return result

chapter_name = "导数及其应用"
results = get_random_knowledge_points(chapter_name)

print(results)