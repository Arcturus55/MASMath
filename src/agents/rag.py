from camel.embeddings import OpenAICompatibleEmbedding
from camel.storages import QdrantStorage
from camel.retrievers import VectorRetriever
import PyPDF2
import pathlib
import os
import json
from dotenv import load_dotenv

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / ".env"
load_dotenv(dotenv_path=str(env_path))

modeltype = os.getenv("Embedding_Model_ID")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_URL")

# print(api_key, base_url)

embedding_instance = OpenAICompatibleEmbedding(model_type=modeltype, api_key=api_key, url=base_url)

text_embeddings = embedding_instance.embed_list(["What is the capital of France?"])
print(len(text_embeddings[0]))

storage_instance = QdrantStorage(
    vector_dim=3072,
    # path="local_data",
    collection_name="test",
)

vector_retriever = VectorRetriever(embedding_model=embedding_instance,
                                   storage=storage_instance)

# def read_pdf_with_pypdf2(file_path):
#     with open(file_path, "rb") as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text += page.extract_text()
#     return text

input_path = "/home/ubuntu/wangpengyuan/lzb/MASMath/data/jsons/quanguo2.json"
with open(input_path, 'r') as f:
    datas = json.loads(f.read())
# pdf_text = read_pdf_with_pypdf2(input_path)

# print(pdf_text)

# print("向量化中...")

# vector_retriever.process(
#     content=pdf_text,
# )

for d in datas:

    vector_retriever.process(
        content=d,
        should_chunk=False,
        max_characters=1000
    )


# vector_retriever.storage.

# print("向量化完成")

prompt = '''已知数列 $\{a_n\}$ 满足 $a_1 = 1$，且 $a_{n+1} = 2a_n + 1$。

（1）求证：数列 $\{a_n + 1\}$ 是等比数列，并求数列 $\{a_n + 1\}$ 的通项公式。

（2）设 $b_n = \frac{1}{a_n + 1}$，求数列 $\{b_n\}$ 的前 $n$ 项和 $S_n$。'''


resp = vector_retriever.query(prompt, top_k=3, similarity_threshold=0.0)

print(resp)