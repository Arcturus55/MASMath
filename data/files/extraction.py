import os
import pathlib
from dotenv import load_dotenv
from openai import OpenAI

env_path = "/home/ubuntu/wangpengyuan/lzb/MASMath/src/.env"
load_dotenv(dotenv_path=str(env_path))

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_URL")

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

client.chat.completions.create
