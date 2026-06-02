import os
import httpx

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_NIM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    http_client=httpx.Client(
        verify=False,
        timeout=60
    )
)

print("Sending request...")

response = client.embeddings.create(
    model=os.getenv("EMBEDDING_MODEL"),
    input=["hello world"]
)

print("Success")
print(len(response.data[0].embedding))