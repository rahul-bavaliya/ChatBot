import os

from dotenv import load_dotenv

from openai import OpenAI

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

import httpx


class NvidiaEmbeddings(Embeddings):

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_NIM_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            http_client=httpx.Client(verify=False)
        )

        self.model = os.getenv("EMBEDDING_MODEL")

    def embed_documents(self, texts):

        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float",
            extra_body={
                "input_type": "passage",
                "truncate": "NONE"
            }
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text):

        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            encoding_format="float",
            extra_body={
                "input_type": "query",
                "truncate": "NONE"
            }
        )

        return response.data[0].embedding


client = OpenAI(
    api_key=os.getenv("NVIDIA_NIM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    http_client=httpx.Client(verify=False)
)

embeddings = NvidiaEmbeddings()

db = FAISS.load_local(
    "serviceorder_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("\nService Order Chatbot Ready")
print("Type 'exit' to quit\n")


while True:

    question = input("Question: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(
        question,
        k=10
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a heavy equipment service order analyst.

Rules:
1. Use ONLY the provided context.
2. Do not invent information.
3. If data is unavailable, say:
   'Information not found in service order data.'

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=[
            {
                "role": "system",
                "content": "You answer questions about service orders."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    print("\nAnswer:")
    print(answer)
    print("\n" + "-" * 80 + "\n")