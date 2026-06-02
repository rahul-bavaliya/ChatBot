import os
import pandas as pd
import httpx
from dotenv import load_dotenv

from openai import OpenAI

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


class NvidiaEmbeddings(Embeddings):

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_NIM_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            http_client=httpx.Client(verify=False)
        )

        self.model = os.getenv("EMBEDDING_MODEL")

    def embed_documents(self, texts):

        print(f"\nRequesting embeddings for {len(texts)} documents...")

        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float",
            extra_body={
                "input_type": "passage",
                "truncate": "NONE"
            }
        )

        print(f"Embeddings received for {len(response.data)} documents.")

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


print("Loading CSV...")

df = pd.read_csv("serviceorder.csv")

documents = []

for _, row in df.iterrows():

    text = f"""
Service Order ID: {row['ServiceOrderID']}
Work Order Number: {row['Wo Number']}
Make: {row['Make']}
Model: {row['Model']}
Meter Reading: {row['Meter']}
Work Order Segment: {row['Wo Segment']}
Description: {row['Description']}
Contract Type: {row['Contract Type']}
Actual Labor Hours: {row['M_Actual_ServiceOrderLaborHrsTotal']}
Work Order Status: {row['Wo Status']}
"""

    documents.append(
        Document(
            page_content=text,
            metadata={
                "service_order_id": str(row["ServiceOrderID"]),
                "wo_number": str(row["Wo Number"]),
                "model": str(row["Model"]),
                "description": str(row["Description"]),
            }
        )
    )

print(f"Creating embeddings for {len(documents)} records...")

embeddings = NvidiaEmbeddings()

BATCH_SIZE = 25

db = None

for start in range(0, len(documents), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(documents))

    batch_docs = documents[start:end]

    print(
        f"Embedding documents {start+1}-{end} "
        f"of {len(documents)}"
    )

    if db is None:
        db = FAISS.from_documents(
            batch_docs,
            embeddings
        )
    else:
        db.add_documents(batch_docs)

    print(
        f"Completed: {end/len(documents)*100:.1f}%"
    )

db.save_local("serviceorder_index")

print("Done.")
print("FAISS index saved to serviceorder_index/")