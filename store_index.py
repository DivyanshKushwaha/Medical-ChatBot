from src.helper import load_pdf, text_split, download_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY,environment=PINECONE_API_ENV)

# Connect to your existing index
index_name = "medical-chatbot"
index = pc.Index(index_name)

#Creating Embeddings for Each of The Text Chunks & storing
upsert_data = [(str(i), embeddings.embed_query(text_chunks[i].page_content),{"text": text_chunks[i].page_content}) for i in range(len(text_chunks))]

# storing in batch to avoid size issues
batch_size = 100  # or any number that suits your data size
for i in range(0, len(upsert_data), batch_size):
    batch = upsert_data[i:i + batch_size]
    index.upsert(vectors=batch)