from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_embeddings()

# Initialize Pinecone using the new method
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Initialize the index
index = pinecone_client.Index(index_name)

# Create the Pinecone vector store
vectorstore = Pinecone(
    index=index,  # Pinecone index instance
    embedding=embeddings.embed_query,  # Embedding function
    text_key="text"  # Key in metadata containing the document text
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                   model_type="llama",
                   config={'max_new_tokens': 512,
                           'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa.run(input)
    print("Response : ", result)
    return str(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
