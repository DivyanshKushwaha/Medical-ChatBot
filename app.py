# from flask import Flask, render_template, jsonify, request
# from src.helper import get_gemini_embedding
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from src.prompt import prompt_template
# from langchain_google_genai import ChatGoogleGenerativeAI

# # from langchain.vectorstores import Pinecone

# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.document_loaders import DirectoryLoader,PyPDFDirectoryLoader,PyPDFLoader
# from langchain.document_loaders import TextLoader
# import warnings
# warnings.filterwarnings("ignore")
# from dotenv import load_dotenv
# import os


# app = Flask(__name__)

# load_dotenv()
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')



# # Initialize Pinecone using the new method
# from pinecone import Pinecone, ServerlessSpec
# pc = Pinecone(api_key=PINECONE_API_KEY)


# index_name = 'llama-chatbot'
# index = pc.Index(index_name)


# # Create the Pinecone vector store
# from langchain.vectorstores import Pinecone
# vectorstore = Pinecone.from_existing_index(
#     index_name=index_name,  # Pinecone index name
#     embedding=get_gemini_embedding,  # Embedding function for queries
#     namespace="medical-chat"  # Correct namespace
# )

# # Now get a retriever
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # Use an updated prompt template for better conversation handling
# prompt = PromptTemplate(template=prompt_template,input_variables=['context','question'])

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0.8, max_tokens=512)

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt},
# )

# # Predefined responses for general chat
# general_chat_responses = {
#     "hi": "Hi! How can I assist you today?",
#     "hello": "Hello! How can I help?",
#     "how are you?": "I'm fine! How can I assist you today?",
#     "what's up?": "Not much, just here to help! What do you need?",
#     "who are you?": "I'm a helpful AI assistant! Ask me anything.",
# }


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"].strip().lower()
#     print("User:", msg)

#     # Handle general chat
#     if msg in general_chat_responses:
#         result = general_chat_responses[msg]
#     else:
#         result = qa.run(msg)
#         # If the response is too generic, replace with a more conversational one
#         if "I don't have access" in result or "I can't provide" in result:
#             result = "I may not have specific details, but I'm happy to help! What do you need?"

#     print("Response:", result)
#     return str(result)


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)




# 2nd method 

# from flask import Flask, render_template, jsonify, request
# from src.helper import get_gemini_embedding
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from src.prompt import prompt_template
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import warnings
# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone

# warnings.filterwarnings("ignore")

# app = Flask(__name__)

# load_dotenv()
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# INDEX_HOST = os.environ.get('INDEX_HOST')

# # Initialize Pinecone using the SDK
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index("llama-chatbot")

# # Define retrieval function using Pinecone SDK
# def retrieve_documents(text, top_k=3, namespace="medical-chat"):
#     query_embedding = list(get_gemini_embedding(text))
#     results = index.query(vector=query_embedding, top_k=top_k, namespace=namespace, include_metadata=True)
#     documents = [match['metadata']['text'] for match in results['matches']]
#     return documents

# # Prompt setup
# prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# # Initialize LLM
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8, max_tokens=512)

# # Function to handle responses
# def ask_question(question):
#     retrieved_docs = retrieve_documents(question)
#     context = "\n".join(retrieved_docs)
#     formatted_prompt = prompt.format(context=context, question=question)
#     response = llm.invoke(formatted_prompt)
#     return response.content

# # Predefined responses for general chat
# general_chat_responses = {
#     "hi": "Hi! How can I assist you today?",
#     "hello": "Hello! How can I help?",
#     "how are you?": "I'm fine! How can I assist you today?",
#     "what's up?": "Not much, just here to help! What do you need?",
#     "who are you?": "I'm a helpful AI assistant! Ask me anything.",
# }

# @app.route("/")
# def index():
#     return render_template('chat.html')

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"].strip().lower()
#     print("User:", msg)

#     # Handle general chat
#     if msg in general_chat_responses:
#         result = general_chat_responses[msg]
#     else:
#         result = ask_question(msg)
#         if "I don't have access" in result or "I can't provide" in result:
#             result = "I may not have specific details, but I'm happy to help! What do you need?"

#     print("Response:", result)
#     return str(result)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)





# 3rd method 


from flask import Flask, render_template, jsonify, request
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from src.prompt import prompt_template

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

# Initialize Pinecone and connect to the index
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "llama-chatbot"
namespace = "new-test"

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Pinecone Vector Store
vectorstore = PineconeVectorStore(
    index_name=index_name,
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embedding_model,
    namespace=namespace
)

# Define retrieval function
def retrieve_documents(text, top_k=3):
    results = vectorstore.similarity_search(text, k=top_k)
    documents = [doc.page_content for doc in results]
    return documents

# Setup prompt
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8, max_tokens=512)

# Function to handle responses
def ask_question(question):
    retrieved_docs = retrieve_documents(question)
    context = "\n".join(retrieved_docs)
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    return response.content

# Predefined responses for general chat
general_chat_responses = {
    "hi": "Hi! How can I assist you today?",
    "hello": "Hello! How can I help?",
    "how are you?": "I'm fine! How can I assist you today?",
    "what's up?": "Not much, just here to help! What do you need?",
    "who are you?": "I'm a helpful AI assistant! Ask me anything.",
}

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"].strip().lower()
    print("User:", msg)

    # Handle general chat
    if msg in general_chat_responses:
        result = general_chat_responses[msg]
    else:
        result = ask_question(msg)
        if "I don't have access" in result or "I can't provide" in result:
            result = "I may not have specific details, but I'm happy to help! What do you need?"

    print("Response:", result)
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

