from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os 
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')



#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents



#Create text chunks
def text_split(data_extracted):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data_extracted)
    return text_chunks



#download embedding model
# def download_embeddings():
#     # Define the path where the embeddings should be saved
#     cache_dir = os.path.join(os.getcwd(), "modelEmbedd")
    
#     # Initialize embeddings with the specified cache directory
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         cache_folder=cache_dir
#     )
#     return embeddings


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=GEMINI_API_KEY)

def get_gemini_embedding(text):
    result = embeddings.embed_query(text)
    return result

