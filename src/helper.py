from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Function for Loading pdf data
def pdf_load(data):
  loader = DirectoryLoader(
              data,
              glob="*.pdf",
              loader_cls=PyPDFLoader)
  
  docs = loader.load()
  return docs


# Function for splitting the loaded data
def split_text(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
  text_chunks = text_splitter.split_documents(extracted_data)
  return text_chunks


# Download embedding model
def download_hf_embeddings():
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embeddings