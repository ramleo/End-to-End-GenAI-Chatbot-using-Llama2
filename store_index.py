import os
import pinecone
from src.helper import (pdf_load, split_text, 
                        download_hf_embeddings)
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


load_dotenv()

# Note as on 07/11/2024, we are not required to mention Pinecone Environment
api_key = os.getenv("PINECONE_API_KEY")

# Load the pdf file
extracted_data = pdf_load("data/")

# Create chunks of pdf file
text_chunks = split_text(extracted_data)

# Create vector embeddings using huggingface "all-MiniLM-L6-v2" model
# It takes 3 mins to execute
embeddings = download_hf_embeddings()

# Initializing the pinecone
pc = pinecone.Pinecone(api_key=api_key)

# Provide pinecone index
index_name = "genai-chatbot"

# Store embedded chunks in pincecone vector db
chunks_embeddings = PineconeVectorStore.from_texts([chunk.page_content for chunk in text_chunks], embeddings, index_name=index_name)

