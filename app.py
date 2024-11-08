import os
import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from flask import Flask, render_template, request
from src.helper import download_hf_embeddings
from langchain_pinecone import PineconeVectorStore
from src.prompt import *


app = Flask(__name__)

load_dotenv()

# Assign Pinecone API key
api_key = os.getenv("PINECONE_API_KEY")

# Create vector embeddings using huggingface "all-MiniLM-L6-v2" model
embeddings = download_hf_embeddings()

# Initializing the pinecone
pc = pinecone.Pinecone(api_key=api_key)

# Provide pinecone index
index_name = "genai-chatbot"

# Exract stored embedded chunks in pincecone vector db
chunks_embeddings = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Initialize prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# LLM initialization
llm = CTransformers(
  model="model\llama-2-7b-chat.ggmlv3.q4_0.bin",
  model_type="llama",
  config={"max_new_tokens": 700, "temperature": 0.7}
)

# Instead of using RetrievalQA() directly, use RetrievalQA.from_chain_type():
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chunks_embeddings.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa.invoke({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
   app.run(host="0.0.0.0", port= 8080, debug= True)