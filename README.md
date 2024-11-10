# End-to-End-GenAI-Chatbot-using-Llama2

## Steps to run project

# Create virtual environment
```bash
python -m venv genaichatbot
```

# Activate virtual environment
```bash
source ./genaichatbot/Scripts/activate
```

# Run requirements.txt
```bash
pip install requirements.txt
```

# Download the quantize model from the link provided in model folder & keep the model in the model directory:

## Download the Llama 2 Model:
```
llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

# run the following command
```bash
python store_index.py
```

# Finally run the following command
```bash
python app.py
```

# Then open localhost
```
open up localhost:8080 or  http://127.0.0.1:8080
```

Techstack Used:
- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone

Steps explaining the code:

Import packages:
```
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
```
1. Downloading the LLM:

 ```
!wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```

This line uses wget to download the Llama 2 7B Chat GGML model from Hugging Face. 
This is the Large Language Model (LLM) that will be used for answering questions.

2. Setting up Pinecone:
```
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
embeddings = download_hf_embeddings()
pinecone.Pinecone(api_key=api_key)
```
These lines set up the Pinecone connection.
First, the Pinecone API key is set as an environment variable and then retrieved.
The download_hf_embeddings() function is called to download and initialize a Hugging Face embeddings model.
Finally, pinecone.Pinecone(api_key=api_key) initializes the Pinecone connection using the provided API key.

3. Defining the Prompt Template:
```
prompt_template = """
Use the following pieces of information to answer the user's questions.
If you don't know the answer, just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only return helpful answer and nothing else.
Helpful answer:
"""
```
This defines the prompt template that will be used to structure the questions and context for the LLM. 
It includes placeholders for context and question.

4. Loading the Vector Store and Creating the Prompt:
```
index_name = "genai-chatbot"
chunks_embeddings = PineconeVectorStore.from_existing_index(index_name,embeddings)
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}
```
`index_name` is set to "genai-chatbot", which is the name of your Pinecone index.
`PineconeVectorStore.from_existing_index` loads the vector store from your Pinecone index using the provided index name and embeddings. 
Using index which is already created.
The prompt template is initialized using PromptTemplate.
chain_type_kwargs is a dictionary to store the prompt and will be passed into the QA chain later.

5. Initializing the LLM:
```
llm = CTransformers(
  model="/content/llama-2-7b-chat.ggmlv3.q4_0.bin",
  model_type="llama",
  config={"max_new_tokens": 1000, "temperature": 0.7}
)
```
This initializes the CTransformers LLM with the downloaded Llama 2 model.
max_new_tokens limits the length of the generated response.
temperature controls the randomness of the output.

6. Creating the QA Chain:
```
qa1 = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chunks_embeddings.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)
```
This line of code creates a question-answering system using LangChain's RetrievalQA class. 
It connects different components to enable answering questions based on a provided set of documents.

Here's a breakdown of each argument:

`RetrievalQA.from_chain_type(...):` This part initializes a RetrievalQA chain, 
which is a specialized type of chain designed for question-answering tasks that 
retrieve relevant context from a document store.

`llm=llm:` This specifies the large language model (LLM) to use for generating answers.
In this case, it's set to the llm object (which is the Llama2 model).

`chain_type="stuff":` This sets the chain type to "stuff". 
In a "stuff" chain, the entire retrieved context is inserted into the prompt before being sent to the LLM.
This is a simple but effective way to provide context to the model.

`retriever=chunks_embeddings.as_retriever(search_kwargs={"k": 2}):`
This configures the retriever, which is responsible for fetching relevant 
documents or information from your document store (Pinecone in this case).

`chunks_embeddings.as_retriever()` converts chunks_embeddings (which is the Pinecone vector store)
into a retriever object that can be used by the chain.
`search_kwargs={"k": 2}` tells the retriever to return the top 2 most relevant documents for each query.
`return_source_documents=True:` This tells the chain to include the source documents that were used 
to generate the answer in the output. This can be useful for understanding the reasoning behind the model's answer.

`chain_type_kwargs=chain_type_kwargs:` This provides any additional keyword arguments specific to the chain type. 
In this case, we're passing in the chain_type_kwargs dictionary that was defined earlier.
It contains the prompt template to guide the LLM's response.

7. Defines a route for the web:
```
@app.route("/")
def index():
    return render_template('chat.html')
```

`@app.route("/")`: This is a decorator. It tells Flask to associate the following function (index) with the root URL ("/") of our website.
This means when a user visits the main URL of your app, this function will be executed.

`def index():`: This defines a function named "index". This function will be called when the root URL is accessed.

`return render_template('chat.html')`: This is the core of what the function does.
It uses Flask's render_template function to find an HTML file named chat.html and 
sends its content back to the user's web browser to be displayed.

8. Handling User Input and Generating Responses:
```
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result=qa.invoke({"query": input})
    return str(result["result"])
```

`@app.route("/get", methods=["GET", "POST"])`:
This decorator registers the function chat to handle requests sent to the /get URL.
It specifies that this route accepts both GET and POST requests. 
This means it can handle data submitted through a form (POST) or accessed directly through the URL (GET).

`def chat():`:
This defines the function named chat that will be executed when a request is made to the /get route.

`msg = request.form["msg"]`:
This line retrieves the user's input from the request.
request.form is a dictionary-like object that contains data submitted through an HTML form.
"msg" is the name of the form field where the user's input is expected.

`input = msg:`
This line assigns the retrieved message to a variable named input.

`result = qa.invoke({"query": input})`:
This is the core of the logic.
It calls the invoke method of your question-answering system (qa)
with the user's input as the query.
The result of the question-answering system is stored in the result variable.

`return str(result["result"]):`
This line sends the response back to the user's web browser.
str(result["result"]) converts the response to a string before sending it.

9. Starting the Flask Application:
```
if __name__ == '__main__':
   app.run(host="0.0.0.0", port= 8080)
```
This is a conditional statement commonly used in Python scripts. 
It checks if the script is being run directly (as opposed to being imported as a module). 
If the script is being run directly, the code within this block will be executed.
This code block starts your Flask application and makes it accessible to other devices on your network.

10. Run app.py:
```
python app.py
```

11. Open-up port:
```
http://127.0.0.1:8080
```
