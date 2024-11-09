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

1. Downloading the LLM:

 ```
!wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```

This line uses wget to download the Llama 2 7B Chat GGML model from Hugging Face. 
This is the Large Language Model (LLM) that will be used for answering questions.

2. Setting up Pinecone:
```
os.environ['PINECONE_API_KEY'] = "3be36663-e91a-430b-b0d5-fe557e0a9dde" 
api_key = os.environ.get('PINECONE_API_KEY')
embeddings = download_hf_embeddings()
pinecone.Pinecone(api_key=api_key)
```
These lines set up the Pinecone connection.
First, the Pinecone API key is set as an environment variable and then retrieved.
The download_hf_embeddings() function (which was defined earlier) is called to download and initialize a Hugging Face embeddings model.
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
In this case, it's set to the llm object defined earlier (which is the Llama2 model).

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
In this case, we're passing in the chain_type_kwargs dictionary that you defined earlier.
It contains the prompt template to guide the LLM's response.

7. Running a Query:
```
query = "What is transformer?."
result1 = qa1.invoke({"query":query})
result1["result"]
```
query is assigned the question you want to ask the system.
`qa1.invoke({"query": query})` runs the query through the question-answering chain that we set up previously. 

It uses the invoke method, which executes a step in the chain, passing the query as input.
`result1["result"]` accesses the 'result' key from the output of the query. 

This contains the answer generated by the LLM based on the context retrieved from Pinecone and the prompt template. 
To see the output, you would need to run the code.
