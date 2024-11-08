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