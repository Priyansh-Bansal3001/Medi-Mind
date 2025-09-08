import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
import torch

# Load your embeddings (same as used in Colab)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the existing Chroma DB folder
vector_store = Chroma(
    persist_directory="./chroma_db",  # path to your Chroma folder from Colab
    embedding_function=embeddings,
    collection_name="sample"          # use the same collection name as in Colab
)

# Get retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
model_id = "distilgpt2"  # your HF model name

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_id)
from accelerate import init_empty_weights

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=None,                # don’t auto offload
    dtype=torch.float32,# <- correct way now
    low_cpu_mem_usage=True
).to("cpu")


# Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    truncation = True,
    pad_token_id=tokenizer.eos_token_id
)

hf_pipeline = HuggingFacePipeline(pipeline=pipe)

# Now pass into ChatHuggingFace
llm = ChatHuggingFace(llm=hf_pipeline)

qa = RetrievalQA.from_chain_type(
    llm=hf_pipeline,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

def parse(text: str) -> str:
    # HuggingFace models often output: <|user|> ... </s><|assistant|> answer ...
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    # Remove </s> or any trailing tags
    text = text.replace("</s>", "").strip()
    return text


st.title("🩺 Medical Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", "")

if user_input:
    result = qa.invoke({"query": user_input})

    # If result is a dict, get only the answer
    if isinstance(result, dict):
        answer = result.get("result", "")
    else:
        answer = result

    st.write("Bot:", answer)

# Display chat history
for user_msg, bot_msg in st.session_state.history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Bot:** {bot_msg}")
    st.markdown("---")