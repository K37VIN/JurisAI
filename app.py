import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.embeddings import Embeddings
from typing import List, Tuple
import sys
import logging

# Configure logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

# Initialize session state
if 'chain' not in st.session_state:
    st.session_state.chain = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize LLM
@st.cache_resource
def initialize_llm():
    try:
        llm_pipeline = pipeline(
            "text-generation",
            model="distilgpt2",
            max_new_tokens=512,
            pad_token_id=50256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1
        )
        return HuggingFacePipeline(
            pipeline=llm_pipeline,
            model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Initialize FAISS and create chain
@st.cache_resource
def initialize_chain():
    try:
        embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        db = FAISS.load_local(
            "my_vector_store",
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        llm = initialize_llm()
        if not llm:
            return None
            
        prompt = PromptTemplate(
            template="""
            <s>[INST]You are a legal assistant specializing in Indian law. Based on the following information, provide a clear and concise response:

            CONTEXT: {context}
            PREVIOUS CONVERSATION: {chat_history}
            CURRENT QUESTION: {question}

            Please provide your response:[/INST]
            """,
            input_variables=["context", "chat_history", "question"]
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )
        
        return chain
    except Exception as e:
        st.error(f"Error initializing chain: {str(e)}")
        return None

def get_conversation_history() -> List[Tuple[str, str]]:
    """Safely create conversation history pairs."""
    history = []
    messages = st.session_state.get('messages', [])
    
    # Create pairs of messages, skipping incomplete pairs
    for i in range(0, len(messages)-1, 2):
        if i+1 < len(messages):
            user_msg = messages[i]
            assistant_msg = messages[i+1]
            
            # Verify message roles and contents
            if (user_msg.get("role") == "user" and 
                assistant_msg.get("role") == "assistant" and 
                "content" in user_msg and 
                "content" in assistant_msg):
                history.append((
                    str(user_msg["content"]),
                    str(assistant_msg["content"])
                ))
    
    return history

def process_response(response_text: str) -> str:
    """Clean up the response text."""
    if not response_text:
        return ""
    
    cleaned = response_text.replace("<s>", "").replace("</s>", "")
    cleaned = cleaned.replace("[INST]", "").replace("[/INST]", "")
    cleaned = cleaned.strip()
    
    return cleaned

def generate_response(question: str) -> str:
    """Generate a response using the chain."""
    try:
        if not st.session_state.chain:
            st.session_state.chain = initialize_chain()
            
        if not st.session_state.chain:
            return "I apologize, but I'm having trouble initializing. Please try again."
            
        # Get conversation history
        history = get_conversation_history()
        
        # Generate response
        response = st.session_state.chain.invoke({
            "question": question,
            "chat_history": history
        })
        
        return process_response(response["answer"])
        
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

# UI Setup
st.title("Juris AI")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about Indian law..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            
            if response:
                # Display with typing effect
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.01)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Store assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

# Reset button
if st.button("Reset Chat 🗑"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()