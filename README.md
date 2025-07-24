# Multilingual-Retrieval-Augmented-Generation-RAG-System
#Simple Multilingual Retrieval-Augmented Generation (RAG) System, capable of understanding and responding to both English and Bengali queries.

#setup guide:
Environment: Google Colab
#mount drive (only if you want to access files from drive in colab)
from google.colab import drive
drive.mount('/content/drive')

#install dependencies:
!pip install langchain faiss-cpu sentence-transformers pypdf fastapi uvicorn openai scikit-learn
!pip install -U langchain-community

#OpenAIkey
 from getpass import getpass
 import os
 os.environ["OPEN_API_KEY"] = getpass("Enter your openAI API key: ")

 #Tools,Libraries & Packages:
 Langchain-document loading, chunking
 FAISS-efficient vector similarity search
 SentenceTransformers-multilingual embeddings
 FastAPI-Rest API for interaction
 OpenAI API- LLM for answer generation
 Scikit-learn-cosine similarity for evaluation
 PyPDF-Bangla PDF parsing

 #Sample Que & OUTPUTs


 #API documentation
 Request Body:
 { "question": "<your query>"
 }
 Response:
 {
  "question": "...",
  "answer": "...",
  "retrieved_chunks": ["...", "..."],
  "scores": [0.01, 0.03, 0.07],
  "groundedness": 0.866,
  "relevance": 0.801
 }

 #Evaluation Metrics
Groundness- Cosine similarity between generated answer and context
Relevance- Cosine similarity between query and retrieved chunks
