# Multilingual-Retrieval-Augmented-Generation-RAG-System
#Simple Multilingual Retrieval-Augmented Generation (RAG) System, capable of understanding and responding to both English and Bengali queries.

#setup guide:
Environment: Google Colab Pro
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
"অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
অনুপমের ভাষায় সুপুরুষকে বলা হয়েছে সেই ব্যক্তি, যিনি শারীরিক সৌন্দর্য এবং নৈতিক গুণাবলীতে উজ্জ্বল।
Groundedness: 0.85199434
Relevance: 0.7941223

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
  "groundedness": 0.851,
  "relevance": 0.794
 }

#Evaluation Metrics
Groundness- Cosine similarity between generated answer and context
Relevance- Cosine similarity between query and retrieved chunks


---------------Q&A-------------------------
q1.	What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
ans: I used PyPDFLoader from LangChain. This supports multi-page PDF loading and works well with Bangla text.Yes, I faced some formatting issues such as extra whitespace and special characters,specially OCR distortion.
Query: অনুপমের বড় ভাইয়ের নাম কী ছিল?
Ans:"প্রশ্নের দেওয়া প্রসঙ্গে অনুপমের বড় ভাইয়ের নাম উল্লেখ করা হয়নি।"
Which is technically incorrect because the information is present in pdf but was not retrieved properly due to distorted character encoding.

q2.	What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?
ans: I used RecursiveCharacterTextSplitter with a 500-character chunk size and 50-character overlap. This balances semantic coherence and retrievability, ensuring complete context in each chunk while being small enough for FAISS vector indexing.

q3.	What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?
ans: I used sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 because it supports over 50 languages including Bangla, and it performs well for semantic similarity tasks while being fast enough for real-time querying.

q4.	How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
ans: Each query is embedded using the same multilingual embedding model, and compared to stored chunk embeddings using FAISS with L2 similarity (internally optimized). Cosine similarity is used for evaluation.

q5.How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?
ans: All embeddings are in the same multilingual vector space. Chunk overlap helps preserve full thoughts. If a query is vague, the model might retrieve general chunks; a potential improvement is query rephrasing or adding query classification/reranking.

q6.	Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?
ans: Yes, most test cases with high groundedness (>0.85). Relevance could be improved by:
-Better chunking (sentence or paragraph-aware)
-Using a larger or fine-tuned embedding model
-Expanding the PDF corpus  
