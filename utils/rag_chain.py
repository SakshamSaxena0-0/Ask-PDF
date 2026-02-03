# utils/rag_chain.py
import os
from dotenv import load_dotenv
from typing import List
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

# Load environment variables (ensure GEMINI_API_KEY is set)
load_dotenv()

class RAGChatbot:
    def __init__(self, model_name: str = os.getenv("GEMINI_MODEL", "gemini-pro")):
        # Initialize conversational memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Initialize Google Gemini chat model via LangChain
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.retriever = None
        self.qa_chain = None

    def load_document(self, text: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        embed_model = HuggingFaceEmbeddings()
        vect = FAISS.from_texts(chunks, embed_model)
        self.retriever = vect.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False
        )

    def query(self, question: str) -> str:
        if not self.qa_chain:
            return "Please upload and process a document first."
        return self.qa_chain.run(question)
