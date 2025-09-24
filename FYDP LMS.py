#Installation
!pip install langchain
!pip install langchain_community
!pip install langchain_openai
!pip install langchain-google-genai google-genai
!pip install chromadb
!pip install pypdf

#Import
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import os

#Ready Environment Variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyBP0IRO1RV5AboDqSNngmCHLDQTWFHKk1E"

#Get LLM
largeLanguageModelGemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY")   # make sure this is set!
)

#Load the PDF
loader = PyPDFLoader(r"C:\Users\kumai\Documents\UniversityStuff\7 Sem\NIS\LABS\LAB-MANUAL.pdf")
docs = loader.load()

#Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

#Get Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))

#Get Vectorstore
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#Template for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#Conversation Logic & Response Generation
conversation = ConversationalRetrievalChain.from_llm(
    llm=largeLanguageModelGemini,
    retriever=retriever,
    memory=memory
)

resp = conversation({"question": "What is in the lab manual?"})
print(resp["answer"])
