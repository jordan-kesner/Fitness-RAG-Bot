import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai

# Set your API key (better to use environment variables)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to save the vector database
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vectordb", "db1")

# Documents directory
DOCUMENTS_DIRECTORY = os.path.join(BASE_DIR, "data", "documents")

# Create directories if they don't exist
for directory in [PERSIST_DIRECTORY, DOCUMENTS_DIRECTORY]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_web_documents(urls):
    """Load documents from web URLs"""
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents

def load_text_documents(directory=DOCUMENTS_DIRECTORY):
    """Load documents from a directory"""
    if not os.path.exists(directory):
        print(f"Warning: Documents directory {directory} does not exist")
        return []
        
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {directory}")
    return documents

def process_documents(documents):
    """Process documents by splitting them into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} text chunks")
    return texts

def create_vector_db(texts, persist_directory=PERSIST_DIRECTORY):
    """Create and persist a vector database from the texts"""
    if not texts:
        print("No texts to process!")
        return None
        
    embeddings = OpenAIEmbeddings()
    
    # Create and persist the database
    print(f"Creating vector database in {persist_directory}...")
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vector database created and persisted successfully")
    return vectordb

def update_vector_database(urls=None, custom_docs=None):
    """Update the vector database with new documents from URLs or custom documents"""
    all_docs = []
    
    if urls:
        web_docs = load_web_documents(urls)
        all_docs.extend(web_docs)
    
    if custom_docs:
        all_docs.extend(custom_docs)
    
    if not all_docs:
        print("No documents to update!")
        return None
    
    processed_texts = process_documents(all_docs)
    vectordb = create_vector_db(processed_texts)
    
    return vectordb

'''if __name__ == "__main__":
    # URLs to load (add your URLs here)
    urls = [
        "https://example.com/page1",
        "https://example.com/page2"
    ]
    
    # Load documents
    print("Loading documents...")
    web_docs = load_web_documents(urls)
    text_docs = load_text_documents()
    all_docs = web_docs + text_docs
    
    if not all_docs:
        print("No documents were loaded! Please check your documents directory and URLs.")
        exit(1)
        
    # Process documents
    processed_texts = process_documents(all_docs)
    
    # Create and save the vector database
    vectordb = create_vector_db(processed_texts)
    
    print(f"Vector database created with {len(processed_texts)} text chunks and saved to {PERSIST_DIRECTORY}")
    print(f"You can now run your Streamlit app which will use this vector store.")'''