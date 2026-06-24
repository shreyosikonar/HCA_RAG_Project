from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load document
loader = TextLoader("data/hca_company_knowledge_base.txt")
documents = loader.load()

# Create chunks
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"],
    chunk_size=1500,
    chunk_overlap=200
)


chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# Create ChromaDB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vector_store"
)
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print("=" * 50)
    print(chunk.page_content)
    print("=" * 50)
print("Embeddings stored successfully!")