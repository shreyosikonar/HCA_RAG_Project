from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
loader = TextLoader("data/hca_company_knowledge_base.txt")
documents = loader.load()

print(f"Pages Loaded: {len(documents)}")

# Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")

# Display chunks
for i, chunk in enumerate(chunks):
    print(f"\n----- Chunk {i+1} -----")
    print(chunk.page_content[:300])