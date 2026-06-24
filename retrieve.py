from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


db = Chroma(
    persist_directory="vector_store",
    embedding_function=embedding_model
)

question = "Who founded Hari Chand Anand & Co.?"

results = db.similarity_search_with_score(
    question.lower(),
    k=5
)

for i, (doc, score) in enumerate(results):
    print(f"\n----- Result {i+1} -----")
    print("Score:", score)
    print(doc.page_content[:300])