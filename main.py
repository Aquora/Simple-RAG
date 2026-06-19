from fastapi import FastAPI
import ollama
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()

client = chromadb.PersistentClient(path = "./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function = ef,
)

class DocumentSumbission(BaseModel):
    userName: str
    content: str

@app.post("/documents")
def addDocument(submission: DocumentSumbission):
    
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]

    collection.add(
        ids = [f"{submission.userName}-chunk{i}" for i in range(len(chunks))],
        documents = chunks,
        metadatas = [
            {"source" : "profile", "userName": submission.userName, "chunk_index": i}
            for i in range(len(chunks))
        ]
    )

    return{

        "message" : f"Added {len(chunks)} chunks for user'{submission.userName}'.",
        "userName": submission.userName,
        "chunks_added": len(chunks),
    }




@app.get("/ask")
def ask(question: str, user: str = None):  

    query_params = {
        "query_texts": [question],
        "n_results": 2,
    }

    if user:
        query_params["where"] = {"user_name": user}  

    results = collection.query(**query_params)  
    context = "\n\n".join(results["documents"][0])

    augmented_prompt = f"""
        Use the following context to answer the question.
        If the context doesn't contain relevant information, say so.

        Context: {context}

        Question: {question}
    """

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
        "filtered_by_user": user,  # Shows which user was filtered (or None for all)
    }

