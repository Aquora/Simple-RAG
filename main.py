from fastapi import FastAPI
import ollama
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

@app.get("/ask")
def ask(question: str):
    
    results = collection.query(
        query_texts= [question],
        n_results= 2,
    )

    context = "\n\n".join(results["documents"][0])

    augmented_prompt = f""" 
        Use the following contexst to answer the question.
        If the context doesn't contain relevant information, say so.

        Context: {context}

        Question: {question}
    """

    response = ollama.chat(
        model= "qwen2.5:0.5b",
        messages = [{"role": "user", "content": augmented_prompt}],
    )

    return{
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }
