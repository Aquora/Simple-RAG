import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama 

from fastapi import FastAPI


CHROMA = "chroma"
app = FastAPI()

load_dotenv()


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

@app.get("/query")
def query(query_text: str):

    embedding_function = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

    db = Chroma(persist_directory=CHROMA, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text} \n\nSources: {sources}"
    return {
        "question": query_text,
        "response": formatted_response,
    }

