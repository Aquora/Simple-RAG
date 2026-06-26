import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import openai
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import shutil

load_dotenv()

DATA_PATH = "data"
CHROMA = "chroma"

embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents

def splitText(documents):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True,
    )
    chunks = text_spliter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def persistDB(chunks):

    db = Chroma(
        embedding_function= embeddings, persist_directory = CHROMA
    )

    chunkID = calculateChunkID(chunks)

    existing_items = db.get(include=[])  
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunkID:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculateChunkID(chunks):

    lastPageID = None
    currentChunkIndex = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")

        current_page_id = f"{source}:{page}"

        if current_page_id == lastPageID:
            currentChunkIndex += 1
        else:
            currentChunkIndex = 0

        chunk_id = f"{current_page_id}:{currentChunkIndex}"
        lastPageID = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clearDB():
    if os.path.exists(CHROMA):
        shutil.rmtree(CHROMA)


def main():
    documents = load_documents()
    chunks = splitText(documents)
    persistDB(chunks)


if __name__ == "__main__":
    main()
