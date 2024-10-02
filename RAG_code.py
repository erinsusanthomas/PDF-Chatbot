from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

ollama_model_type = "stablelm-zephyr" # reference:https://ollama.com/library/stablelm-zephyr
ollama_embedding = 'nomic-embed-text' # reference: https://ollama.com/library/nomic-embed-text

def load_documents(data):
    document_loader = PyPDFDirectoryLoader(data) 
    print("Loading Pdf documents... ✅")
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, 
                                                   length_function=len, is_separator_regex=False)
    print("Splitting documents... ✅")
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        last_page_id = current_page_id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
    return chunks

def get_embedding_function():
    embeddings = OllamaEmbeddings(model=ollama_embedding)
    return embeddings

index = faiss.IndexFlatL2(1024)
db = FAISS(
    embedding_function=get_embedding_function(),
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)

def add_to_chroma(store_dir, chunks: list[Document]):
    # db = Chroma(persist_directory = store_dir, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    print(f"No. of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
        print(f"All new documents added ✅")
    else:
        print("No new documents to add ✅")

PROMPT_TEMPLATE = """
Answer the question based only on the following context: {context}
---
Answer the question based on the above context: {question}
"""

def get_response(store_dir, query_text):
    # db = Chroma(persist_directory=store_dir, embedding_function=get_embedding_function())
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Prompt created ✅")

    model = OllamaLLM(model = ollama_model_type)
    response_text = model.invoke(prompt)
    print("Response retrieved ✅")
    return response_text