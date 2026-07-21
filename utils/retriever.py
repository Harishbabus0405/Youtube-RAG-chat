from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    """
    Create FAISS vector store.
    """

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store


def save_vector_store(vector_store, path="vectorstore/faiss_index"):
    """
    Save FAISS index locally.
    """

    vector_store.save_local(path)


def load_vector_store(
    embeddings,
    path="vectorstore/faiss_index"
):
    """
    Load FAISS index from disk.
    """

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_chunks(
    vector_store,
    query,
    k=5
):
    """
    Retrieve top-k chunks.
    """

    return vector_store.similarity_search(
        query=query,
        k=k
    )
