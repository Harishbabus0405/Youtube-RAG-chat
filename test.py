from utils.youtube_loader import (
    extract_video_id,
    get_transcript
)

from utils.chunker import create_chunks
from utils.embeddings import load_embedding_model

from utils.retriever import (
    create_vector_store,
    retrieve_chunks
)

from utils.rag_chain import generate_answer


url = input("Enter YouTube URL: ")

video_id = extract_video_id(url)

print(f"\nVideo ID: {video_id}")

transcript = get_transcript(video_id)

chunks = create_chunks(transcript)

print(f"\nTotal Chunks: {len(chunks)}")

print("\nLoading Embedding Model...")

embeddings = load_embedding_model()

print("Embedding Model Loaded!")

print("\nCreating FAISS Vector Store...")

vector_store = create_vector_store(
    chunks,
    embeddings
)

print("FAISS Vector Store Created!")

query = input("\nAsk a Question: ")

retrieved_docs = retrieve_chunks(
    vector_store,
    query,
    k=5
)

print("\nGenerating Answer...\n")

answer = generate_answer(
    query,
    retrieved_docs
)

print("=" * 80)
print("\nANSWER:\n")
print(answer)
print("\n" + "=" * 80)