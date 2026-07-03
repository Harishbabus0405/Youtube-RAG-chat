from utils.youtube_loader import (
    extract_video_id,
    get_transcript
)

from utils.chunker import create_chunks

from utils.embeddings import (
    load_embedding_model
)

from utils.retriever import (
    create_vector_store,
    save_vector_store,
    load_vector_store
)

url = input("Enter YouTube URL: ")

video_id = extract_video_id(url)

transcript = get_transcript(video_id)

chunks = create_chunks(transcript)

embeddings = load_embedding_model()

print("\nCreating FAISS...")

vector_store = create_vector_store(
    chunks,
    embeddings
)

print("Saving FAISS...")

save_vector_store(vector_store)

print("Loading FAISS...")

loaded_store = load_vector_store(
    embeddings
)

print("\nTotal Documents:")

print(loaded_store.index.ntotal)