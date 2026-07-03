from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(transcript: str):
    """
    Split transcript into chunks for embedding.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(transcript)

    return chunks