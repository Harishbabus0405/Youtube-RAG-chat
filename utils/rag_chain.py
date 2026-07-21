from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_answer(query, retrieved_docs):
    """
    Generate answer using retrieved transcript chunks.
    """

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful AI assistant.

Answer ONLY from the provided transcript context.

If the answer is not available in the context, say:

"I could not find that information in the video transcript."

Keep answers concise and accurate.
"""
            },
            {
                "role": "user",
                "content": f"""
Transcript Context:

{context}

Question:
{query}

Answer:
"""
            }
        ],
        temperature=0.3,
        max_completion_tokens=500
    )

    return response.choices[0].message.content
