from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_summary(transcript):
    """
    Generate video summary.
    """

    transcript = transcript[:12000]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
You are an expert YouTube video summarizer.

Create:

1. Overview
2. Key Points
3. Main Takeaways

Use bullet points where appropriate.
"""
            },
            {
                "role": "user",
                "content": transcript
            }
        ],
        temperature=0.3,
        max_completion_tokens=800
    )

    return response.choices[0].message.content