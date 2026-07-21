import re
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL.
    """

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"

    match = re.search(pattern, url)

    if not match:
        raise ValueError("Invalid YouTube URL")

    return match.group(1)


def get_transcript(video_id: str) -> str:
    """
    Fetch transcript in English or Hindi.
    """

    try:

        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en", "hi"]
        )

        transcript_text = " ".join(
            item.text for item in transcript
        )

        return transcript_text

    except Exception:

        raise Exception(
            "Unable to fetch transcript. YouTube may be blocking transcript access from the server, or captions are unavailable for this video."
        )
