"""Stream Sync — Synchronous streaming with stream().

Demonstrates:
- agent.stream(input) for token-by-token output
- Collecting chunks into full text
- StreamChunk properties (index, text, is_final)

Run: python -m examples.08_streaming.stream_sync
Visit: http://localhost:8000/playground
Requires: uv pip install syrin[serve]
"""

from pathlib import Path

from dotenv import load_dotenv

from examples.models.models import almock
from syrin import Agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class StreamAgent(Agent):
    name = "stream-agent"
    description = "Streams token-by-token output"
    model = almock
    system_prompt = "You are a helpful assistant."


if __name__ == "__main__":
    agent = StreamAgent()
    chunks = list(agent.stream("Tell me a short story"))
    full_text = "".join(c.text for c in chunks)
    print(full_text[:200] + "...")
    print(f"Total chunks: {len(chunks)}")
    print("Serving at http://localhost:8000/playground")
    agent.serve(port=8000, enable_playground=True, debug=True)
