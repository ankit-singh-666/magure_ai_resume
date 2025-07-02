from together import Together
from collections import defaultdict
import asyncio

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    grouped = defaultdict(list)
    for chunk in retrieved_chunks:
        grouped[chunk["source_file"]].append(chunk["text"][:300])  # Truncate

    context_blocks = []
    for source_file, chunks in grouped.items():
        combined_text = "\n".join(chunks)
        context_blocks.append(f"Candidate from file: {source_file}\n{combined_text}")

    full_context = "\n\n".join(context_blocks)

    return f"""You are an HR assistant that answers questions based only on resume (CV) information.

Context:
{full_context}

Instructions:
... [your full prompt here]
Question:
{question}

Answer:"""

async def async_query_with_together_sdk(prompt: str) -> str:
    return await asyncio.to_thread(_sync_query, prompt)

def _sync_query(prompt: str) -> str:
    client = Together(api_key=TOGETHER_API_KEY)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are an HR assistant that answers questions about candidate resumes."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
