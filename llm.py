# llm.py

from together import Together
from collections import defaultdict

def query_with_together_sdk(prompt: str, api_key: str) -> str:
    client = Together(api_key=api_key)

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {
                "role": "system",
                "content": "You are an HR assistant that answers questions about candidate resumes."
            },
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    # Group chunks by source file (e.g., the CV PDF)
    grouped = defaultdict(list)
    for chunk in retrieved_chunks:
        grouped[chunk["source_file"]].append(chunk["text"])
    
    # Format grouped chunks into contextual blocks per candidate
    context_blocks = []
    for source_file, chunks in grouped.items():
        combined_text = "\n".join(chunks)
        context_blocks.append(f"Candidate from file: {source_file}\n{combined_text}")

    full_context = "\n\n".join(context_blocks)

    return f"""You are an HR assistant that answers questions based only on resume (CV) information.

Context:
{full_context}

Instructions:
- Use only the provided context to answer the question.
- If the question is unrelated to candidate skills, experience, education, or resumes in general, respond with:
  "I'm sorry, I can only answer questions related to candidate resumes and qualifications."
- If no relevant candidates are found, respond with:
  "No suitable candidates found."

If the question is relevant:
1. Return your answer in **valid JSON format** with two main keys:
   - "summary": a string that **begins** with "Based on the provided context, ..." and briefly summarizes the findings.
   - "candidate_details": a list of up to 3 candidate objects. Each object should have:
     - "candidate_name": "Candidate from file: [filename]" (or extract name from text if clearly available)
     - "file_name": the source file name
     - "details": a bullet-point list (as a string) of relevant skills, experience, and resume highlights.

2. The format must be clean JSON — no extra commentary or explanation outside the JSON object.

Question:
{question}

Answer:
"""
