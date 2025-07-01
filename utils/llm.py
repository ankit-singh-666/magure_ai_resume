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


Edge Cases:
-Suppose a candidate is from testing background and has written "reactivity" word in their resume , now that should 
not be confused with react js or react developers
-Don't count internship or freelance experience. 

If the question is relevant:
1. Return your answer in valid JSON format with three main keys:
   - "summary": a string that begins with "Based on the provided context, ..." if and only if there is atleast 1 suitable 
   candidate matching description and then briefly summarizes the findings. 
   If no suitable candidate found, then this will have "1" as value, if the query is irrelevant and not related to 
   candidate resume, skills or experience, this will have "2" as value
   - "candidate_details":This should be null if summary is either "1" or "2", otherwise,  a list of up to 3 candidate 
        objects. Each object should have:
     - "candidate_name": "Candidate from file: [filename]" (or extract name from text if clearly available)
     - "file_name": the source file name
     - "details": a bullet-point list (as a string) of relevant skills, experience, and resume highlights.
   - "score_card":This should be null if summary is either "1" or "2", otherwise, 
     scoring of resume on different parameters 
        -"experience_score": scoring based on total no. of experience. 1-2 years 
        -"loyality_score": longevity in a company, how long they have serverd. 2 to 3 years is good but more than that is great
        -"reputation_score":worked with reputed companies like FAANG or MNCs 
        -"clarity_score": score based on clarity in resumes , they should not use obscure words 

2. The format must be clean JSON — no extra commentary or explanation outside the JSON object.


Question:
{question}

Answer:
"""
