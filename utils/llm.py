# llm.py

import os
import json
import logging
import asyncio
from collections import defaultdict
from flask.cli import load_dotenv
import openai

load_dotenv()

# Set OpenAI key
openai_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.AsyncOpenAI(api_key=openai_key) # This is used for openai.ChatCompletion.* calls

# --- Normalize response from OpenAI ---
def normalize_llm_response(raw_response: dict) -> dict:
    """
    Normalize the 'answer' key from a stringified JSON to a proper dictionary.
    """
    try:
        answer_str = raw_response.get("answer", "")
        if isinstance(answer_str, str):
            parsed_answer = json.loads(answer_str)
        else:
            parsed_answer = answer_str  # already a dict

        return {
            "answer": parsed_answer,
            "results": raw_response.get("results", [])
        }

    except json.JSONDecodeError as e:
        return {
            "error": "Invalid JSON format in 'answer'",
            "details": str(e),
            "raw": raw_response
        }

# --- Synchronous wrapper for OpenAI async call ---
def query_with_openai_sdk(prompt: str) -> dict:
    """
    Calls OpenAI chat model and returns parsed JSON response.
    Uses asyncio.run to bridge async call from Flask sync route.
    """
    async def _call():
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an HR assistant that answers questions about candidate resumes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content.strip())

        except Exception as e:
            logging.error("Error calling OpenAI LLM", exc_info=True)
            return {"error": str(e)}

    return asyncio.run(_call())

# --- Build context-rich prompt ---
def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    grouped = defaultdict(list)
    for chunk in retrieved_chunks:
        grouped[chunk["source_file"]].append(chunk["text"])

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

Edge Cases:
- Suppose a candidate is from testing background and has written "reactivity" word in their resume, now that should 
not be confused with React.js or React developers.
- Don't count internship or freelance experience.

1. Return your answer in ***valid JSON format*** with two main keys:
   - "summary": a string that begins with "Based on the provided context, ..." if and only if there is at least 1 suitable 
   candidate matching description. If no suitable candidate is found, then this will have value "1". If the query is irrelevant 
   (not related to candidate resume, skills or experience), this will have value "2".
   - "candidate_details": This should be null if summary is either "1" or "2", otherwise a list of the following 
     objects. Each object should have:
     - "candidate_name": "Candidate from file: extract name from text if clearly available or N/A"
     - "file_name": the source file name
     - "details": a bullet-point list (as a string) of relevant skills, experience, and resume highlights.
     - "score_card": null if summary is "1" or "2", otherwise:
        - "experience_score": integer 1-10
        - "loyalty_score": integer 1-10 (based on duration in companies)
        - "reputation_score": integer 1-10 (based on working with FAANG/MNCs)
        - "clarity_score": integer 1-10 (based on clarity of resume content)

2. The format must be **clean JSON** — no extra commentary, no markdown, no backticks, no `\\n`, no slashes. Only valid JSON.

Question:
{question}

Answer:
"""
