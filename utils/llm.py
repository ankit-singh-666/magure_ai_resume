from together import Together
from typing import Dict, List

# --- REVERTED TO SINGLE-AGENT PROMPTS ---
# A router will select ONE of these prompts to use for the single LLM call.
AGENT_SYSTEM_PROMPTS = {
    "skill_analyzer": "You are an AI assistant specializing in matching technical skills. Your primary goal is to identify candidates with the skills mentioned in the user's query.",
    "experience_analyzer": "You are an AI assistant specializing in analyzing work experience. Your primary goal is to identify candidates with the relevant job titles and years of experience.",
    "relevancy_scorer": "You are an AI assistant that scores and ranks candidates. Your primary goal is to provide a ranked list of the best-matching candidates with a score and justification.",
    "seniority_detector": "You are an AI assistant that detects job seniority. Your primary goal is to classify candidates' seniority (Junior, Mid-level, Senior) and explain why.",
    "general_analyzer": "You are a world-class HR AI Assistant. Your primary goal is to synthesize all provided information to find the best candidates that match the user's query and explain your choices."
}

# --- NEW: PROMPT BUILDER FOR THE ROUTER ARCHITECTURE ---
# This function combines the router's choice with your detailed JSON instructions.
def build_prompt_with_router(question: str, retrieved_chunks: List[dict], agent_role: str) -> str:
    """
    Constructs the final prompt by combining the retrieved context with a
    dynamically selected agent role.

    Args:
        question: The original user query.
        retrieved_chunks: A list of chunk dictionaries from the vector store.
        agent_role: The role selected by the query router (e.g., "skill_analyzer").

    Returns:
        The fully formatted prompt string for the final LLM call.
    """
    
    # Format the retrieved chunks into a single context block
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(f"--- Candidate from file: {chunk['source_file']} ---\n{chunk['text']}")
    full_context = "\n\n".join(context_blocks)

    # This is your detailed, JSON-formatting prompt, now guided by the selected agent's role.
    return f"""You are a world-class HR assistant AI. Your assigned role for this specific query is: **{agent_role}**.
You will answer questions based ONLY on the resume context provided below.

Context:
{full_context}

Instructions:
- Your main goal is to fulfill your assigned role while adhering strictly to the JSON output format.
- If the question is unrelated to resumes, your entire response must be a JSON object with a "summary" key set to "2".
- If no relevant candidates are found, your entire response must be a JSON object with a "summary" key set to "1".

Edge Cases & Rules:
- A word like "reactivity" should NOT be confused with the "React" framework.
- Do not count internship or freelance experience when evaluating scores.

If the question is relevant and suitable candidates are found:
1.  Your entire response MUST be a single, valid JSON object. Do not add any text or comments outside of the JSON structure.
2.  The JSON object must have three main keys: "summary", "candidate_details", and "score_card".

JSON Schema Definition:
{{
  "summary": "A string that MUST begin with 'Based on the provided context, ...'. If no suitable candidate is found, this value must be the string '1'. If the query is irrelevant, it must be the string '2'.",
  "candidate_details": "This must be null if summary is '1' or '2'. Otherwise, it is a list of up to 3 candidate objects. Each object must have: 'candidate_name' (string), 'file_name' (string, the source file name from the context), and 'details' (string, a bullet-point list of relevant skills and experience).",
  "score_card": "This must be null if summary is '1' or '2'. Otherwise, it is a list of scoring objects, one for each candidate in 'candidate_details'. Each object must have: 'candidate_name' (string), 'experience_score' (integer, 1-10), 'loyalty_score' (integer, 1-10), 'reputation_score' (integer, 1-10), and 'clarity_score' (integer, 1-10)."
}}

Original Question:
{question}

Answer (must be a single valid JSON object):
"""

# --- API QUERY FUNCTION (Unchanged) ---
def query_with_together_sdk(prompt: str, api_key: str, system_message: str = "You are a helpful AI assistant.") -> str:
    if not api_key:
        raise ValueError("TOGETHER_API_KEY is not set.")
    client = Together(api_key=api_key)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", # Using a known available model
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}, 
    )
    return response.choices[0].message.content.strip()