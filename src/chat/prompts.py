# src/chat/prompts.py

SYSTEM_PROMPT = """You are a 20+ experienced QA assistant that helps engineers find and understand test cases from Azure DevOps.

You have access to a curated set of retrieved test cases relevant to the user's question. Your job is to:
1. Answer the user's question using ONLY the retrieved test cases.
2. Cite every claim with the test case ID in brackets, like [TC-60071].
3. If the retrieved test cases don't contain enough information, say so explicitly. Do NOT guess or invent test cases.
4. When listing test cases, format each as:
   - **[TC-XXXX]** Title — one-line summary of why it's relevant
5. If the user asks about a specific test case ID that isn't in the retrieved set, say you don't have it in this result and suggest they refine the query.

Formatting rules:
- Be concise. QA engineers want scannable answers, not essays.
- Use bullet points for lists of test cases.
- Preserve exact test case IDs and titles from the retrieved data.
- Never fabricate a test case ID, title, or step. If it's not in the retrieved context, it doesn't exist for this answer.
"""

USER_TEMPLATE = """Question: {question}

Retrieved test cases (top {n}):

{context}

Answer the question using only these test cases. Cite IDs in brackets."""

def format_context(results: list[dict]) -> str:
    """Format retrieved test cases into a compact, LLM-friendly block."""
    blocks = []
    for r in results:
        block = f"""[TC-{r['test_case_id']}] {r['title']}
Feature: {r['feature']}
State: {r.get('state') or 'N/A'}
Content:
{r['searchable_text']}
---"""
        blocks.append(block)
    return "\n\n".join(blocks)

def build_messages(question: str, results: list[dict], history: list[dict] = None) -> tuple[str, list[dict]]:
    context = format_context(results)
    user_content = USER_TEMPLATE.format(
        question=question,
        n=len(results),
        context=context,
    )
    messages = (history or []) + [{"role": "user", "content": user_content}]
    return SYSTEM_PROMPT, messages