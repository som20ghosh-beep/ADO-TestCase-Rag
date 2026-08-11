# src/ui/app.py
import os
from datetime import datetime

import chainlit as cl
from chainlit.types import Feedback
from dotenv import load_dotenv
from sqlmodel import Session

from src.chat.pipeline import ChatPipeline
from src.chat.validator import validate_citations
from src.db import engine
from src.llm.factory import get_llm
from src.models import ChatFeedback

load_dotenv()

llm = get_llm(os.getenv("LLM_PROVIDER", "claude"))
pipeline = ChatPipeline(llm=llm)

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Ask me about your test cases. I'll search the ADO index and cite the results.",
    ).send()

@cl.on_message
async def on_message(msg: cl.Message):
    history = cl.user_session.get("history") or []
    question = msg.content

    # Optional: extract inline filters like "feature:Payments <query>"
    filters = _extract_filters(question)
    clean_question = _strip_filter_syntax(question)

    # Streaming answer message
    answer_msg = cl.Message(content="")
    sources_shown = False
    full_answer = ""
    retrieved_ids = []

    for event in pipeline.answer_stream(clean_question, history=history, filters=filters):
        if event["type"] == "sources" and not sources_shown:
            sources = event["data"]
            retrieved_ids = [r["test_case_id"] for r in sources]
            # Show sources as a collapsible element
            elements = [
                cl.Text(
                    name=f"TC-{r['test_case_id']}",
                    content=f"**{r['title']}**\nFeature: {r['feature']}\nScore: {r['score']:.3f}",
                    display="inline",
                )
                for r in sources
            ]
            answer_msg.elements = elements
            sources_shown = True

        elif event["type"] == "text":
            await answer_msg.stream_token(event["data"])
            full_answer += event["data"]

        elif event["type"] == "done":
            validation = validate_citations(full_answer, retrieved_ids)
            if validation["is_grounded"]:
                badge = f"✅ {validation['cited_count']} citations verified"
            else:
                badge = f"⚠️ Warning: {len(validation['hallucinated_ids'])} unverified IDs: {validation['hallucinated_ids']}"
            await answer_msg.stream_token(f"\n\n_{badge}_")

    await answer_msg.send()

    # Update history — keep last 6 turns to avoid context bloat
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("history", history[-12:])

def _extract_filters(text: str) -> dict:
    """Support 'feature:X state:Y actual query' syntax."""
    import re
    filters = {}
    for m in re.finditer(r"(\w+):(\S+)", text):
        filters[m.group(1)] = m.group(2)
    return filters

def _strip_filter_syntax(text: str) -> str:
    import re
    return re.sub(r"\w+:\S+\s*", "", text).strip()

@cl.on_feedback
async def on_feedback(feedback: Feedback):
    # feedback.value == 1 (thumbs up) or 0 (thumbs down)
    # Persist to SQLite
    last_turn = cl.user_session.get("last_turn")
    if not last_turn:
        return
    with Session(engine) as s:
        s.add(ChatFeedback(
            query=last_turn["question"],
            answer=last_turn["answer"],
            retrieved_ids=",".join(map(str, last_turn["retrieved_ids"])),
            cited_ids=",".join(map(str, last_turn["cited_ids"])),
            hallucinated_ids=",".join(map(str, last_turn["hallucinated_ids"])),
            is_grounded=last_turn["is_grounded"],
            feedback="up" if feedback.value == 1 else "down",
            comment=feedback.comment,
            llm_provider=os.getenv("LLM_PROVIDER", "claude"),
            created_at=datetime.utcnow(),
        ))
        s.commit()