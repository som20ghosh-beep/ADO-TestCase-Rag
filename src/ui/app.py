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
from src.models import ChatFeedback, TestCase

load_dotenv()

llm = get_llm(os.getenv("LLM_PROVIDER", "claude"))
pipeline = ChatPipeline(llm=llm)

def _clear_action() -> cl.Action:
    return cl.Action(
        name="clear_history",
        payload={},
        label="Clear chat history",
        icon="trash-2",
        tooltip="Clear the conversation history",
    )

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Ask me about your test cases. I'll search the ADO index and cite the results.",
        actions=[_clear_action()],
    ).send()

@cl.action_callback("clear_history")
async def on_clear_history(action: cl.Action):
    cl.user_session.set("history", [])
    for message in cl.chat_context.get():
        await message.remove()
    await cl.Message(
        content="🧹 Conversation history cleared. Ask me anything about your test cases.",
        actions=[_clear_action()],
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
            # Clickable, per-test-case chips — click opens the full details panel
            answer_msg.elements = [_build_testcase_element(r) for r in sources]
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

    answer_msg.actions = [_clear_action()]
    await answer_msg.send()

    # Update history — keep last 6 turns to avoid context bloat
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("history", history[-12:])

def _build_testcase_element(result: dict) -> cl.Text:
    """Build a clickable chip for a search result; click opens full test case details."""
    with Session(engine) as s:
        tc = s.get(TestCase, result["test_case_id"])
    content = (
        _format_testcase_details(tc, result["score"], result.get("why", ""))
        if tc
        else f"**{result['title']}**\nFeature: {result['feature']}\nScore: {result['score']:.3f}"
    )
    return cl.Text(name=f"TC-{result['test_case_id']}", content=content, display="inline")

def _format_testcase_details(tc: TestCase, score: float, why: str) -> str:
    steps = tc.steps_json or []
    if steps:
        steps_text = "\n".join(
            f"{i}. {step.get('action', '')} → {step.get('expected', '')}"
            for i, step in enumerate(steps, start=1)
        )
    else:
        steps_text = "N/A"

    return f"""**{tc.title}**

**Area:** {tc.area_path}
**State:** {tc.state}
**Priority:** {tc.priority if tc.priority is not None else 'N/A'}
**Automation status:** {tc.automation_status or 'N/A'}
**Assigned to:** {tc.assigned_to or 'Unassigned'}
**Score:** {score:.3f}

**Preconditions:**
{tc.preconditions or 'N/A'}

**Steps:**
{steps_text}

**Expected result:**
{tc.expected_result or 'N/A'}

**Why matched:** {why or 'N/A'}
"""

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