# src/embedding/text_builder.py
import hashlib

from src.models import TestCase


def build_searchable_text(tc: TestCase) -> str:
    """Compose the text that gets embedded and reranked in later phases."""
    parts = [f"Title: {tc.title}"]

    if tc.area_path:
        # Take the leaf feature name — 'GEM\Guru.com\Invite and Incentives' → 'Invite and Incentives'
        feature = tc.area_path.split("\\")[-1]
        parts.append(f"Feature: {feature}")

    if tc.state:
        parts.append(f"State: {tc.state}")

    if tc.preconditions:
        parts.append(f"Preconditions: {tc.preconditions}")

    if tc.steps_json:
        step_lines = [
            f"Step {s['step']}: {s['action']} → Expected: {s['expected']}"
            for s in tc.steps_json
        ]
        parts.append("Steps:\n" + "\n".join(step_lines))

    if tc.expected_result:
        parts.append(f"Overall Expected: {tc.expected_result}")

    return "\n\n".join(parts)


def compute_content_hash(searchable_text: str) -> str:
    return hashlib.sha256(searchable_text.encode("utf-8")).hexdigest()