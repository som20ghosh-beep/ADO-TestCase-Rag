# src/chat/validator.py
import re

CITATION_PATTERN = re.compile(r"\[TC-(\d+)\]|\bTC-(\d+)\b")

def extract_cited_ids(answer: str) -> set[int]:
    matches = CITATION_PATTERN.findall(answer)
    return {int(a or b) for a, b in matches}

def validate_citations(answer: str, retrieved_ids: list[int]) -> dict:
    cited = extract_cited_ids(answer)
    retrieved_set = set(retrieved_ids)
    hallucinated = cited - retrieved_set

    return {
        "cited_count": len(cited),
        "retrieved_count": len(retrieved_set),
        "cited_ids": sorted(cited),
        "hallucinated_ids": sorted(hallucinated),
        "is_grounded": len(hallucinated) == 0,
    }