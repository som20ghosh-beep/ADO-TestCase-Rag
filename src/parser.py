# src/parser.py
from lxml import etree, html
import re

def parse_test_steps(steps_xml: str | None) -> list[dict]:
    if not steps_xml:
        return []
    try:
        root = etree.fromstring(steps_xml.encode('utf-8'))
    except etree.XMLSyntaxError:
        return []

    parsed = []
    for step in root.findall('.//step'):
        strings = step.findall('parameterizedString')
        action = _clean_html(strings[0].text) if len(strings) > 0 else ""
        expected = _clean_html(strings[1].text) if len(strings) > 1 else ""
        parsed.append({
            "step": int(step.get('id', 0)),
            "action": action,
            "expected": expected,
        })
    return parsed

def _clean_html(text: str | None) -> str:
    if not text:
        return ""
    # Strip HTML tags, decode entities, normalize whitespace
    stripped = html.fromstring(text).text_content()
    return re.sub(r'\s+', ' ', stripped).strip()