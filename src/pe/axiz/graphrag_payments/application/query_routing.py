from __future__ import annotations

import re


EXPLICIT_REASON_CODE_PATTERN = re.compile(
    r"\b(?:c[oó]digo|code)(?:\s+de\s+(?:rechazo|respuesta|error))?\s*[:#-]?\s*([A-Za-z0-9_]{2,40})\b",
    flags=re.IGNORECASE,
)


def extract_explicit_reason_codes(question: str) -> list[str]:
    """Extract reason-code identifiers only when explicitly labelled in the query.

    Requiring words such as ``código`` / ``code`` prevents amounts, dates and
    payment identifiers from being misinterpreted as payment reason codes.
    """

    found: list[str] = []
    for match in EXPLICIT_REASON_CODE_PATTERN.finditer(question):
        code = match.group(1).strip().upper()
        if code and code not in found:
            found.append(code)
    return found
