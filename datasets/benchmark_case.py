"""Fictional, reproducible incident-retrieval benchmark cohort.

The incident truth is shipped separately in data/benchmark_truth.json and is NEVER
sent to Neo4j. `BENCH-` payments are training inputs without incident labels.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

CASE_CONFIG = (
    ("switch_cascade", ("91", "96")),
    ("authentication_disruption", ("3DS_TIMEOUT", "05")),
    ("capture_delay", ("CAPTURE_FAILED", "96")),
    ("risk_rule_misconfiguration", ("FRAUD_VELOCITY", "05")),
)


def generate_benchmark_payments() -> list[dict[str, object]]:
    """32 incident examples + 16 confounders; all payment/card/customer IDs are fictional.

    Each incident spans distinct merchants, acquirers and reason codes. Hard
    negatives are interleaved in the SAME time window and share attributes; a
    single matching reason or acquirer is not a ground-truth incident label.
    """
    start = datetime(2026, 9, 18, 10, tzinfo=timezone(timedelta(hours=-5)))
    output: list[dict[str, object]] = []
    for family, (_incident, reason_codes) in enumerate(CASE_CONFIG):
        for offset in range(12):
            index = family * 12 + offset + 1
            is_control = offset >= 8
            branch = offset % 4
            code = reason_codes[branch % 2]
            # A confounder can have exactly the same observed descriptors as an
            # incident payment. Its independent *simulated intervention* differs.
            merchant = f"B-MRC-{family}-{branch // 2}"
            acquirer = f"B-ACQ-{family}-{branch % 2}"
            output.append(
                {
                    "payment_id": f"BENCH-{index:03d}",
                    "customer_id": f"B-CUST-{index:03d}",
                    "merchant_id": merchant,
                    "merchant_name": f"Comercio demostración {family + 1}.{branch // 2 + 1}",
                    "acquirer_id": acquirer,
                    "acquirer_name": f"Adquirente demostración {family + 1}.{branch % 2 + 1}",
                    "method_id": f"B-CARD-{index:03d}",
                    "brand": ("VISA", "MASTERCARD")[index % 2],
                    "last4": f"{index:04d}",
                    "amount": float(110 + 13 * (offset % 4) + 7 * family),
                    "currency": "PEN",
                    "status": "FAILED" if code in {"96", "3DS_TIMEOUT", "CAPTURE_FAILED"} else "DECLINED",
                    "reason_code": code,
                    "created_at": (start + timedelta(minutes=90 * family + 3 * offset)).isoformat(),
                }
            )
    return output
