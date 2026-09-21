"""Reproducible, fictional payment examples for the GraphSAGE demo.

These are not real incidents or labels for a validated classifier. The additional
connectivity simply makes the structural-embedding exercise more illustrative.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone


SCENARIOS = (
    ("APPROVED", None),
    ("APPROVED", None),
    ("DECLINED", "05"),
    ("DECLINED", "51"),
    ("DECLINED", "91"),
    ("DECLINED", "91"),
    ("FAILED", "96"),
    ("FAILED", "3DS_TIMEOUT"),
    ("DECLINED", "FRAUD_VELOCITY"),
    ("FAILED", "CAPTURE_FAILED"),
)
MERCHANTS = (
    ("MRC-TECH", "TechStore Perú"),
    ("MRC-TRAVEL", "TravelNow"),
    ("MRC-GROCERY", "Mercado Uno"),
    ("MRC-STREAM", "StreamPlay"),
    ("MRC-GAMING", "GameBox"),
    ("MRC-FOOD", "Comidas Express"),
    ("MRC-PARK", "ParkNow"),
)
ACQUIRERS = (
    ("ACQ-ANDES", "Andes Acquiring"),
    ("ACQ-LIMA", "Lima Payments"),
    ("ACQ-SUR", "Sur Payment Network"),
)


def generate_synthetic_payments(count: int, seed: int = 42) -> list[dict[str, object]]:
    """Generate deterministic test entities; no personal/card data from users."""
    if count < 0 or count > 5000:
        raise ValueError("SYNTHETIC_PAYMENT_COUNT debe estar entre 0 y 5000")
    rng = random.Random(seed)
    start = datetime(2026, 9, 18, 8, tzinfo=timezone(timedelta(hours=-5)))
    rows: list[dict[str, object]] = []
    for index in range(1, count + 1):
        status, reason = SCENARIOS[rng.randrange(len(SCENARIOS))]
        merchant_id, merchant_name = MERCHANTS[rng.randrange(len(MERCHANTS))]
        acquirer_id, acquirer_name = ACQUIRERS[rng.randrange(len(ACQUIRERS))]
        # Fictional outage pattern: two merchants share this acquirer during a window.
        if index % 11 in (0, 1, 2):
            status, reason = "DECLINED", "91"
            acquirer_id, acquirer_name = "ACQ-LIMA", "Lima Payments"
        rows.append(
            {
                "payment_id": f"SYN-{index:05d}",
                "customer_id": f"SYN-CUST-{index % 79:03d}",
                "merchant_id": merchant_id,
                "merchant_name": merchant_name,
                "acquirer_id": acquirer_id,
                "acquirer_name": acquirer_name,
                "method_id": f"SYN-CARD-{index % 113:03d}",
                "brand": ("VISA", "MASTERCARD")[index % 2],
                "last4": f"{index % 10000:04d}",
                "amount": round(rng.uniform(12, 2600), 2),
                "currency": "PEN",
                "status": status,
                "reason_code": reason,
                "created_at": (start + timedelta(minutes=index * 2)).isoformat(),
            }
        )
    return rows
