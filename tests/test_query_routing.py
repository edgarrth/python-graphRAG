from pe.axiz.graphrag_payments.application.query_routing import extract_explicit_reason_codes


def test_extracts_explicit_numeric_reason_code() -> None:
    assert extract_explicit_reason_codes("¿Para qué es el código 91?") == ["91"]
    assert extract_explicit_reason_codes("Explica el codigo 05") == ["05"]
    assert extract_explicit_reason_codes("Explica el código de rechazo 51") == ["51"]


def test_does_not_treat_arbitrary_numbers_as_reason_codes() -> None:
    assert extract_explicit_reason_codes("Tengo un pago de 91 PEN y el PAY-1001 falló") == []


def test_deduplicates_explicit_reason_codes() -> None:
    assert extract_explicit_reason_codes("código 91 vs código 91") == ["91"]
