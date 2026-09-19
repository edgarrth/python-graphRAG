from __future__ import annotations

from functools import lru_cache

from neo4j import Driver, GraphDatabase, RoutingControl

from pe.axiz.graphrag_payments.settings import get_settings


@lru_cache
def get_driver() -> Driver:
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    driver.verify_connectivity()
    return driver


def check_connectivity(driver: Driver) -> None:
    settings = get_settings()
    driver.execute_query(
        "RETURN 1 AS ok",
        database_=settings.neo4j_database,
        routing_=RoutingControl.READ,
    )
