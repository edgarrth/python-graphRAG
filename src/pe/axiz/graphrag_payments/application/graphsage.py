"""Train GraphSAGE on the payment topology using Neo4j GDS Community.

Embeddings measure structural similarity, *not* fraud, causal attribution or risk.
No customer identifiers, card digits or raw text are used as model features.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from neo4j import Driver
from neo4j.exceptions import Neo4jError

from pe.axiz.graphrag_payments.domain.models import (
    GraphSageNeighbor,
    GraphSageSimilarResponse,
    GraphSageStatus,
    GraphSageTrainRequest,
    GraphSageTrainResponse,
)
from pe.axiz.graphrag_payments.settings import Settings

# Payment, Merchant, Acquirer and ReasonCode each get a SAME-LENGTH float vector.
# One-hot node type (0..3); amount; approved; failure; availability; 3DS; fraud.
FEATURE_STATEMENTS = (
    """MATCH (n:Payment) SET n.sage_features = [
      1.0, 0.0, 0.0, 0.0,
      CASE WHEN n.amount IS NULL THEN 0.0 ELSE toFloat(n.amount) / 10000.0 END,
      CASE WHEN n.status = 'APPROVED' THEN 1.0 ELSE 0.0 END,
      CASE WHEN n.status IN ['DECLINED', 'FAILED'] THEN 1.0 ELSE 0.0 END,
      0.0, 0.0, 0.0
    ]""",
    """MATCH (n:Merchant) SET n.sage_features = [
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]""",
    """MATCH (n:Acquirer) SET n.sage_features = [
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]""",
    """MATCH (n:ReasonCode) SET n.sage_features = [
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      CASE WHEN n.code IN ['91', '96'] THEN 1.0 ELSE 0.0 END,
      CASE WHEN n.code = '3DS_TIMEOUT' THEN 1.0 ELSE 0.0 END,
      CASE WHEN n.code = 'FRAUD_VELOCITY' THEN 1.0 ELSE 0.0 END
    ]""",
)

# Undirected so messages reach payments FROM linked merchant/acquirer/reason nodes.
PROJECT = """
CALL gds.graph.project(
    $graph_name,
    {Payment: {properties: ['sage_features']},
     Merchant: {properties: ['sage_features']},
     Acquirer: {properties: ['sage_features']},
     ReasonCode: {properties: ['sage_features']}},
    {AT_MERCHANT: {orientation: 'UNDIRECTED'},
     ROUTED_TO: {orientation: 'UNDIRECTED'},
     FAILED_WITH: {orientation: 'UNDIRECTED'}}
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
"""
TRAIN = """
CALL gds.beta.graphSage.train($graph_name, {
  modelName: $model_name,
  featureProperties: ['sage_features'],
  embeddingDimension: $dimension,
  sampleSizes: $sample_sizes,
  aggregator: 'mean',
  activationFunction: 'sigmoid',
  batchSize: 32,
  epochs: $epochs,
  maxIterations: 10,
  randomSeed: $seed,
  concurrency: 2
}) YIELD modelInfo, trainMillis
RETURN modelInfo.metrics AS metrics, trainMillis
"""
WRITE = """
CALL gds.beta.graphSage.write($graph_name, {
  modelName: $model_name,
  writeProperty: 'sage_embedding',
  nodeLabels: ['Payment'],
  concurrency: 2
}) YIELD nodePropertiesWritten
RETURN nodePropertiesWritten
"""

SIMILAR = """
MATCH (source:Payment {payment_id: $payment_id})
WHERE source.sage_embedding IS NOT NULL
MATCH (candidate:Payment)
WHERE candidate <> source AND candidate.sage_embedding IS NOT NULL
WITH source, candidate,
     vector.similarity.cosine(source.sage_embedding, candidate.sage_embedding) AS score
WHERE score IS NOT NULL AND score >= $min_similarity
OPTIONAL MATCH (candidate)-[:AT_MERCHANT]->(m:Merchant)
OPTIONAL MATCH (candidate)-[:ROUTED_TO]->(a:Acquirer)
OPTIONAL MATCH (candidate)-[:FAILED_WITH]->(r:ReasonCode)
RETURN candidate.payment_id AS payment_id, candidate.status AS status,
       candidate.amount AS amount, candidate.currency AS currency,
       m.name AS merchant, a.name AS acquirer, r.code AS reason_code,
       score AS similarity
ORDER BY similarity DESC, payment_id ASC
LIMIT $top_k
"""


class GraphSageNotReadyError(RuntimeError):
    """No persisted embeddings exist for this dataset yet."""


class GraphSagePaymentNotFoundError(LookupError):
    """The payment id is not present in Neo4j."""


class GraphSageService:
    def __init__(self, driver: Driver, settings: Settings) -> None:
        self.driver = driver
        self.settings = settings

    def _run(self, query: str, **parameters: Any) -> list[Any]:
        records, _, _ = self.driver.execute_query(
            query, database_=self.settings.neo4j_database, **parameters
        )
        return records

    def status(self) -> GraphSageStatus:
        rows = self._run(
            """MATCH (p:Payment)
               RETURN count(p) AS total,
                      count(p.sage_embedding) AS embedded,
                      count(DISTINCT size(p.sage_embedding)) AS dimensions"""
        )
        total = int(rows[0]["total"]) if rows else 0
        embedded = int(rows[0]["embedded"]) if rows else 0
        detail = self._run(
            """OPTIONAL MATCH (run:GraphSageRun {run_id: 'active'})
               RETURN run.model_name AS model_name, run.trained_at AS trained_at,
                      run.dimension AS dimension, run.train_millis AS train_millis"""
        )[0]
        # Model catalogs are in-memory and can disappear after Neo4j restart.
        return GraphSageStatus(
            ready=total > 0 and embedded == total,
            payment_count=total,
            embedded_payments=embedded,
            dimension=detail["dimension"],
            model_name=detail["model_name"],
            trained_at=detail["trained_at"],
            train_millis=detail["train_millis"],
        )

    def train(self, request: GraphSageTrainRequest) -> GraphSageTrainResponse:
        # Do not erase a prior successful run until its replacement has been trained.
        try:
            self._run("RETURN gds.version() AS version")
        except Neo4jError as exc:
            raise GraphSageNotReadyError(
                'El plugin Neo4j Graph Data Science no está instalado o habilitado.'
            ) from exc
        if not self._run("MATCH (p:Payment) RETURN p LIMIT 1"):
            raise GraphSageNotReadyError('Carga primero el dataset de pagos.')

        graph_name = self.settings.graphsage_graph_name
        model_name = self.settings.graphsage_model_name
        # Re-running the training is intentional. GDS model/graph catalog entries
        # are temporary; replacement is safe because embeddings remain persisted.
        graph_exists = self._run(
            "CALL gds.graph.exists($name) YIELD exists RETURN exists", name=graph_name
        )[0]["exists"]
        if graph_exists:
            self._run(
                "CALL gds.graph.drop($name) YIELD graphName RETURN graphName", name=graph_name
            )
        model_exists = self._run(
            "CALL gds.model.exists($name) YIELD exists RETURN exists", name=model_name
        )[0]["exists"]
        if model_exists:
            self._run(
                "CALL gds.model.drop($name) YIELD modelName RETURN modelName", name=model_name
            )
        for statement in FEATURE_STATEMENTS:
            self._run(statement)
        projection = self._run(PROJECT, graph_name=graph_name)[0]
        if projection["relationshipCount"] == 0:
            raise GraphSageNotReadyError('El grafo proyectado no tiene relaciones de pagos.')
        result = self._run(
            TRAIN,
            graph_name=graph_name,
            model_name=model_name,
            dimension=request.embedding_dimension,
            sample_sizes=request.sample_sizes,
            epochs=request.epochs,
            seed=request.seed,
        )[0]
        written = self._run(WRITE, graph_name=graph_name, model_name=model_name)[0]
        # All payment vectors must be written before recording a successful run.
        count = int(written["nodePropertiesWritten"])
        total = int(self._run("MATCH (p:Payment) RETURN count(p) AS count")[0]["count"])
        if count != total:
            raise RuntimeError(f'Embeddings incompletos: {count} de {total} pagos.')
        trained_at = datetime.now(UTC).isoformat()
        self._run(
            """MERGE (run:GraphSageRun {run_id: 'active'})
               SET run.model_name=$model_name, run.trained_at=$trained_at,
                   run.dimension=$dimension, run.train_millis=$train_millis""",
            model_name=model_name,
            trained_at=trained_at,
            dimension=request.embedding_dimension,
            train_millis=result["trainMillis"],
        )
        return GraphSageTrainResponse(
            model_name=model_name,
            graph_name=graph_name,
            graph_nodes=int(projection["nodeCount"]),
            graph_relationships=int(projection["relationshipCount"]),
            embedded_payments=count,
            embedding_dimension=request.embedding_dimension,
            train_millis=int(result["trainMillis"]),
            trained_at=trained_at,
            epoch_losses=[float(n) for n in result["metrics"].get("epochLosses", [])],
        )

    def similar(
        self, payment_id: str, *, top_k: int = 5, min_similarity: float = 0.0
    ) -> GraphSageSimilarResponse:
        if not self._run(
            "MATCH (p:Payment {payment_id: $payment_id}) RETURN p.payment_id AS id",
            payment_id=payment_id,
        ):
            raise GraphSagePaymentNotFoundError(payment_id)
        if not self._run(
            """MATCH (p:Payment {payment_id: $payment_id})
               WHERE p.sage_embedding IS NOT NULL RETURN p.payment_id AS id""",
            payment_id=payment_id,
        ):
            raise GraphSageNotReadyError('GraphSAGE aún no fue entrenado para este pago.')
        rows = self._run(
            SIMILAR,
            payment_id=payment_id,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        return GraphSageSimilarResponse(
            payment_id=payment_id,
            model_name=self.settings.graphsage_model_name,
            neighbors=[GraphSageNeighbor.model_validate(dict(record)) for record in rows],
        )
