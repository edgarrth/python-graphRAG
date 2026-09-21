# Experimento: ¿GraphSAGE recupera pagos relevantes adicionales?

## Objetivo y límite de la afirmación

Medir si el **top-K de GraphSAGE**, usando embeddings entrenados por Neo4j GDS, aporta pagos del **mismo incidente ficticio** que **no están** en el top-K de una línea base de relaciones explícitas y reglas de negocio. Además medir precisión, recall, hit-rate y nDCG de ambas recuperaciones bajo **el mismo presupuesto K**. El resultado puede ser cero o negativo. No se mide aquí la calidad del LLM ni la capacidad real de diagnosticar incidentes/fraude.

## Caso experimental concreto

En un procesador de pagos se simulan cuatro incidentes separados:

1. Degradación de switch con códigos `91`/`96`.
2. Disrupción de autenticación con `3DS_TIMEOUT`/`05`.
3. Retraso de captura con `CAPTURE_FAILED`/`96`.
4. Configuración incorrecta de reglas de riesgo con `FRAUD_VELOCITY`/`05`.

Cada caso contiene ocho pagos positivos del mismo incidente simulado, que abarcan distintos comercios, adquirentes y códigos; y cuatro **negativos difíciles** en la misma franja horaria, con atributos coincidentes, que simulan fallas independientes. Hay **48 pagos** `BENCH-001…BENCH-048`, **32 positivos**, **16 controles** y **8 consultas ancla** (dos por incidente). La referencia de verdad es el archivo `datasets/data/benchmark_truth.json`: contiene pertenencia al incidente y consultas ancla, no entra a Neo4j, no se añade a los nodos ni se usa en GDS. Los pagos están definidos en `datasets/benchmark_case.py` y se cargan automáticamente por `datasets/load_dataset.py` salvo que `ENABLE_BENCHMARK_DATA=false`.

**Advertencia de validez:** los incidentes y las etiquetas son una simulación educativa basada en intervenciones hipotéticas, **no evidencia de causas reales**. Algunos negativos son deliberadamente indistinguibles por los atributos observados: ningún modelo podría inferir con certeza una causa distinta a partir de esos mismos datos. No hay garantías de que GraphSAGE obtenga mejores métricas. Para inferencia generalizable se necesitarían incidentes reales adjudicados, más consultas, controles de fuga temporal y evaluación prospectiva.

## Tratamientos y evaluación predefinidos

- **Línea base de grafo y reglas:** una consulta Cypher recupera para cada pago los nodos `Merchant`, `Acquirer`, `ReasonCode`, `Customer`, `PaymentMethod`, atributos y tiempo. Se ordena a todos los candidatos `BENCH-` por **pesos fijos**: mismo comercio +5, adquirente +4, código +3, cliente +2, medio +2, distancia temporal ≤15 min +2, mismo estado +1 y montos dentro de ±10 % +1; desempate determinista por identificador. No se ajustan los pesos usando la referencia. Estos son ejemplos configurables en `application/benchmark.py`, **no reglas clínicas/financieras validadas**.
- **GraphSAGE:** el entrenamiento existente de `application/graphsage.py` genera embeddings `sage_embedding` sobre los pagos y la estructura del grafo. El experimento solicita **realmente a Neo4j** `vector.similarity.cosine` entre cada ancla y los candidatos `BENCH-`, y ordena por similitud. El experimento no sustituye estos valores por predicciones de juguete. El archivo de etiquetas no participa en ninguno de los rankings.
- **Top-K**: K predeterminado = 5, modificable entre 1 y 20, con exactamente los mismos 47 candidatos elegibles por consulta para los dos algoritmos y **sin filtrar puntuaciones** negativas o cero. Nunca se incluye el pago consultado entre sus vecinos.
- **Ground truth:** para una consulta ancla, son relevantes exactamente los **otros siete pagos** del mismo incidente simulado. Los controles negativos y pagos de otros incidentes no son relevantes. Es una etiqueta ajena a las dos estrategias de recuperación.
- **Métricas macro** por las ocho consultas: Precision@K = relevantes recuperados/K; Recall@K = relevantes recuperados/7; Hit@K = al menos un relevante; nDCG@K = DCG binario normalizado. **Hallazgos adicionales** = relevantes en GraphSAGE@K **ausentes** de Reglas@K, contabilizados por consulta y como pagos distintos globales.
- **Unión exploratoria** = deduplicación de los primeros K resultados de reglas + primeros K de GraphSAGE. Se calcula separadamente con presupuesto de **hasta 2K**; **no** debe interpretarse como una ganancia a igual coste frente a cualquiera de los métodos @K.

**Qué NO se está comparando:** la recuperación de documentos del `HybridCypherRetriever` ni la respuesta del generador. En este repo dicho retriever busca `KnowledgeChunk`, no ordena todos los nodos `Payment` de forma equivalente a este ranking de candidatos. El benchmark compara la **recuperación de pagos mediante el grafo explícito/reglas** con **recuperación de pagos GraphSAGE**; no equivale a un ensayo completo de GraphRAG híbrido vs. GraphSAGE.

**Fuga / generalización:** las etiquetas no entran al grafo, pero el modelo actual entrena sobre una proyección que **sí incluye los nodos evaluados sin sus etiquetas**. No es un test inductivo con nodos futuros nunca observados. La similitud coseno no es probabilidad de causa común y el top-K puede compartir solo patrones superficiales.

## Ejecutar con el ZIP

Desde la raíz del proyecto, conservando el archivo `.env` si ya tienes credenciales o parámetros locales:

```bash
# 1) Levantar/actualizar la BD, cargar datos y API/frontend (duración variable por descarga de dependencias/modelos).
docker compose -f infrastructure/docker-compose.yml up -d --build

# 2) Verificar que la carga del dataset haya terminado (espera activa; duración variable).
docker compose -f infrastructure/docker-compose.yml logs dataset-loader

# 3) Entrenar con GDS; la operación puede tardar según CPU/memoria y tamaño del grafo.
curl -f -X POST http://localhost:8000/api/v1/graphsage/train \
  -H 'Content-Type: application/json' \
  -d '{"epochs":5,"embedding_dimension":32,"sample_sizes":[10,5],"seed":42}'

# 4) Comparar (requiere TODOS los embeddings de la cohorte).
curl -f 'http://localhost:8000/api/v1/experiments/graphsage-value?top_k=5' \
  -o graphsage_value_benchmark.json
```

Alternativamente abre `http://localhost:8501`: barra lateral → **GraphSAGE · administración y pruebas** → **Entrenar GraphSAGE**; luego en **Nueva conversación**, pulsa la pregunta de ejemplo **¿GraphSAGE aporta pagos relevantes? (K=5)**. La evaluación se ejecuta dentro del chat, no en el estrecho panel lateral: presenta métricas, una tabla de estrategias, detalles por pago en desplegables de ancho completo y descarga JSON. También puedes escribir `/evaluar graphsage` o `/evaluar graphsage k=10` (1–20). El resultado y sus tablas permanecen en el historial de esa conversación sin repetir la llamada HTTP en cada rerender. Esta operación llama al endpoint de evaluación; no se envía al LLM ni al endpoint SSE de consultas normales. Si la API devuelve 409 por falta de carga o embeddings, el chat muestra el motivo y no conserva un resultado anterior. El chat tradicional/inteligente sigue funcionando independientemente.

Si el dataset no se cargó por completo o GraphSAGE aún no generó embeddings para todos los pagos `BENCH-`, la API responde **409** con instrucciones en lugar de generar resultados ficticios. Si `ENABLE_BENCHMARK_DATA=false` en una base **nueva**, los pagos BENCH- no se cargan y el experimento no estará disponible; esa bandera **no elimina** pagos BENCH- que ya existan en un volumen persistido. Para volver a realizar una comparación válida tras cambiar la configuración de entrenamiento, vuelve a entrenar y ejecuta de nuevo el experimento.

### Lectura de un resultado

- `additional_relevant_hits = 0`: **no se observó** recuperación relevante exclusiva de GraphSAGE@K en estas consultas; no implica que nunca pueda aportar valor.
- Valor positivo: al menos una consulta obtuvo un pago relevante ausente de Reglas@K; **no** demuestra por sí solo que GraphSAGE sea mejor. Compara simultáneamente Precision@K, Recall@K, nDCG@K y ejemplos negativos.
- `distinct_additional_relevant`: elimina repetidos entre distintas consultas, pero no es una medida estadística de generalización.
- `union_at_2k`: mayor presupuesto; no usar para comparar mejoras a igual K.

## Verificación y próximos pasos para evaluar uso productivo

`PYTHONPATH=src pytest -q` valida generación reproducible, verdad separada, fórmula de métricas, reglas, control de preparación y orquestación con transporte simulado. Las pruebas unitarias **no** equivalen a haber ejecutado entrenamiento real ni recuperaciones reales en Neo4j. La integración real requiere iniciar Compose, entrenar GDS y consultar el endpoint. Antes de inferir valor en producción, sustituir el dataset sintético por incidentes adjudicados y anonimizados, establecer baseline de expertos, pre-registrar pesos y K, separar datos temporalmente y evaluar variabilidad por incidente; nunca incluir identificadores de incidente ni outcomes futuros como features del grafo de evaluación.

## Corrección del error 500 y textos de la interfaz

La primera versión buscaba `benchmark_truth.json` subiendo cinco directorios desde
`benchmark.py`. Eso funciona con `PYTHONPATH=src`, pero no en Docker: al instalar el
paquete en `site-packages`, la ruta resultante no apunta a `/app/datasets`. El
endpoint fallaba con un `FileNotFoundError` no controlado. La referencia se
incluye ahora en el paquete instalable (fuera de Neo4j) y se verifica su
existencia; si falta, la API devuelve `409` con instrucciones, nunca un 500
opaco. Otros errores se registran en los logs del servicio API.

Para aplicar la corrección sin borrar datos existentes, en la raíz del proyecto:

```bash
docker compose -f infrastructure/docker-compose.yml up -d --build --force-recreate api frontend
# Si faltan pagos BENCH-: iniciar el cargador y luego volver a entrenar GraphSAGE.
docker compose -f infrastructure/docker-compose.yml logs --tail=80 api
curl -i 'http://localhost:8000/api/v1/experiments/graphsage-value?top_k=5'
```

La evaluación se ejecuta desde una pregunta de ejemplo y se presenta en el chat,
con tablas de ancho completo; el informe JSON conserva todas las métricas.
Un fallo se muestra en un turno independiente y no presenta métricas de ejecuciones
anteriores como si pertenecieran al intento fallido.

## Evaluación integrada al chat

La evaluación ya no aparece en el sidebar. Su pregunta de ejemplo y el comando explícito `/evaluar graphsage k=5` llaman a `GET /api/v1/experiments/graphsage-value?top_k=5` desde el área central. El spinner se muestra dentro del turno del asistente, el informe se guarda en ese mensaje y cada consulta se abre en una sola tabla ancha (sin columnas paralelas estrechas). Se mantiene el límite sintético de la evaluación y se señala que la unión dispone de hasta 2K candidatos. Los errores HTTP 409/500 se muestran como respuestas del chat; una llamada fallida nunca presenta métricas de ejecuciones previas.
