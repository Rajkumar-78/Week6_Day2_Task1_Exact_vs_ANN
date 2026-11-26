# HNSW in 5 lines
1. HNSW builds a multi-layer navigable graph; upper layers are sparse and have long-range links.
2. Search begins at an entry point in the top layer and greedily moves to closer vectors.
3. It descends one layer at a time, carrying the best candidate into the next layer.
4. The dense bottom layer explores a limited candidate set controlled by `efSearch`.
5. Larger `M` and `efSearch` often improve recall but increase memory, construction work, and/or query time.

| Aspect | Exact search | ANN (HNSW) |

| Accuracy | Exact top-k | Approximate; may miss neighbours |
| Speed | Checks all vectors; cost grows with N | Usually faster for large datasets |
| Memory | Vectors plus little flat-index overhead | Vectors plus graph links |
| Choose when | Small data or exact results required | Large data, low latency, high recall acceptable |
