import time
import numpy as np
import matplotlib.pyplot as plt
import faiss

SEED,N,D,K=42,50000,128,10
rng=np.random.default_rng(SEED)
x=rng.random((N,D),dtype=np.float32)
q=rng.random((1,D),dtype=np.float32)

t=time.perf_counter()
d=np.sum((x-q)**2,axis=1)
ids=np.argpartition(d,K-1)[:K]
ids=ids[np.argsort(d[ids])]
numpy_ms=(time.perf_counter()-t)*1000
truth=set(map(int,ids))
print(f"Dataset: {N} vectors x {D} dimensions; k={K}; seed={SEED}")
print(f"NumPy exact search: {numpy_ms:.3f} ms")

flat=faiss.IndexFlatL2(D)
t=time.perf_counter(); flat.add(x); build=(time.perf_counter()-t)*1000
t=time.perf_counter(); fd,fi=flat.search(q,K); flat_ms=(time.perf_counter()-t)*1000
recall=len(truth.intersection(map(int,fi[0])))/K
print(f"FAISS IndexFlatL2 build: {build:.3f} ms")
print(f"FAISS IndexFlatL2 search: {flat_ms:.3f} ms")
print(f"IndexFlatL2 recall@10: {recall:.2f}")

rows=[]
for M in (16,32,48):
    idx=faiss.IndexHNSWFlat(D,M,faiss.METRIC_L2)
    idx.hnsw.efConstruction=100
    t=time.perf_counter(); idx.add(x); build_ms=(time.perf_counter()-t)*1000
    for ef in (16,32,64,128,256):
        idx.hnsw.efSearch=max(ef,K)
        idx.search(q,K) # warmup
        t=time.perf_counter(); _,found=idx.search(q,K)
        ms=(time.perf_counter()-t)*1000
        r=len(truth.intersection(map(int,found[0])))/K
        rows.append((M,ef,build_ms,ms,r))
        print(f"HNSW M={M:2d} efSearch={ef:3d}: build={build_ms:.1f} ms, query={ms:.3f} ms, recall@10={r:.2f}")
plt.figure(figsize=(9,6))
for M in (16,32,48):
    sub=[a for a in rows if a[0]==M]
    plt.plot([a[3] for a in sub],[a[4] for a in sub],marker="o",label=f"M={M}")
    for a in sub: plt.annotate(f"ef={a[1]}",(a[3],a[4]),xytext=(4,4),textcoords="offset points",fontsize=8)
plt.xlabel("HNSW query time (ms)"); plt.ylabel("Recall@10 against NumPy exact")
plt.title("HNSW recall vs query time (50k x 128)"); plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig("recall_vs_time.png",dpi=160)
print("Saved plot: recall_vs_time.png")
print("Exact top-10 IDs:",ids.tolist())
print("FAISS exact top-10 IDs:",fi[0].tolist())
