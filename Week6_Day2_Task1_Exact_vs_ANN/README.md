# Task 1 — Exact vs Approximate Search
### Run on Windows PowerShell
Install Python 3.10/3.11, open this folder in VS Code:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python task1_exact_vs_ann.py
```
Creates 50,000 seeded 128-d vectors and one query; compares NumPy exact, FAISS IndexFlatL2 exact, and HNSW for M=16/32/48 and efSearch=16/32/64/128/256. Prints build/query times and recall@10; saves `recall_vs_time.png`.
Example timings in the assignment are illustrative; your computer's actual times will differ.
