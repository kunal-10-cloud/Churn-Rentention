from src.rag import build_index, retrieve

build_index()

query = "customer with high monthly charges and month-to-month contract"

results = retrieve(query)
print("RESULTS:", results)
for r in results:
    print("\n---\n", r)