from search_params import evaluate_params

# CASE C: lam0=20, lam1=10, delta=20, f=600
res = evaluate_params(20, 10, 20, 600)
print("CASE C:")
for k, v in res.items():
    print(f"  {k}: {v}")
