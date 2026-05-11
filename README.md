# Scalable LinUCB

LinUCB contextual bandit with a dynamical low-rank approximation of the
per-arm inverse design matrix, updated by the projector-splitting integrator
(PSI). Two variants:

- **Rank-1 updates** (`models/linucb_psi_rank1.py`) — one interaction per step
- **Batch updates** (`models/linucb_psi_batch.py`) — rank-`B` updates from a
  buffer of `B` interactions per arm

Both keep memory and per-step cost `O(d·r)` instead of `O(d^2)`.

## Layout

```
models/
  linucb_psi_rank1.py
  linucb_psi_batch.py
  _factors.py             # PSI integrator + functions
envs/
  mnist_env.py            # MNIST contextual bandit environment
experiments/
  mnist_psi.ipynb         # PSI rank-1 on MNIST
```

## Setup

```bash
pip install -r requirements.txt
```


```python
from models import LinUCBwithPSI_rank1, LinUCBwithPSI_Batch

# rank-1:
model = LinUCBwithPSI_rank1(num_arms=K, d=d, epsilon=1.0, alpha=1.0, rank=64)
for ctx in stream:
    a = int(np.argmax([model.score(ctx[a], a) for a in range(K)]))
    r = env.step(a)
    model.update(ctx[a], a, r)

# batch:
model = LinUCBwithPSI_Batch(num_arms=K, d=d, epsilon=1.0, alpha=1.0, rank=64)
# X_batch: (d, B), rewards: (B,)
model.update(X_batch, arm, rewards)
```
