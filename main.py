# Main file is just to compare speed on CPU and GPU between the numba and the JAX implementations.
import time

import jax
import matplotlib.pyplot as plt
import numpy as np

import hilbert_sort.jax as jax_backend
from hilbert_sort.numba import hilbert_sort as nb_hilbert_sort

jax.config.update("jax_enable_x64", True)  # this needs to be set to true in order to use int64

backend = "gpu"
n_runs = 10
jax_hilbert_sort = jax.jit(jax_backend.hilbert_sort, backend=backend)

Ds = np.arange(1, 7, dtype=int)
Ns = np.logspace(1, 6, num=6, dtype=int, base=10)

runtime_nb = np.empty((Ns.shape[0], Ds.shape[0]))
runtime_jax = np.empty((Ns.shape[0], Ds.shape[0]))

for i, n in enumerate(Ns):
    for j, d in enumerate(Ds):
        x = np.random.randn(n, d)

        # compilation run
        _ = nb_hilbert_sort(x)

        # Numba runtime loop
        tic = time.time()
        for _ in range(n_runs):
            _ = nb_hilbert_sort(x)
        nb_time = (time.time() - tic) / n_runs

        # compilation run
        res = jax_hilbert_sort(x)
        res.block_until_ready()

        # JAX runtime loop
        tic = time.time()
        for _ in range(n_runs):
            res = jax_hilbert_sort(x)
            res.block_until_ready()
        jax_time = (time.time() - tic) / n_runs

        runtime_nb[i, j] = nb_time
        runtime_jax[i, j] = jax_time
    print(f"Iteration {i+1} out of {Ns.shape[0]} done (N={n}). avg nb = {runtime_nb[i].mean()}, avg jax = {runtime_jax[i].mean()}")

print()
NN, DD = np.meshgrid(Ns, Ds)

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

vmin = np.minimum(runtime_nb.min(), runtime_jax.min())
vmax = np.maximum(runtime_nb.max(), runtime_jax.max())

axes[0].pcolor(NN, DD, runtime_nb, vmin=vmin, vmax=vmax)
axes[0].set_title("Numba runtime (s)")
im = axes[1].pcolor(NN, DD, runtime_jax, vmin=vmin, vmax=vmax)
axes[1].set_title("JAX runtime (s)")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
