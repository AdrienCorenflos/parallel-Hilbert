# Main file is just to compare speed on CPU and GPU between the numba and the JAX implementations.
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import hilbert_sort.jax as jax_backend
from hilbert_sort.numba import hilbert_sort as nb_hilbert_sort

jax.config.update("jax_enable_x64", True)  # this needs to be set to true in order to use int64

backend = "cpu"
n_runs = 10
jax_hilbert_sort = jax.jit(jax_backend.hilbert_sort, backend=backend)

Ds = np.arange(1, 6, dtype=int)
Ns = np.logspace(1, 5, num=5, dtype=int, base=10)

runtime_nb = np.empty((Ds.shape[0], Ns.shape[0]))
runtime_jax = np.empty((Ds.shape[0], Ns.shape[0]))

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

        runtime_nb[j, i] = nb_time
        runtime_jax[j, i] = jax_time
    print(f"Iteration {i+1} out of {Ns.shape[0]} done (N={n}). avg nb = {runtime_nb[:, i].mean()}, avg jax = {runtime_jax[:, i].mean()}")

print()
NN, DD = np.meshgrid(Ns, Ds)

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

vmin = np.minimum(runtime_nb.min(), runtime_jax.min())
vmax = np.maximum(runtime_nb.max(), runtime_jax.max())

axes[0].pcolormesh(np.log10(NN), DD, runtime_nb, norm=LogNorm(vmin=vmin, vmax=vmax))
axes[0].set_title("Numba runtime (s)")
im = axes[1].pcolormesh(np.log10(NN), DD, runtime_jax, norm=LogNorm(vmin=vmin, vmax=vmax))
axes[1].set_title("JAX runtime (s)")
axes[0].set_xlabel("$\log_{10}(N)$")
axes[0].set_ylabel("$d_X$")
axes[1].set_xlabel("$\log_{10}(N)$")

fig.subplots_adjust(right=0.8)
fig.suptitle(f"Runtime comparison between Numba and JAX ({backend.upper()}) Hilbert sort", fontsize=15)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
