"""
This code is similar to the datafold tutorial. Here we form the swiss-roll data set
and use the built-in functionality of DiffusionMaps in datafold library.

We then plot the eigen vectors pair-wise.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.utils.plot import plot_pairwise_eigenvector

nr_samples = 5000
nr_samples_plot = 5000
idx_plot = np.random.permutation(nr_samples)[0:nr_samples_plot]

# generate point cloud for swiss-roll dataset
data, data_color = make_swiss_roll(nr_samples,
                                   random_state=3,
                                   noise=0)

# plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data[idx_plot, 0], data[idx_plot, 1], data[idx_plot, 2],
           c=data_color[idx_plot],
           cmap=plt.cm.Spectral)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("point cloud on S-shaped manifold")
ax.view_init(10, 70)
plt.show()

data_pcm = pfold.PCManifold(data)
data_pcm.optimize_parameters(result_scaling=0.5)

# Plot eigen vectors pair-wise
dmap = dfold.DiffusionMaps(kernel=pfold.GaussianKernel(epsilon=data_pcm.kernel.epsilon),
                           n_eigenpairs=9,
                           dist_kwargs=dict(cut_off=data_pcm.cut_off))
dmap = dmap.fit(data_pcm)
evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

plot_pairwise_eigenvector(eigenvectors=dmap.eigenvectors_[idx_plot, :],
                          n=1,
                          fig_params=dict(figsize=[15, 15]),
                          scatter_params=dict(cmap=plt.cm.Spectral,
                                              c=data_color[idx_plot]))
plt.show()
