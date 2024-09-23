# pathfinder


We have multiple datasets arranged into $D$ domains and $M$ modalities. For example, domains can be "species", and modalities can be different types of measurements per species. 

Each dataset is a matrix $\mathbf{X_{d,m}}$ denoting data from domain $d\in \{1,\dots,D\}$ and modality $m\in \{1,\dots,M\}$. The size of such matrix is $r_d\times c_m$, i.e. it has $r_d$ rows and $c_m$ columns. 

Through this notation, it can be seen that within any given domain, all datasets share the row dimension, and within any given modality, all the datasets share the column dimension.

We further assume that we don't necessarily have access to all modalities in all domains. We are missing some of the $\mathbf{X_{d,m}}$'s.


Our objective is to find a set of low-rank matrix decompositions $\mathbf{X_{d,m}} = \mathbf{A_{d}}\mathbf{S_{m}^T}$ for all $d$ and all $m$. For a rank-$k$ decomposition, we have $\mathbf{A_d}\in \mathbb{R}^{r_d}\times\mathbb{R}^{k}$ and $\mathbf{S_m}\in \mathbb{R}^{c_m}\times\mathbb{R}^{k}$.

Such decompositions mean that we want to find common subspaces within each domain $d$ and each modality $m$. 

The above decompositions are ill-defined unless we add additional constraints on the matrices $\mathbf{A_d}$ and $\mathbf{S_m}$. There are many options, but here, we will use the constraints used in ridge regression, i.e. use the L2-norm.

Before we write the loss function, we will further assume that we do not have access to all the datasets $\mathbf{X_{d,m}}$. Some of the modalities might be missing in some of the domains. We define a mask $\mathcal{M}$ as the set of matrices that we do have access to, i.e. $\mathcal{M}=\left\{(d,m) \mid \mathbf{X_{d,m}} \text{ exists} \right\}$. 

Now we can write the overall loss function as:

$$ \mathrm{Loss} = \sum_{(d,m)\in\mathcal{M}} \frac{1}{2}\left\lVert \mathbf{X_{d,m}}-\mathbf{A_{d}}\mathbf{S_{m}^T} \right\rVert^2_{\mathrm{F}} + \alpha \sum_{d=1}^D \left\lVert \mathbf{A_d} \right\rVert^{2}_2 + \alpha \sum_{m=1}^M \left\lVert \mathbf{S_{m}} \right\rVert^2_2$$

# how to run it
TBD
