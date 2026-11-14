# Kernel ridge regression

> [!summary]
> Kernel ridge regression (KRR) learns a real-valued function by balancing squared prediction error against its norm in a reproducing kernel Hilbert space (RKHS). Although the search space can be infinite-dimensional, the optimal function is a linear combination of kernel functions centered at the training inputs. Training therefore reduces to solving a regularized linear system.

## 1. Learning problem and notation

Given observations $(x_i,y_i)_{i=1}^n$ with $x_i\in\mathcal X$ and $y_i\in\mathbb R$, we want to predict the response at a new input $x$.

For example, $x$ might contain a home's area, age, and location, and $y$ its sale price. A kernel can model nonlinear relationships without explicitly constructing a large feature vector. This is a motivating example, not a reported experiment or a selected homework application.

Choose a symmetric positive-semidefinite kernel $k:\mathcal X\times\mathcal X\to\mathbb R$. Positive semidefiniteness means that, for every finite collection of inputs and coefficients $c_i\in\mathbb R$,

$$
\sum_{i,j}c_i c_j k(x_i,x_j)\geq 0.
$$

Let $\mathcal H_k$ be its RKHS. Define

$$
K_{ij}=k(x_i,x_j),\qquad K\in\mathbb R^{n\times n},\qquad
\mathbf y=(y_1,\ldots,y_n)^\top\in\mathbb R^n.
$$

The **kernel** is a function defined on the domain; the **Gram matrix** $K$ is its evaluation on a particular training set.

## 2. Why an RKHS is useful

For each $x\in\mathcal X$, the function $k(x,\cdot)$ belongs to $\mathcal H_k$, and the reproducing property states

$$
f(x)=\langle f,k(x,\cdot)\rangle_{\mathcal H_k}.
$$

Thus, evaluating a function is an inner product with a known element of the space. In particular,

$$
|f(x)|\leq \|f\|_{\mathcal H_k}\sqrt{k(x,x)}.
$$

This connects pointwise predictions to Hilbert-space geometry and makes norm regularization mathematically meaningful. The norm measures complexity **relative to the chosen kernel**; it is not a universal measure of smoothness.

## 3. Regularized objective

For $\lambda>0$, KRR solves

$$
\boxed{
\widehat f=\arg\min_{f\in\mathcal H_k}
\left\{\frac1n\sum_{i=1}^n(f(x_i)-y_i)^2
+\lambda\|f\|_{\mathcal H_k}^2\right\}.
}
$$

- The loss rewards agreement with observed responses.
- The RKHS norm discourages functions that are expensive in the kernel's geometry.
- $\lambda$ sets the trade-off: larger values produce stronger shrinkage.

This formulation has no separate unpenalized intercept. Adding one changes the optimization problem and the linear system; it should not be silently assumed.

## 4. Representer theorem: why finitely many coefficients suffice

For this objective, the representer argument requires only that the data-fitting term depend on finitely many evaluations and that the penalty strictly increase with the RKHS norm. Here $\lambda>0$ ensures that strict increase.

Let

$$
S=\operatorname{span}\{k(x_1,\cdot),\ldots,k(x_n,\cdot)\}.
$$

Since $S$ is finite-dimensional, decompose any $f\in\mathcal H_k$ as $f=f_S+f_\perp$, with $f_\perp\perp S$. The reproducing property gives

$$
f_\perp(x_i)=\langle f_\perp,k(x_i,\cdot)\rangle_{\mathcal H_k}=0.
$$

Removing $f_\perp$ therefore leaves every training prediction unchanged, while

$$
\|f\|_{\mathcal H_k}^2=\|f_S\|_{\mathcal H_k}^2+\|f_\perp\|_{\mathcal H_k}^2.
$$

Any nonzero perpendicular component strictly increases the objective. Consequently,

$$
\widehat f(\cdot)=\sum_{j=1}^n\alpha_j k(x_j,\cdot).
$$

**Interpretation:** directions invisible to the training evaluations cannot improve the loss, so regularization removes them.

## 5. Deriving the linear system

For a coefficient vector $\boldsymbol\alpha\in\mathbb R^n$,

$$
\mathbf f_X=K\boldsymbol\alpha,
\qquad
\|f\|_{\mathcal H_k}^2=\boldsymbol\alpha^\top K\boldsymbol\alpha.
$$

The coefficient objective is

$$
J(\boldsymbol\alpha)=\frac1n\|K\boldsymbol\alpha-\mathbf y\|_2^2
+\lambda\boldsymbol\alpha^\top K\boldsymbol\alpha.
$$

Differentiating using symmetry of $K$ gives

$$
\nabla J=\frac2n K(K\boldsymbol\alpha-\mathbf y)+2\lambda K\boldsymbol\alpha.
$$

A solution of

$$
\boxed{(K+n\lambda I_n)\boldsymbol\alpha=\mathbf y}
$$

makes this gradient zero and hence minimizes the convex coefficient objective. Because $K\succeq0$ and $\lambda>0$, the shifted matrix is positive definite and invertible.

> [!warning] Do not cancel a possibly singular Gram matrix
> The gradient equation is $K[(K+n\lambda I_n)\boldsymbol\alpha-\mathbf y]=0$. Deriving the displayed system by cancelling $K$ would require invertibility. Instead, the system provides a valid canonical minimizing coefficient vector even when $K$ is singular. Other minimizing coefficient vectors can differ by elements of $\ker K$, but they represent the same RKHS function: $\|\sum_j v_j k(x_j,\cdot)\|_{\mathcal H_k}^2=v^\top Kv=0$ for $v\in\ker K$.

The minimizer as a **function** is unique because the squared RKHS norm makes the function-space objective strictly convex. The explicit finite-dimensional construction also establishes existence.

For a new input, define $\mathbf k_x=(k(x_1,x),\ldots,k(x_n,x))^\top$. Then

$$
\boxed{\widehat f(x)=\mathbf k_x^\top(K+n\lambda I_n)^{-1}\mathbf y.}
$$

In code, solve the linear system rather than explicitly forming the inverse; Cholesky factorization is a natural choice.

### Normalization matters

If the loss is an **unnormalized sum**, $\sum_i(f(x_i)-y_i)^2+\eta\|f\|^2$, the system is $(K+\eta I_n)\boldsymbol\alpha=\mathbf y$. The conventions agree when $\eta=n\lambda$.

## 6. Spectral interpretation of regularization

Write $K=U\operatorname{diag}(\mu_1,\ldots,\mu_n)U^\top$, with $\mu_j\geq0$. The fitted training responses are

$$
\widehat{\mathbf y}
=K(K+n\lambda I_n)^{-1}\mathbf y
=U\operatorname{diag}\!\left(\frac{\mu_j}{\mu_j+n\lambda}\right)U^\top\mathbf y.
$$

Each Gram-matrix eigenvector direction is shrunk by a factor between zero and one. Small-eigenvalue directions are suppressed most strongly; zero-eigenvalue directions cannot be fitted at all.

- As $\lambda\to\infty$, predictions approach zero for this zero-intercept formulation.
- As $\lambda\downarrow0$, training predictions approach the projection of $\mathbf y$ onto $\operatorname{range}(K)$.
- If $K$ is invertible, the latter limit interpolates the training responses. Interpolation alone says nothing about test accuracy.

## 7. Connection to ordinary ridge regression

For $x,z\in\mathbb R^d$, the linear kernel $k(x,z)=x^\top z$ yields ordinary ridge regression. Let $X\in\mathbb R^{n\times d}$ have rows $x_i^\top$. Then $K=XX^\top$ and

$$
\widehat w=(X^\top X+n\lambda I_d)^{-1}X^\top\mathbf y
=X^\top(XX^\top+n\lambda I_n)^{-1}\mathbf y.
$$

The predictor is $\widehat f(x)=x^\top\widehat w$. More generally, a kernel supplies an implicit feature map satisfying $k(x,z)=\langle\phi(x),\phi(z)\rangle$. KRR performs ridge regression in that feature space without explicitly constructing its coordinates.

## 8. Choosing a kernel and evaluating a model

On $\mathbb R^d$, a common choice is the Gaussian/RBF kernel

$$
k_\ell(x,z)=\exp\left(-\frac{\|x-z\|_2^2}{2\ell^2}\right),\qquad \ell>0.
$$

The length scale $\ell$ controls how quickly similarity decreases with distance. It and $\lambda$ must be selected jointly: the kernel determines the available functions and their norm, while regularization determines the cost of using them.

A minimal evaluation procedure is:

1. Split data into training, validation, and held-out test sets using a scheme appropriate to the sampling process.
2. Fit preprocessing, including feature scaling, on training data only.
3. Select kernel parameters and regularization using validation data or cross-validation. Refit preprocessing inside each cross-validation fold.
4. Refit the chosen pipeline on training plus validation data, then report test error once, for example RMSE.
5. Compare with a mean-prediction baseline and linear ridge regression. Record splits, seeds, preprocessing, and hyperparameters.

For temporal or grouped observations, use time-aware or group-aware splits rather than assuming random splits are valid.

Dense exact KRR typically uses $O(n^2)$ memory for $K$ and $O(n^3)$ arithmetic for factorization. A prediction requires $n$ kernel evaluations. Large datasets may require approximations such as Nyström features or random features.

## 9. What this does—and does not—say about manifolds

KRR is a useful bridge from RKHS theory to machine learning: the reproducing property explains evaluation, the representer argument gives a finite model, and regularization yields a stable linear solve.

However, an ambient Euclidean RBF kernel does **not automatically encode intrinsic manifold geometry**. If inputs lie near a manifold, exploiting that structure requires additional assumptions or a justified geometry-aware construction. Substituting geodesic distance into a Gaussian formula does not automatically preserve positive semidefiniteness.

For the Manifold Approximations assignment, KRR remains a possible application, not a settled topic. Alignment with the course and the choice of dataset require the assignment instructions and covered results.

## Provenance and scope

This is an assistant-written mathematical synthesis with self-contained derivations, not a summary of a specific paper or lecture. No experiments were run and no dataset-specific claims are made. Bibliographic sources and course theorem references should be verified before adapting the note into a manuscript.

## Related project notes

- [From RKHS to kernel machine learning and Gaussian processes](from-rkhs-to-kernel-machine-learning-and-gaussian-processes.md) — broader context connecting interpolation, regularized learning, and Gaussian processes. Compare regularization conventions carefully: this note uses mean squared loss and hence the shift $n\lambda I$.
- [Real-world use cases](real-world-use-cases-of-rkhs-kernel-learning-and-gaussian-processes.md) — candidate applications for the assignment.
- [What is a Gaussian process?](what-is-a-gaussian-process.md) — the probabilistic viewpoint related to kernel prediction.
