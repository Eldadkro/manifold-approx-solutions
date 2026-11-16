# Singular Gram matrices in RKHS interpolation

This note explains when a kernel Gram matrix can be singular, which target values can still be interpolated, and why singularity does not destroy uniqueness of the minimum-norm interpolating function. The statements below are general RKHS background; alignment with specific course results has not been checked against lecture notes.

## 1. Setup: minimum-norm interpolation

Let $\mathcal H$ be a real RKHS of functions on a domain $X$, with reproducing kernel $k:X\times X\to\mathbb R$. Write $k_x=k(\cdot,x)$. The reproducing property is

$$
f(x)=\langle f,k_x\rangle_{\mathcal H}.
$$

Given sites $x_1,\ldots,x_n\in X$ and targets $y=(y_1,\ldots,y_n)^\top\in\mathbb R^n$, consider

$$
\min_{f\in\mathcal H}\|f\|_{\mathcal H}^2
\quad\text{subject to}\quad f(x_i)=y_i,\quad i=1,\ldots,n.
$$

Here “optimal” means minimum RKHS norm among all exact interpolants. Define the finite-sample Gram matrix

$$
K\in\mathbb R^{n\times n},\qquad K_{ij}=k(x_i,x_j).
$$

The kernel $k$ is a function on $X\times X$; $K$ is a matrix depending on the chosen sites. These are distinct objects.

## 2. Why the Gram matrix may be singular

An RKHS kernel is symmetric and positive semidefinite. For every $c\in\mathbb R^n$,

$$
c^\top Kc
=\left\|\sum_{i=1}^n c_i k_{x_i}\right\|_{\mathcal H}^2
\ge 0.
$$

Positive semidefiniteness alone does not imply invertibility. In fact,

$$
K\text{ is invertible}
\iff k_{x_1},\ldots,k_{x_n}\text{ are linearly independent}.
$$

Typical causes of singularity include:

- **Repeated sites:** if $x_i=x_j$, the corresponding rows and columns coincide.
- **Finite-dimensional features:** if $k(x,z)=\phi(x)^\top\phi(z)$ with $\phi(x)\in\mathbb R^d$, then $K=\Phi\Phi^\top$, where row $i$ of $\Phi\in\mathbb R^{n\times d}$ is $\phi(x_i)^\top$. Thus $\operatorname{rank}(K)\le d$, and $n>d$ forces singularity.
- **Other dependencies among kernel sections:** distinct sites alone are not enough to guarantee invertibility.

A kernel is **strictly positive definite** if $c^\top Kc>0$ for every nonzero $c$ and every finite collection of pairwise distinct sites. This stronger property, together with distinct sites, guarantees invertibility.

## 3. Reduction to a finite linear system

Let

$$
S=\operatorname{span}\{k_{x_1},\ldots,k_{x_n}\}\subseteq\mathcal H.
$$

Since $S$ is finite dimensional, every $f\in\mathcal H$ has an orthogonal decomposition $f=s+h$, with $s\in S$ and $h\in S^\perp$. The reproducing property gives

$$
h(x_i)=\langle h,k_{x_i}\rangle_{\mathcal H}=0.
$$

Consequently, $f$ and $s$ have the same values at all interpolation sites, while

$$
\|f\|_{\mathcal H}^2=\|s\|_{\mathcal H}^2+\|h\|_{\mathcal H}^2.
$$

Thus any minimum-norm interpolant must lie in $S$. Writing

$$
s=\sum_{j=1}^n\alpha_j k_{x_j},\qquad \alpha\in\mathbb R^n,
$$

the constraints become

$$
K\alpha=y.
$$

This argument also shows that an interpolant anywhere in $\mathcal H$ exists if and only if this finite linear system is solvable.

## 4. Exactly which targets are feasible?

Because $K$ is symmetric,

$$
\operatorname{range}(K)=\ker(K)^\perp.
$$

Therefore,

$$
\boxed{
y\in\operatorname{range}(K)
\iff c^\top y=0\text{ for every }c\in\ker(K).
}
$$

Equivalently, interpolation is impossible precisely when there exists $c$ such that

$$
Kc=0\quad\text{but}\quad c^\top y\ne0.
$$

### Interpretation through the reproducing property

If $c\in\ker(K)$, then

$$
\sum_i c_i k_{x_i}=0.
$$

Every function $f\in\mathcal H$ must consequently satisfy

$$
\sum_i c_i f(x_i)
=\left\langle f,\sum_i c_i k_{x_i}\right\rangle_{\mathcal H}
=0.
$$

The targets must obey all these same relations. They are not merely constraints on a coefficient representation; they are constraints on the values of every function in the RKHS.

If $r=\operatorname{rank}(K)$, the feasible targets form an $r$-dimensional subspace of $\mathbb R^n$. A basis of $\ker(K)$ gives $n-r$ independent linear compatibility constraints. If $K$ is invertible, there are no such constraints and every target vector is feasible.

## 5. Examples

### Constant kernel

Let $k(x,z)=1$ on a nonempty domain $X$. The RKHS consists of constant functions. For two sites,

$$
K=\begin{pmatrix}1&1\\1&1\end{pmatrix},
\qquad \ker(K)=\operatorname{span}\{(1,-1)^\top\}.
$$

The compatibility condition is $y_1-y_2=0$:

- $y=(2,2)^\top$ is feasible, with interpolant $f(x)=2$.
- $y=(2,3)^\top$ is infeasible: no constant function has these values.

### Linear kernel at distinct sites

Let $X=\mathbb R$ and $k(x,z)=xz$. This RKHS consists of functions $f(x)=ax$. At $x_1=1$ and $x_2=2$,

$$
K=\begin{pmatrix}1&2\\2&4\end{pmatrix},
\qquad \ker(K)=\operatorname{span}\{(-2,1)^\top\}.
$$

Thus feasibility requires $y_2=2y_1$. For example, $(3,6)^\top$ is feasible, whereas $(3,7)^\top$ is not. The sites are distinct, but the kernel still constrains their values.

### Repeated sites

Whenever $x_i=x_j$, feasibility requires $y_i=y_j$. Agreement at repeated sites is necessary, but it need not be sufficient: the kernel may impose additional relations, as in the examples above.

## 6. Nonunique coefficients, unique optimal function

Suppose $y\in\operatorname{range}(K)$. Let $K^\dagger$ denote the Moore–Penrose pseudoinverse. All solutions of the coefficient system are

$$
\alpha=K^\dagger y+c,\qquad c\in\ker(K).
$$

Although these coefficient vectors can differ, their functions do not, since

$$
\left\|\sum_i c_i k_{x_i}\right\|_{\mathcal H}^2=c^\top Kc=0.
$$

Hence the unique minimum-norm interpolant is

$$
f_* = \sum_{j=1}^n (K^\dagger y)_j k_{x_j},
\qquad
\|f_*\|_{\mathcal H}^2=y^\top K^\dagger y.
$$

Every other exact interpolant has the form $f_*+h$ for $h\in S^\perp$, and

$$
\|f_*+h\|_{\mathcal H}^2=\|f_*\|_{\mathcal H}^2+\|h\|_{\mathcal H}^2.
$$

This proves uniqueness of the minimum-norm function. It does **not** imply uniqueness of all interpolating functions. The pseudoinverse selects the coefficient solution with minimum Euclidean norm; coefficient norm and RKHS function norm are different quantities.

## 7. What if the targets are incompatible?

When $y\notin\operatorname{range}(K)$, no exact RKHS interpolant exists. Applying the pseudoinverse does not remove this obstruction. Instead, the function

$$
f_{\mathrm{LS}}=\sum_j(K^\dagger y)_j k_{x_j}
$$

has fitted values

$$
\widehat y=KK^\dagger y=P_{\operatorname{range}(K)}y.
$$

These are the Euclidean orthogonal projection of $y$ onto the feasible target space. Thus $f_{\mathrm{LS}}$ is the minimum-RKHS-norm function among all minimizers of the unregularized squared training error. The residual

$$
y-\widehat y\in\ker(K)
$$

is the part of the targets that this RKHS cannot represent at these sites.

An alternative is **kernel ridge regression**, which changes the problem to

$$
\min_{f\in\mathcal H}
\frac1n\sum_{i=1}^n(f(x_i)-y_i)^2+\lambda\|f\|_{\mathcal H}^2,
\qquad \lambda>0.
$$

One coefficient representation of its unique minimizing function is

$$
\alpha_\lambda=(K+n\lambda I_n)^{-1}y.
$$

The shifted matrix is positive definite even if $K$ is singular. However, this is regularized fitting, not exact interpolation. In particular, its fitted values still belong to $\operatorname{range}(K)$, so regularization cannot represent an incompatible component of $y$. See [Kernel ridge regression](kernel-ridge-regression.md) and [Why kernel ridge regression stays in the kernel span](why-kernel-ridge-regression-stays-in-the-kernel-span.md).

## 8. Exact singularity versus numerical ill-conditioning

These are different issues:

- **Exact singularity:** a zero eigenvalue reflects an exact dependency and imposes a compatibility constraint.
- **Ill-conditioning:** a positive but very small eigenvalue can make a mathematically invertible system sensitive to perturbations and floating-point error.

With an eigendecomposition $K=Q\operatorname{diag}(\mu_1,\ldots,\mu_n)Q^\top$, exact feasibility requires

$$
q_j^\top y=0\quad\text{whenever }\mu_j=0.
$$

For feasible targets, the minimum-norm formula becomes

$$
\|f_*\|_{\mathcal H}^2
=\sum_{\mu_j>0}\frac{(q_j^\top y)^2}{\mu_j}.
$$

Small positive eigenvalues can therefore make interpolation expensive in RKHS norm when the targets have components in those directions.

In numerical work, use an SVD or symmetric eigendecomposition with a stated, scale-aware tolerance to identify numerical rank. Check the projection residual $\|y-KK^\dagger y\|_2$ relative to the target scale and relevant error tolerances. A numerically truncated pseudoinverse may discard small positive eigenvalues, so this tests approximate compatibility under the selected tolerance, not exact mathematical feasibility. Avoid forming an explicit matrix inverse to solve a linear system.

## Summary

A singular Gram matrix does not automatically prevent optimal interpolation. The decisive question is whether the targets satisfy all relations imposed by the kernel:

$$
y\in\operatorname{range}(K)
\iff y\perp\ker(K).
$$

If they do, the minimum-norm interpolating function is unique despite nonunique coefficients. If they do not, exact interpolation is impossible in the chosen RKHS; pseudoinverse fitting or regularization solves a different approximation problem.
