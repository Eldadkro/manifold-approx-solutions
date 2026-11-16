# Why kernel ridge regression stays in the kernel span

## The question

Kernel ridge regression balances training error against an RKHS norm penalty. Could a function outside the span of the training-point kernels reduce the training error enough to offset its larger norm, especially when the regularization weight is small?

**Answer:** the trade-off is real, but it takes place inside the kernel span. Every function outside that span has an orthogonal projection inside it with exactly the same training predictions and a strictly smaller norm. Thus, for any positive regularization weight, a function outside the span cannot be optimal.

## Setup

Let $k:\mathcal X\times\mathcal X\to\mathbb R$ be a symmetric positive-semidefinite kernel, and let $\mathcal H_k$ be its real RKHS. Given training observations $(x_i,y_i)_{i=1}^n$, kernel ridge regression minimizes

$$
J(f)=\frac1n\sum_{i=1}^n(f(x_i)-y_i)^2
+\lambda\|f\|_{\mathcal H_k}^2,
\qquad \lambda>0.
$$

Define the finite-dimensional subspace

$$
S=\operatorname{span}\{k(x_1,\cdot),\ldots,k(x_n,\cdot)\}.
$$

The claim is that the minimizer belongs to $S$, so it has the form

$$
\widehat f=\sum_{i=1}^n\alpha_i k(x_i,\cdot).
$$

This does not require the kernel sections to be linearly independent.

## The key comparison: a function versus its own projection

Because $S$ is finite-dimensional, it is closed. Every $f\in\mathcal H_k$ therefore admits the orthogonal decomposition

$$
f=s+h,\qquad s\in S,\quad h\in S^\perp.
$$

Here $s$ is the orthogonal projection of $f$ onto $S$.

The argument does **not** compare $f$ with an arbitrary function in $S$. Such a comparison could indeed involve a trade-off between fit and norm. Instead, it compares $f=s+h$ with its own projection $s$.

### The training error cannot decrease by adding $h$

The reproducing property gives

$$
h(x_i)=\langle h,k(x_i,\cdot)\rangle_{\mathcal H_k}=0,
$$

because $h$ is perpendicular to every kernel section generating $S$. Therefore

$$
f(x_i)=s(x_i)+h(x_i)=s(x_i)
$$

at every training input. In particular,

$$
\frac1n\sum_{i=1}^n(f(x_i)-y_i)^2
=
\frac1n\sum_{i=1}^n(s(x_i)-y_i)^2.
$$

The component $h$ may change predictions at unobserved inputs, but it is invisible to the training loss.

### The regularization penalty strictly increases

By orthogonality,

$$
\|s+h\|_{\mathcal H_k}^2
=\|s\|_{\mathcal H_k}^2+\|h\|_{\mathcal H_k}^2.
$$

Combining the two observations yields the exact identity

$$
\boxed{J(s+h)-J(s)=\lambda\|h\|_{\mathcal H_k}^2.}
$$

If $h\neq0$ and $\lambda>0$, this difference is strictly positive. Thus, removing $h$ improves the objective without sacrificing any training fit.

Consequently, every minimizer must have $h=0$ and belong to $S$. This projection argument establishes the span property; existence and uniqueness for KRR are discussed in the [main KRR note](kernel-ridge-regression.md).

## Why a small regularization weight does not change the conclusion

The proposed objection is:

> Perhaps the training error goes down while the norm penalty goes up, and a small regularization weight makes that worthwhile.

This can happen when changing $s$, the component inside $S$. It cannot happen by adding $h\perp S$, since that addition produces **zero reduction in training error**.

Even if $\lambda$ is extremely small, a nonzero $h$ adds a positive cost with no benefit to the training loss. No threshold on the size of $\lambda$ is needed: strict positivity suffices.

A function outside $S$ might outperform some poorly chosen function inside $S$. But it can never outperform its own projection into $S$. That is enough to exclude it as a global minimizer.

## Where the trade-off actually happens

For $s=\sum_j\alpha_j k(x_j,\cdot)$, define the Gram matrix $K\in\mathbb R^{n\times n}$ by $K_{ij}=k(x_i,x_j)$ and the target vector $\mathbf y=(y_1,\ldots,y_n)^\top$. Then

$$
J(s)=\frac1n\|K\boldsymbol\alpha-\mathbf y\|_2^2
+\lambda\boldsymbol\alpha^\top K\boldsymbol\alpha.
$$

Changing the function within $S$ can reduce the first term while increasing the second. The regularization weight determines the preferred balance.

Thus:

- The kernel and training inputs determine the subspace $S$.
- The observed targets and regularization weight determine the optimal function within $S$.
- Components perpendicular to $S$ cannot help fit the training observations and are removed by norm regularization.

## Connection to interpolation

The same geometry explains minimum-norm interpolation:

$$
\min_{f\in\mathcal H_k}\|f\|_{\mathcal H_k}^2
\quad\text{subject to}\quad f(x_i)=y_i\quad(i=1,\ldots,n),
$$

assuming these constraints are feasible.

Removing $h\perp S$ preserves every interpolation constraint and decreases the norm. Hence the minimum-norm interpolant also lies in $S$.

The difference is what determines the function within that span:

- **Minimum-norm interpolation:** exact agreement with the observations is required.
- **Kernel ridge regression:** some training error is allowed in exchange for a smaller norm.

Interpolation alone, without a minimum-norm criterion, does not force a function into $S$. If a nonzero $h\in S^\perp$ exists, adding it to an interpolant preserves all training values.

## A one-observation illustration

Suppose there is one observation $(x_1,y_1)$ and $k(x_1,x_1)=1$. Write

$$
f=\alpha k(x_1,\cdot)+h,\qquad h\perp k(x_1,\cdot).
$$

Then

$$
f(x_1)=\alpha,
\qquad
\|f\|_{\mathcal H_k}^2=\alpha^2+\|h\|_{\mathcal H_k}^2,
$$

so

$$
J(f)=(\alpha-y_1)^2+\lambda\alpha^2+\lambda\|h\|_{\mathcal H_k}^2.
$$

The two choices are transparent:

1. Choose $h=0$, since any nonzero $h$ adds only cost.
2. Balance fit and norm through $\alpha$, giving
   $$
   \widehat\alpha=\frac{y_1}{1+\lambda}.
   $$

Minimum-norm interpolation instead chooses $\alpha=y_1$ and $h=0$. Both solutions lie in the same span, but their coefficients differ.

## What if $\lambda=0$?

With no regularization,

$$
J(s+h)=J(s).
$$

A minimizing function in $S$ can still be chosen, but the objective no longer excludes perpendicular components. If $S^\perp$ contains a nonzero function, adding it to a minimizer produces another minimizer outside $S$.

## Scope of the argument

The projection argument does not depend specifically on squared loss. It applies to an objective of the form

$$
L\bigl(f(x_1),\ldots,f(x_n)\bigr)
+\Omega\bigl(\|f\|_{\mathcal H_k}\bigr)
$$

when $\Omega$ is strictly increasing: projection preserves all arguments of $L$ and strictly reduces the penalty for a function outside $S$.

The essential assumptions are that the data-fitting term depends only on these finitely many evaluations and that the penalty rewards a smaller RKHS norm. The argument does not say that the perpendicular component is irrelevant to an unknown population prediction error, nor does it alone guarantee good generalization.

## Related note

- [Kernel ridge regression](kernel-ridge-regression.md), especially Sections 3–5, for the objective, representer argument, and coefficient system.

This note is a self-contained explanation of the mathematical question, not an empirical result or a claim of alignment with particular course lectures.
