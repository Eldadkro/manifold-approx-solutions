# Plan: connecting RKHS, machine learning, and Gaussian processes

**Status:** Working plan with agreed scope; mathematical emphasis and course alignment remain open.

**Goal:** Develop a short, mathematically grounded account of how RKHS theory leads to a learning algorithm, and how the same kernel yields a Gaussian-process model with uncertainty estimates.

**Agreed scope:** A theoretical exposition of approximately **1–3 pages**, with **no application, dataset, or empirical component**.

**Decisions still open:** Exact course results, mathematical emphasis, and whether references count toward the page limit. Kernel ridge regression (KRR) is the proposed bridge, subject to the author's approval.

## 1. Proposed central question

> How does a kernel turn learning an unknown function into a finite-dimensional computation, and what changes when we interpret that kernel as a covariance function?

The proposed narrative is:

**Bounded evaluation in an RKHS → kernel representations → regularized learning → Gaussian-process conditioning.**

The central comparison is between two approaches to regression:

- **RKHS approach:** Choose a function that balances data fit with an RKHS-norm penalty.
- **GP approach:** Put a probability model on the unknown function and condition on noisy observations.

Under matching assumptions and parameter conventions, the KRR predictor equals the GP posterior mean. The GP formulation additionally supplies a posterior covariance. This agreement does not mean that the two frameworks are identical.

## 2. Confirm the assignment before fixing the scope

Ask the author for:

1. The assignment instructions, to check compatibility with the agreed theoretical scope and clarify whether references count toward the 1–3 pages.
2. The relevant lecture notes or a list of covered definitions and theorems.
3. Whether KRR should be the main learning method or merely a comparison point.

Existing notes refer to lecture definitions and theorem numbers, but these references still need checking against the actual lecture notes. Do not treat them as verified course alignment.

## 3. Mathematical development

### Step A — Explain what the RKHS contributes

Work with real-valued functions on an explicitly specified input domain $\mathcal X$, and a symmetric positive-semidefinite kernel

$$
k:\mathcal X\times\mathcal X\to\mathbb R.
$$

Explain positive semidefiniteness through finite sums, and distinguish the kernel from a sample-dependent Gram matrix.

Introduce the associated RKHS $\mathcal H_k$ and the reproducing property:

$$
f(x)=\langle f,k(\cdot,x)\rangle_{\mathcal H_k}.
$$

**Purpose in the argument:** Function evaluations become inner products, allowing an infinite-dimensional learning problem to be reduced to the span of finitely many kernel sections.

If minimum-norm interpolation is covered in the course, use it as a starting point. Then explain why noisy measurements motivate a soft data-fit penalty instead of exact interpolation.

### Step B — Formulate the regression problem

For observations $(x_i,y_i)_{i=1}^n$, propose squared-loss regression:

$$
\widehat f
=\operatorname*{arg\,min}_{f\in\mathcal H_k}
\left\{\frac1n\sum_{i=1}^n(y_i-f(x_i))^2
+\lambda\|f\|_{\mathcal H_k}^2\right\},
\qquad \lambda>0.
$$

Explain the roles of data fit, regularization, and kernel choice. Avoid describing the RKHS norm as a universal smoothness measure: what it penalizes depends on the kernel.

**Derivation to include:** Write $f=f_V+f_\perp$, where

$$
V=\operatorname{span}\{k(\cdot,x_1),\ldots,k(\cdot,x_n)\}.
$$

By reproduction, $f_\perp(x_i)=0$, while the squared norm splits orthogonally. Since $\lambda>0$, any nonzero perpendicular component increases the objective without improving the fit. Hence the minimizer lies in $V$.

Define

$$
K_{ij}=k(x_i,x_j),\quad K\in\mathbb R^{n\times n},
\qquad
k_x=(k(x_1,x),\ldots,k(x_n,x))^T\in\mathbb R^n,
$$

and $y=(y_1,\ldots,y_n)^T$. Obtain

$$
\widehat f(x)=k_x^T(K+n\lambda I_n)^{-1}y.
$$

**Checks:** Explain why $K+n\lambda I_n$ is invertible even if $K$ is singular. The function minimizer is unique; kernel-expansion coefficients need not be unique when $K$ is singular, although the displayed linear solve gives a valid coefficient vector. In code, solve a linear system rather than forming an inverse.

### Step C — Introduce the GP interpretation

Introduce only the probability needed for the comparison:

$$
f\sim\operatorname{GP}(0,k),\qquad
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2),
$$

with $\sigma^2>0$ and noise independent of $f$. Define a GP through its finite-dimensional Gaussian distributions.

Show the joint Gaussian distribution of training observations and a test function value, then apply Gaussian conditioning to derive

$$
m(x)=\mathbb E[f(x)\mid y]
=k_x^T(K+\sigma^2I_n)^{-1}y,
$$

$$
c(x,x')
=k(x,x')-k_x^T(K+\sigma^2I_n)^{-1}k_{x'}.
$$

**Interpretation:** The posterior mean predicts the latent function. The latent posterior variance is $c(x,x)$; predicting an independently noisy future measurement adds $\sigma^2$.

### Step D — Make the connection precise

Using the same kernel, zero GP prior mean, independent Gaussian observation noise, and the averaged KRR loss above,

$$
\boxed{\sigma^2=n\lambda\quad\Longrightarrow\quad m(x)=\widehat f(x).}
$$

Track kernel amplitude explicitly if it is parameterized separately; do not mix parameter conventions between the two formulations.

Discuss what is shared and what differs:

| Aspect | RKHS / KRR | Gaussian process |
|---|---|---|
| Role of the kernel | Defines a function space and norm | Defines prior covariance |
| Learning principle | Penalized empirical risk minimization | Conditioning a probability model |
| Point prediction | Regularized fitted function | Posterior mean |
| Uncertainty | Not supplied by the KRR point estimate alone | Posterior covariance under the model |

**Important qualification:** Do not say that a GP is simply a random element of its associated RKHS. For common infinite-dimensional kernels, GP sample paths lie outside that RKHS almost surely, even though the posterior mean has a finite kernel expansion in it. A function-space MAP interpretation is unnecessary for the main argument and should be omitted unless carefully justified.

## 4. Length and scope

The manuscript will be a **1–3-page theoretical exposition**, without an application, dataset, simulations, or empirical evaluation. Regression is the mathematical setting, not a proposed experiment.

Prioritize one coherent argument over a survey:

- Define only the RKHS and probability concepts needed for the comparison.
- Retain the short orthogonal-decomposition argument and the resulting KRR predictor.
- State the joint Gaussian model and use the conditioning formula; omit a general derivation of Gaussian conditioning if space is tight.
- Highlight the matching posterior mean and explain what posterior covariance adds.
- Keep qualifications concise and use a small number of verified references.

Aim for roughly two pages of exposition, with flexibility up to three pages for derivations and references. Whether references count toward the limit remains to be clarified.

## 5. Relationship to manifold approximation

Keep this distinction explicit:

- RKHS theory provides a direct connection to the course if it is part of the covered material.
- Using a kernel does **not** by itself establish that an algorithm exploits an underlying data manifold.

Unless the assignment requires more, use the course's function-space and approximation results as the connection. Geometry-aware kernels or manifold-dependent guarantees would be a separate scope decision requiring additional assumptions and evidence.

## 6. Provisional manuscript outline

1. **Motivation and RKHS background:** The central question, kernel, and reproducing property (approximately half a page).
2. **From approximation to learning:** Regularized objective, finite-span argument, and KRR formula (approximately half to one page).
3. **Probabilistic view and comparison:** GP model, conditioning formulas, matching posterior mean, and uncertainty (approximately half to one page).
4. **Closing remarks and references:** Brief qualifications and the precise course connection, within the overall 1–3-page target.

Keep broader topics—classification, deep kernels, sparse GP methods, and manifold learning—out of the initial scope.

## 7. Sources and existing material

Use these local notes as starting material rather than independently verified references:

- [RKHS to kernel ML and GPs](from-rkhs-to-kernel-machine-learning-and-gaussian-processes.md)
- [Kernel ridge regression](kernel-ridge-regression.md)
- [What is a Gaussian process?](what-is-a-gaussian-process.md)
- [What is a posterior?](what-is-a-posterior.md)

Application notes are outside the agreed scope.

Before drafting, verify sources for the RKHS facts, representer argument, and GP conditioning formulas. Separate verified course material from additional statistical background.

## 8. Action checklist

- [ ] Obtain the assignment and relevant lecture notes.
- [ ] Approve or revise the proposed RKHS → KRR → GP narrative.
- [x] Agree on scope: 1–3 pages, no application, dataset, or empirical component.
- [ ] Clarify whether references count toward the page limit.
- [ ] Fix notation, loss normalization, kernel amplitude, and noise conventions.
- [ ] Draft and check the finite-span and Gaussian-conditioning derivations.
- [ ] Verify references and course theorem numbering.
- [ ] Write the LaTeX manuscript in `paper/` and compile the PDF.

**Next discussion:** Within the agreed short theoretical scope, does the RKHS → KRR → GP narrative match the author's intended emphasis, and which results can be assumed from the course?
