# Random functions, covariance kernels, and the connection to KRR

This note supplies probability background for [From RKHS to kernel machine learning and Gaussian processes](from-rkhs-to-kernel-machine-learning-and-gaussian-processes.md). It does not assume that these probabilistic results were covered in the lectures. All functions and kernels below are real-valued.

## 1. What is a random function?

An ordinary function assigns a fixed value to each input:

$$
f:\Omega\to\mathbb R.
$$

A **random function**, or stochastic process, assigns a random variable to each input. To distinguish the input domain from the probability space, write

$$
F:\Omega\times S\to\mathbb R,
$$

where $(S,\mathcal F,\mathbb P)$ is a probability space. There are two ways to hold one argument fixed:

- Fix an input $x\in\Omega$: $F(x,\cdot)$ is a random variable, usually abbreviated $F(x)$.
- Fix a random outcome $\omega\in S$: $x\mapsto F(x,\omega)$ is one deterministic **sample function**, also called a sample path or realization.

Thus, drawing a random function means drawing an entire collection of mutually related values, not drawing each value independently. In particular, evaluating one realization twice at the same input gives the same value. Additional measurement noise is a separate modeling ingredient.

For example, let $A$ and $B$ be independent standard normal random variables and define

$$
F(x)=A+Bx,\qquad x\in\mathbb R.
$$

One draw of $(A,B)$ produces a whole straight line. Values at different inputs are dependent because they share the same random intercept and slope.

Formally, a stochastic process is a family $\{F(x):x\in\Omega\}$ of random variables. Statements about continuous or differentiable sample functions require additional regularity assumptions; they do not follow just from calling the object a random function.

## 2. The mean function

Assume $\mathbb E[F(x)^2]<\infty$ for every input. The mean function is

$$
\boxed{m(x)=\mathbb E[F(x)].}
$$

This averages over random realizations **at a fixed input**, not over the input domain or the training dataset.

For the random line above, $m(x)=0$. This does not mean that a typical sampled line is identically zero. Positive and negative realizations cancel in expectation.

A nonzero mean can encode a baseline trend. Writing

$$
F(x)=m(x)+Z(x),\qquad \mathbb E[Z(x)]=0,
$$

separates that trend from the random deviation around it.

## 3. The covariance kernel

The covariance kernel is the function on pairs of inputs defined by

$$
\boxed{K(x,z)=\operatorname{Cov}(F(x),F(z))
=\mathbb E[(F(x)-m(x))(F(z)-m(z))].}
$$

It describes how deviations at two inputs vary together:

- $K(x,x)=\operatorname{Var}(F(x))\geq0$ is the marginal variance.
- Positive covariance means the deviations tend to have the same sign; negative covariance means they tend to have opposite signs.
- Zero covariance means uncorrelated, not necessarily independent. For jointly Gaussian variables, zero covariance does imply independence.

When both variances are positive, correlation is the normalized quantity

$$
\rho(x,z)=\frac{K(x,z)}{\sqrt{K(x,x)K(z,z)}}\in[-1,1].
$$

A covariance kernel need not itself be normalized, and its off-diagonal values need not be positive. Covariance symmetry gives $K(x,z)=K(z,x)$, and the Cauchy–Schwarz inequality gives

$$
|K(x,z)|^2\leq K(x,x)K(z,z).
$$

For the random line,

$$
K(x,z)=\mathbb E[(A+Bx)(A+Bz)]=1+xz.
$$

Here independence and zero means eliminate the cross terms. Its variance is $K(x,x)=1+x^2$.

## 4. Why is a covariance kernel positive semidefinite?

A symmetric kernel is **positive semidefinite (PSD)** if, for every finite choice of inputs $x_1,\ldots,x_n$ and real coefficients $c_1,\ldots,c_n$,

$$
\sum_{i,j=1}^n c_ic_jK(x_i,x_j)\geq0.
$$

To prove this for a covariance kernel, consider the random variable

$$
U=\sum_{i=1}^n c_i(F(x_i)-m(x_i)).
$$

It has mean zero. Expanding its variance gives

$$
\begin{aligned}
0\leq\operatorname{Var}(U)
&=\mathbb E\left[\left(\sum_i c_i(F(x_i)-m(x_i))\right)^2\right]\\
&=\sum_{i,j}c_ic_j\mathbb E[(F(x_i)-m(x_i))(F(x_j)-m(x_j))]\\
&=\sum_{i,j}c_ic_jK(x_i,x_j).
\end{aligned}
$$

**PSD is the requirement that every finite linear combination of process values has nonnegative variance.**

For selected inputs, the Gram matrix

$$
G_{ij}=K(x_i,x_j),\qquad G\in\mathbb R^{n\times n},
$$

is their covariance matrix, and the same statement becomes $c^TGc\geq0$. The kernel is a function on all pairs of inputs; $G$ is one finite matrix obtained from it.

PSD does not mean invertible. In the random-line example,

$$
G=\mathbf 1\mathbf 1^T+vv^T,\qquad v=(x_1,\ldots,x_n)^T,
$$

so its rank is at most two, regardless of the number of observations. All process values depend on only two random coefficients. Strict positive definiteness would require a positive quadratic form for every nonzero coefficient vector at distinct inputs, a stronger property.

## 5. Why does the same condition appear in RKHS theory?

In an RKHS $H_K$, the reproducing property gives

$$
K(x,z)=\langle K(\cdot,x),K(\cdot,z)\rangle_{H_K}.
$$

Consequently,

$$
\sum_{i,j}c_ic_jK(x_i,x_j)
=\left\|\sum_i c_iK(\cdot,x_i)\right\|_{H_K}^2\geq0.
$$

The same quadratic form is therefore:

- a **variance** in the random-function interpretation;
- a **squared norm** in the RKHS interpretation.

Conversely, every real symmetric PSD kernel defines a unique associated RKHS. It also defines the covariance of a zero-mean Gaussian process through consistent finite-dimensional Gaussian distributions. This existence statement does not guarantee continuous sample paths.

These are two constructions from the same kernel—not a claim that random sample functions must belong to that RKHS.

## 6. What makes a process Gaussian?

A process is a **Gaussian process (GP)** if every finite vector of its values has a multivariate Gaussian distribution:

$$
(F(x_1),\ldots,F(x_n))^T\sim\mathcal N(m_X,G),
\qquad m_X=(m(x_1),\ldots,m(x_n))^T.
$$

We write $F\sim\mathcal{GP}(m,K)$. Singular covariance matrices are allowed: a Gaussian distribution may be concentrated on a lower-dimensional subspace.

For a GP, the mean and covariance specify all finite-dimensional distributions. For a general random function, they do not determine its full probability law. Gaussianity is an additional modeling assumption, not a consequence of covariance being PSD.

For example, replacing the independent normal coefficients of the random line by independent variables taking $+1$ and $-1$ with equal probabilities preserves its zero mean and kernel $1+xz$, but does not give a GP.

## 7. From a prior random function to predictions

Suppose

$$
F\sim\mathcal{GP}(0,K),\qquad y_i=F(x_i)+\varepsilon_i,
$$

where the errors are independent $\mathcal N(0,\sigma^2)$ variables and are independent of the entire process. Treat the training inputs as fixed and take $\sigma^2>0$ here.

Let $y\in\mathbb R^n$ be the observed outputs. Then

$$
\operatorname{Cov}(y)=C:=G+\sigma^2I_n.
$$

Independent noise adds variance only to the diagonal. Since $G$ is PSD and $\sigma^2>0$, $C$ is positive definite and invertible.

At a query input $x$, define

$$
k(x)=\begin{pmatrix}K(x,x_1)\\\vdots\\K(x,x_n)\end{pmatrix}\in\mathbb R^n.
$$

This vector appears for a specific probabilistic reason:

$$
\operatorname{Cov}(y_i,F(x))=K(x_i,x)=K(x,x_i).
$$

Thus, $k(x)$ collects the cross-covariances between the unknown value and the observations. The joint distribution is

$$
\begin{pmatrix}y\\F(x)\end{pmatrix}
\sim\mathcal N\left(0,
\begin{pmatrix}C&k(x)\\k(x)^T&K(x,x)\end{pmatrix}\right).
$$

### Deriving the conditional mean and variance

Set $a=C^{-1}k(x)$ and define the residual $R=F(x)-a^Ty$. Then

$$
\operatorname{Cov}(y,R)=k(x)-Ca=0.
$$

Because $R$ and $y$ are jointly Gaussian, they are independent. Therefore conditioning on $y$ leaves the residual's zero mean and variance unchanged, yielding

$$
\boxed{\mathbb E[F(x)\mid y]=k(x)^TC^{-1}y,}
$$

$$
\boxed{\operatorname{Var}(F(x)\mid y)=K(x,x)-k(x)^TC^{-1}k(x).}
$$

Indeed, expanding $\operatorname{Var}(R)$ gives $K(x,x)-2a^Tk(x)+a^TCa$, which simplifies to the displayed variance. Its nonnegativity follows because it is a variance.

The posterior mean is a deterministic fitted function once the data and hyperparameters are fixed. It is optimal under conditional squared-error loss: for any scalar prediction $b$,

$$
\mathbb E[(F(x)-b)^2\mid y]
=\operatorname{Var}(F(x)\mid y)+(\mathbb E[F(x)\mid y]-b)^2.
$$

## 8. Exactly how does this connect to kernel ridge regression?

For the same kernel, KRR solves

$$
\min_{f\in H_K}\left\{\sum_{i=1}^n(y_i-f(x_i))^2+\lambda\|f\|_{H_K}^2\right\},\qquad\lambda>0.
$$

The representer theorem reduces this to a kernel expansion. A coefficient vector producing the unique minimizing function is

$$
\alpha=(G+\lambda I_n)^{-1}y,
$$

and prediction is

$$
\boxed{f_\lambda(x)=k(x)^T(G+\lambda I_n)^{-1}y.}
$$

If $G$ is singular, different coefficient vectors can represent the same function; the displayed choice is still valid.

Comparing formulas shows that **KRR equals the zero-mean GP posterior mean when $\lambda=\sigma^2$**, provided both use exactly the same kernel and the sum-of-squares objective above.

Normalization matters. If the KRR loss is instead

$$
\frac1n\sum_i(y_i-f(x_i))^2+\lambda\|f\|_{H_K}^2,
$$

its matrix is $G+n\lambda I_n$, so matching the GP requires $n\lambda=\sigma^2$.

Kernel amplitude matters too. If the GP covariance is $\tau^2K_0$, then its posterior mean can be written

$$
k_0(x)^T\left(G_0+\frac{\sigma^2}{\tau^2}I_n\right)^{-1}y.
$$

Thus, KRR using the base kernel $K_0$ and the sum loss matches with $\lambda=\sigma^2/\tau^2$. It is the noise-to-signal variance ratio that controls shrinkage in this parametrization.

The equivalence is between **point predictors**. KRR by itself does not specify a probability law or calibrated uncertainty; those require additional assumptions such as the GP model.

## 9. Useful qualifications and extensions

### Nonzero mean

For a fixed, known mean function $m$, the GP posterior mean becomes

$$
m(x)+k(x)^TC^{-1}(y-m_X).
$$

The corresponding regularized problem fits a residual $g\in H_K$ to $y_i-m(x_i)$ and predicts $m+g$. The baseline $m$ need not belong to $H_K$. Estimating it introduces further modeling choices.

### Latent values versus noisy future observations

The posterior variance above concerns $F(x)$, the latent function value. For a future measurement $Y_*=F(x)+\varepsilon_*$ with fresh independent noise of variance $\sigma^2$,

$$
\operatorname{Var}(Y_*\mid y)=\operatorname{Var}(F(x)\mid y)+\sigma^2.
$$

### The posterior covariance kernel

Conditioning gives another covariance kernel:

$$
K_{\mathrm{post}}(x,z)=K(x,z)-k(x)^TC^{-1}k(z).
$$

It is PSD because it is a conditional covariance kernel. For fixed kernel parameters, noise variance, and training inputs, it does not depend on the observed output values. If hyperparameters are estimated from outputs, it can depend on them indirectly.

### Noise-free interpolation and the power function

If observations are exact and $G$ is invertible, setting the noise variance to zero gives

$$
\mathbb E[F(x)\mid y]=k(x)^TG^{-1}y=f^*(x).
$$

The posterior variance is

$$
K(x,x)-k(x)^TG^{-1}k(x)=P_X(x)^2,
$$

where $P_X$ is the RKHS power function. For a deterministic target $f\in H_K$ and its minimum-norm interpolant $s_Xf$,

$$
|f(x)-s_Xf(x)|\leq P_X(x)\|f\|_{H_K}.
$$

This deterministic bound and the GP variance use the same expression but make different assumptions. A GP credible interval is not automatically a deterministic error bound for an arbitrary target.

### A GP sample is not generally an RKHS element

For $K(s,t)=\min(s,t)$ on $[0,1]$, the GP is standard Brownian motion. Its RKHS consists of absolutely continuous functions vanishing at zero with square-integrable derivative. Brownian paths almost surely do not lie in this RKHS.

So avoid describing KRR simply as selecting the most probable GP sample by minimizing an RKHS norm. Infinite-dimensional probability has no ordinary Lebesgue density over functions. The posterior-mean identity derived above is precise and needs no such interpretation.

### Modeling uncertainty is not a guarantee of correctness

Posterior variances and credible intervals are conditional on the chosen kernel, mean, noise model, and treatment of hyperparameters. A misspecified model can be confidently wrong. Likewise, choosing an RKHS is an inductive bias, not a guarantee of generalization or of exploiting manifold geometry.

## 10. Summary

| Object | RKHS / KRR interpretation | Random-function / GP interpretation |
| --- | --- | --- |
| $K(x,z)$ | Inner product of evaluation representers | Covariance of latent values |
| $G$ | Training Gram matrix | Covariance of latent training values |
| $k(x)$ | Kernel evaluations used to evaluate an expansion | Cross-covariances with observations |
| Positive semidefiniteness | Squared norms are nonnegative | Variances are nonnegative |
| $k(x)^T(G+\lambda I)^{-1}y$ | Regularized fitted function | Posterior mean when $\lambda=\sigma^2$ |
| $P_X(x)^2$ | Squared norm of an evaluation-representer residual | Noise-free posterior variance |

**The kernel specifies the shared geometry. KRR uses that geometry to select a function by optimization; a GP uses it to model dependence and condition a probability distribution on observations.**
