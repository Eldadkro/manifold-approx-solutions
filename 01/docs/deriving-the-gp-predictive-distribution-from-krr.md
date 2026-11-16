# Deriving the GP predictive distribution from kernel ridge regression

This note expands the derivation in Section 5 of [From RKHS to kernel machine learning and Gaussian processes](from-rkhs-to-kernel-machine-learning-and-gaussian-processes.md), explaining step by step the Gaussian-process (GP) conditional distribution

$$
\boxed{f(x_*)\mid y\sim\mathcal N(\mu_*,s_*^2).}
$$

**Important distinction:** KRR alone defines a deterministic optimization problem, not a probability distribution. To derive this distribution, we introduce a GP prior and independent Gaussian observation noise. Under matching parameter conventions, the resulting posterior mean equals the KRR predictor; the probabilistic model additionally supplies a posterior variance.

## 1. Start with the KRR predictor

Let $k:\mathcal X\times\mathcal X\to\mathbb R$ be a symmetric positive-semidefinite kernel with RKHS $\mathcal H_k$. Fix training inputs $x_1,\ldots,x_n$ and responses $y=(y_1,\ldots,y_n)^\top\in\mathbb R^n$.

Following the main KRR note, define

$$
K_{ij}=k(x_i,x_j),\qquad K\in\mathbb R^{n\times n}.
$$

### Notation correspondence with the source note

The source note uses $K$ for the kernel, $G$ for its Gram matrix, and $k(x_*)$ for the query vector. Here we retain the notation of the companion KRR note: $k$ for the kernel, $K$ for the Gram matrix, and $k_*$ for the query vector. Thus the correspondence is

| Source note | This derivation |
|---|---|
| Kernel $K(x,z)$ | Kernel $k(x,z)$ |
| Gram matrix $G$ | Gram matrix $K$ |
| Query vector $k(x_*)$ | Query vector $k_*$ |

In the source notation, the result derived below is exactly

$$
\mu_*=k(x_*)^\top(G+\sigma^2I_n)^{-1}y,
\qquad
s_*^2=K(x_*,x_*)-k(x_*)^\top(G+\sigma^2I_n)^{-1}k(x_*).
$$

The source uses a **sum** of squared errors, so its regularization parameter matches $\sigma^2$ directly. The companion KRR note and the objective below use a **mean** of squared errors, giving $n\lambda=\sigma^2$ instead. These are equivalent conventions, not conflicting results.

For $\lambda>0$, the KRR objective is

$$
\widehat f=\arg\min_{f\in\mathcal H_k}
\left\{\frac1n\sum_{i=1}^n(f(x_i)-y_i)^2+\lambda\|f\|_{\mathcal H_k}^2\right\}.
$$

At a new input $x_*$, define

$$
k_*=(k(x_1,x_*),\ldots,k(x_n,x_*))^\top\in\mathbb R^n,
\qquad k_{**}=k(x_*,x_*).
$$

The main note derives

$$
\widehat f(x_*)=k_*^\top(K+n\lambda I_n)^{-1}y.
$$

We now construct a probabilistic model that yields the same mean prediction.

## 2. Add a zero-mean GP prior and a noise model

Assume

$$
f\sim\operatorname{GP}(0,k).
$$

By definition, every finite collection of function evaluations is jointly Gaussian. In particular,

$$
f_X=(f(x_1),\ldots,f(x_n))^\top\sim\mathcal N(0,K).
$$

Assume observations satisfy

$$
y=f_X+\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,\sigma^2I_n),\qquad \sigma^2>0,
$$

with $\varepsilon$ independent of the entire random function $f$. Inputs, kernel parameters, and noise variance are treated as fixed throughout.

The prior's zero mean does not assert that the unknown function is identically zero. It specifies the expected function value before conditioning on the observations.

### Why is $y$ Gaussian?

It is the sum of two independent Gaussian vectors. Therefore it is Gaussian, with

$$
\mathbb E[y]=0,
\qquad
\operatorname{Cov}(y)=K+\sigma^2I_n=:C.
$$

Thus

$$
\boxed{y\sim\mathcal N(0,C).}
$$

This is an exact consequence of the modeling assumptions, not a central-limit approximation. More explicitly, for every $a\in\mathbb R^n$,

$$
a^\top y=a^\top f_X+a^\top\varepsilon
\sim\mathcal N(0,a^\top Ka+\sigma^2a^\top a).
$$

Every linear combination is Gaussian, which characterizes a jointly Gaussian vector. The observations need not be independent: their covariance matrix generally has nonzero off-diagonal entries.

Distinguish the conditional and marginal distributions:

$$
y\mid f_X\sim\mathcal N(f_X,\sigma^2I_n),
\qquad
y\sim\mathcal N(0,K+\sigma^2I_n).
$$

The first retains only observation-noise uncertainty; the second also includes prior uncertainty about $f_X$.

## 3. Include the new function value in the joint distribution

Write $f_*=f(x_*)$. The GP prior gives

$$
\begin{pmatrix}f_X\\f_*\end{pmatrix}
\sim\mathcal N\left(
0,
\begin{pmatrix}K&k_*\\k_*^\top&k_{**}\end{pmatrix}
\right).
$$

Adding independent Gaussian noise to the training evaluations preserves joint Gaussianity. The cross-covariance is

$$
\operatorname{Cov}(y_i,f_*)
=\operatorname{Cov}(f(x_i)+\varepsilon_i,f(x_*))
=k(x_i,x_*),
$$

since the observation noise is independent of $f$. Consequently,

$$
\boxed{
\begin{pmatrix}y\\f_*\end{pmatrix}
\sim\mathcal N\left(
0,
\begin{pmatrix}C&k_*\\k_*^\top&k_{**}\end{pmatrix}
\right).
}
$$

The block dimensions are $n\times n$, $n\times1$, $1\times n$, and $1\times1$, respectively. The vector $k_*$ describes how the observations relate to the new function value; the marginal distribution of $y$ alone would not determine a prediction for $f_*$.

## 4. Derive the conditional distribution using an independent residual

Rather than quoting a Gaussian conditioning formula, construct the residual

$$
r=f_*-k_*^\top C^{-1}y.
$$

The inverse exists because, for every nonzero $v\in\mathbb R^n$,

$$
v^\top Cv=v^\top Kv+\sigma^2\|v\|_2^2>0.
$$

Thus $C$ is positive definite even when $K$ is singular.

### Step 1: show that the residual is independent of the observations

Its cross-covariance with $y$ is

$$
\begin{aligned}
\operatorname{Cov}(y,r)
&=\operatorname{Cov}(y,f_*)-\operatorname{Cov}(y,y)C^{-1}k_*\\
&=k_*-CC^{-1}k_*\\
&=0.
\end{aligned}
$$

The pair $(y,r)$ is jointly Gaussian, since it is a linear transformation of $(y,f_*)$. For jointly Gaussian variables, zero cross-covariance implies independence. Therefore $r$ is independent of $y$.

This implication is specific to the Gaussian setting: uncorrelated random variables need not be independent in general.

### Step 2: compute the residual distribution

The residual is Gaussian with mean zero. Using symmetry of $C^{-1}$, its variance is

$$
\begin{aligned}
\operatorname{Var}(r)
&=\operatorname{Var}(f_*)
-2\operatorname{Cov}(f_*,k_*^\top C^{-1}y)
+\operatorname{Var}(k_*^\top C^{-1}y)\\
&=k_{**}-2k_*^\top C^{-1}k_*
+k_*^\top C^{-1}CC^{-1}k_*\\
&=k_{**}-k_*^\top C^{-1}k_*.
\end{aligned}
$$

Define

$$
s_*^2=k_{**}-k_*^\top C^{-1}k_*.
$$

This quantity is nonnegative because it is a variance. If it is zero, the Gaussian distribution is understood as a point mass.

### Step 3: condition on the measured value of $y$

We have the decomposition

$$
f_*=k_*^\top C^{-1}y+r,
\qquad r\perp\!\!\!\perp y.
$$

Once $y$ is observed, the first term is known. Independence means that conditioning on $y$ does not change the distribution of $r$. Hence

$$
\boxed{
 f(x_*)\mid y\sim\mathcal N(\mu_*,s_*^2),
}
$$

where

$$
\boxed{\mu_*=k_*^\top(K+\sigma^2I_n)^{-1}y,}
$$

and

$$
\boxed{s_*^2=k(x_*,x_*)-k_*^\top(K+\sigma^2I_n)^{-1}k_*.}
$$

The notation suppresses conditioning on the fixed training inputs and model parameters.

## 5. Recover KRR and its kernel-span structure

Compare the posterior mean with the KRR predictor:

$$
\mu_*=k_*^\top(K+\sigma^2I_n)^{-1}y,
\qquad
\widehat f(x_*)=k_*^\top(K+n\lambda I_n)^{-1}y.
$$

For the same kernel, they agree when

$$
\boxed{\sigma^2=n\lambda.}
$$

The factor $n$ comes from the mean-squared-loss convention in the KRR note. For an unnormalized sum of squared errors with penalty weight $\eta$, the corresponding identification is $\sigma^2=\eta$.

With $\alpha=(K+\sigma^2I_n)^{-1}y$, the posterior mean function is

$$
m_D(x)=\sum_{i=1}^n\alpha_i k(x_i,x).
$$

It therefore lies in the training-kernel span. This statement applies to the **posterior mean**, not to all functions described by the posterior GP. In general, GP sample paths are not elements of the corresponding RKHS; the equivalence here does not require them to be.

See also [Why kernel ridge regression stays in the kernel span](why-kernel-ridge-regression-stays-in-the-kernel-span.md) for the deterministic projection argument.

## 6. Interpret the posterior variance

The formula has the form

$$
\underbrace{s_*^2}_{\text{remaining uncertainty}}
=
\underbrace{k_{**}}_{\text{prior uncertainty}}
-
\underbrace{k_*^\top C^{-1}k_*}_{\text{reduction from observing the data}}.
$$

Because $C^{-1}$ is positive definite,

$$
0\le s_*^2\le k_{**}.
$$

If $k_*=0$, then $f_*$ is independent of $y$ under the joint Gaussian model, and the posterior equals the prior:

$$
f_*\mid y\sim\mathcal N(0,k_{**}).
$$

For fixed inputs and hyperparameters, the variance does not depend on the numerical response values $y$. The mean does. If hyperparameters are fitted using the responses, the resulting fitted variance can depend on them indirectly.

This is model-based uncertainty, not an unconditional guarantee of prediction accuracy or calibration on real data.

## 7. Predicting a function value versus a noisy observation

The derived variance concerns the latent value $f(x_*)$. If instead we want a future measurement

$$
y_*=f(x_*)+\varepsilon_*,
$$

where $\varepsilon_*\sim\mathcal N(0,\sigma^2)$ is independent of the function and training noise, then

$$
\boxed{y_*\mid y\sim\mathcal N(\mu_*,s_*^2+\sigma^2).}
$$

Do not add this extra noise variance when the prediction target is $f(x_*)$ itself.

## 8. Computation

The inverse notation describes the mathematics; implementations should solve linear systems. Solve

$$
C\alpha=y,\qquad Cv=k_*.
$$

Then compute

$$
\mu_*=k_*^\top\alpha,
\qquad s_*^2=k_{**}-k_*^\top v.
$$

A Cholesky factorization of $C$ can be reused for both solves.

## Related notes and scope

- [Kernel ridge regression](kernel-ridge-regression.md) — deterministic objective, representer argument, and linear system.
- [What is a Gaussian process?](what-is-a-gaussian-process.md) — the prior over function evaluations.
- [What is a posterior Gaussian process?](what-is-a-posterior-gp.md) — conditioning and the full posterior mean and covariance functions.
- [Documentation index](index.md).

This is a self-contained mathematical derivation under explicit Gaussian assumptions. It does not report an experiment or claim alignment with specific course lectures.
