# From RKHS to kernel machine learning and Gaussian processes

> Source status: Lecture references and numbering below are retained from the supplied text and have not yet been checked against the lecture notes. Section 7 describes a proposed interactive illustration, not an implemented simulation.

The connection between reproducing kernel Hilbert spaces (RKHS) and Gaussian processes (GPs) is particularly interesting because the same mathematical object—a positive-definite kernel—can be understood from two different perspectives:

1. Functional analysis: A kernel defines a Hilbert space of functions, and we seek a function that minimizes a norm.
2. Probability: A kernel defines the covariance of a Gaussian process, and we use observed data to infer an unknown function.

Remarkably, both approaches lead to essentially the same prediction formula.

Your Lecture 2 already develops most of the functional-analytic machinery needed for the first perspective. We will build on its results and then introduce the additional statistical ideas.

## 1. Starting with what you studied in Lecture 2

Recall Definition 2.5. A Hilbert space $H$ of functions is an RKHS if, for every $x$, the point-evaluation functional

$$
L_x(f)=f(x)
$$

is bounded.

By the Riesz representation theorem, there exists a unique element $K(\cdot,x)\in H$ such that

$$
\boxed{f(x)=\langle f,K(\cdot,x)\rangle_H.}
$$

This is the reproducing property.

It is important because it converts evaluating a function at a point into an inner product in a Hilbert space.

Furthermore, the kernel satisfies

$$
K(x,y)=\langle K(\cdot,x),K(\cdot,y)\rangle_H.
$$

Therefore, $K(x,y)$ measures the inner product between two evaluation representers.

### The optimal interpolation problem

Suppose we have observations

$$
f(x_i)=y_i,\qquad i=1,\ldots,n.
$$

In Lecture 2, Definition 2.8 considers

$$
\boxed{\min_{f\in H}\|f\|_H^2 \quad\text{subject to}\quad f(x_i)=y_i.}
$$

By Theorem 2.9, assuming the kernel matrix is invertible, the unique solution is

$$
f^*(x)=\sum_{i=1}^n\alpha_i K(x,x_i),
$$

where

$$
G\alpha=y,\qquad G_{ij}=K(x_i,x_j).
$$

Thus,

$$
\boxed{f^*(x)=k(x)^TG^{-1}y}
$$

where

$$
k(x)=\begin{pmatrix}K(x,x_1)\\ \vdots\\ K(x,x_n)\end{pmatrix}.
$$

This is already a working kernel-based prediction algorithm.

Given data points and a kernel, construct the Gram matrix, solve a linear system, and evaluate the function wherever you like.

## 2. Why does the solution use only $n$ kernel functions?

This is one of the most fundamental results you've learned, and it is worth seeing why it generalizes to machine learning.

Define the finite-dimensional subspace

$$
V=\operatorname{span}\{K(\cdot,x_1),\ldots,K(\cdot,x_n)\}.
$$

Any $f\in H$ can be decomposed orthogonally as

$$
f=f_V+f_\perp,
$$

where $f_V\in V$ and $f_\perp\in V^\perp$.

For every observed point $x_i$,

$$
f_\perp(x_i)=\langle f_\perp,K(\cdot,x_i)\rangle_H=0.
$$

Therefore,

$$
f(x_i)=f_V(x_i).
$$

The component $f_\perp$ has no effect on the training observations.

But by the Pythagorean theorem,

$$
\|f\|_H^2=\|f_V\|_H^2+\|f_\perp\|_H^2.
$$

Any nonzero $f_\perp$ increases the norm without improving the fit to the observations. Consequently, the minimum-norm solution must satisfy $f_\perp=0$.

This is the key geometric idea behind the representer theorem, which generalizes the argument beyond exact interpolation to regularized learning problems.

**Geometry of minimum-norm interpolation**

Conceptual Hilbert-space geometry. The $V^\perp$ component adds norm but does not change the observed values.

This explains how an optimization problem over an infinite-dimensional space of functions reduces to a finite-dimensional linear algebra problem.

## 3. How does this become machine learning?

So far, the problem assumes that the observations are exact:

$$
f(x_i)=y_i.
$$

In machine learning, we typically observe noisy data:

$$
y_i=f_{\mathrm{true}}(x_i)+\varepsilon_i.
$$

Requiring exact interpolation may lead to overfitting.

Instead, we want to minimize two competing objectives: prediction error and function complexity.

### Kernel ridge regression

Consider

$$
\boxed{\min_{f\in H_K}\left[\sum_{i=1}^n(y_i-f(x_i))^2+\lambda\|f\|_{H_K}^2\right].}
$$

The first term measures the fit to observations. The second term is the regularizer.

Using the orthogonal decomposition argument from Section 2, the minimizer must still belong to

$$
V=\operatorname{span}\{K(\cdot,x_i)\}_{i=1}^n.
$$

Thus,

$$
f_\lambda(x)=\sum_{i=1}^n\alpha_iK(x,x_i).
$$

Writing the objective in terms of $\alpha$,

$$
J(\alpha)=\|y-G\alpha\|_2^2+\lambda\alpha^TG\alpha.
$$

Differentiating,

$$
\nabla_\alpha J=-2G(y-G\alpha)+2\lambda G\alpha.
$$

Setting the derivative equal to zero gives

$$
G(G+\lambda I)\alpha=Gy.
$$

When $G$ is invertible, this simplifies to

$$
\boxed{\alpha=(G+\lambda I)^{-1}y.}
$$

For $\lambda>0$, this coefficient choice also yields the unique minimizing function when $G$ is only positive semidefinite.

The resulting prediction is

$$
\boxed{f_\lambda(x)=k(x)^T(G+\lambda I)^{-1}y.}
$$

Notice what happened: the interpolation formula from Lecture 2 barely changed. We simply added a regularization term to the Gram matrix.

This is a widely used principle in machine learning: instead of directly limiting a model's dimension, penalize its complexity in a suitable function space.

## 4. Gaussian processes: a probabilistic view of the same problem

Now comes the more surprising connection.

Instead of defining the function by an optimization problem, suppose we treat the unknown function $f$ as a random function.

A Gaussian process is a collection of random variables $\{f(x):x\in\Omega\}$ such that for any finite set $x_1,\ldots,x_n$,

$$
\begin{pmatrix}f(x_1)\\ \vdots\\ f(x_n)\end{pmatrix}
$$

has a multivariate Gaussian distribution.

We write

$$
f\sim\mathcal{GP}(m,K),
$$

where $m(x)$ is the mean function and $K(x,x')$ is the covariance kernel:

$$
m(x)=\mathbb E[f(x)],
$$

$$
K(x,x')=\operatorname{Cov}(f(x),f(x')).
$$

For simplicity, assume $m(x)=0$.

### Why does a kernel appear again?

The Gram matrix is

$$
G_{ij}=K(x_i,x_j).
$$

Because $G$ is a covariance matrix, it must be symmetric positive semidefinite.

But that is precisely the condition that appeared in Lecture 2, Theorem 2.7 (Moore–Aronszajn).

So the same positive semidefinite kernel defines two closely related mathematical objects:

### RKHS viewpoint

A Hilbert space of functions $H_K$, with a norm describing function complexity.

$$
\|f\|_{H_K}^2
$$

Used for deterministic approximation and regularization.

### Gaussian process viewpoint

A probability distribution over functions, with covariance describing dependence.

$$
\operatorname{Cov}(f(x),f(x'))=K(x,x')
$$

Used for probabilistic prediction and uncertainty quantification.

The kernel has two interpretations, but the resulting prediction formulas are intimately connected.

## 5. Deriving Gaussian process regression

Suppose we observe

$$
y_i=f(x_i)+\varepsilon_i,
$$

where the independent noise variables satisfy

$$
\varepsilon_i\sim\mathcal N(0,\sigma^2).
$$

Let

$$
y=(y_1,\ldots,y_n)^T.
$$

Under our zero-mean GP prior,

$$
y\sim\mathcal N(0,G+\sigma^2I).
$$

Now choose a new location $x_*$, where we want to predict $f(x_*)$.

The joint distribution is

$$
\begin{pmatrix}y\\ f(x_*)\end{pmatrix}
\sim\mathcal N\left(0,
\begin{pmatrix}
G+\sigma^2I & k(x_*)\\
k(x_*)^T & K(x_*,x_*)
\end{pmatrix}\right).
$$

A standard result about conditioning multivariate Gaussian distributions gives

$$
f(x_*)\mid y\sim\mathcal N(\mu_*,s_*^2),
$$

where

$$
\boxed{\mu_*=k(x_*)^T(G+\sigma^2I)^{-1}y}
$$

and

$$
\boxed{s_*^2=K(x_*,x_*)-k(x_*)^T(G+\sigma^2I)^{-1}k(x_*).}
$$

These are the two central equations in Gaussian process regression.

The first is the prediction, and the second quantifies uncertainty about the latent function value.

### Compare the formulas

| Method | Prediction |
| --- | --- |
| RKHS interpolation | $k(x)^TG^{-1}y$ |
| Kernel ridge regression | $k(x)^T(G+\lambda I)^{-1}y$ |
| Gaussian process posterior mean | $k(x)^T(G+\sigma^2I)^{-1}y$ |

For the objectives and noise model defined above, setting

$$
\boxed{\lambda=\sigma^2}
$$

makes kernel ridge regression and GP regression produce exactly the same prediction.

But Gaussian process regression gives us something extra: a posterior predictive distribution, rather than only a point estimate.

## 6. The connection to the power function in Lecture 2

Recall equation (2.3) in your lecture:

$$
P_X(x)^2=K(x,x)-k(x)^TG^{-1}k(x).
$$

Your professor derives this quantity using Hilbert-space geometry. It measures the norm of the component of $K(\cdot,x)$ orthogonal to the span of the observed evaluation representers.

Now compare that with the GP posterior variance in the noise-free case:

$$
\operatorname{Var}(f(x)\mid y)=K(x,x)-k(x)^TG^{-1}k(x).
$$

Therefore,

$$
\boxed{P_X(x)^2=\operatorname{Var}(f(x)\mid y)}
$$

under the noise-free GP model with the same kernel.

This is a particularly elegant result.

One mathematical quantity has two interpretations:

- Deterministic: $P_X(x)$ controls the worst-case interpolation error over a norm-bounded class of functions.
- Probabilistic: $P_X(x)$ is the posterior standard deviation of the unknown function in a Gaussian process model.

Neither interpretation is inherently more correct. They answer different questions using the same kernel geometry.

## 7. Interactive example: how observations change GP predictions

The following example shows Gaussian process regression with a squared-exponential kernel. You can change the observation noise and length scale to see how both the predicted function and its uncertainty change.

**Gaussian process regression — Illustrative simulation**

Proposed controls (not yet implemented):

- Kernel length scale $\ell$: 0.80
- Noise standard deviation $\sigma$: 0.15
- Number of observations: 3 points, 5 points, or 8 points

The shaded region is approximately $\mu(x)\pm1.96s(x)$, representing a pointwise 95% posterior credible interval for the latent function, assuming the specified GP model. Observations are fixed illustrative data.

Try setting the noise to zero. The posterior mean will interpolate the observations, while its uncertainty vanishes there. As you increase the noise, the curve no longer needs to pass through the observations exactly, and uncertainty remains at those locations.

When you increase the kernel length scale, you are expressing a prior preference for longer-range correlations and smoother variation.

## 8. A particularly useful example from your lecture: the Brownian motion kernel

Lecture 2, Example 2.10, constructs the RKHS

$$
H=\left\{f(x)=\int_0^x g(t)\,dt:g\in L^2[0,1]\right\}
$$

with

$$
\|f\|_H^2=\int_0^1|f'(t)|^2\,dt.
$$

It then proves that its reproducing kernel is

$$
\boxed{K(s,t)=\min(s,t).}
$$

Now here is the additional probabilistic connection: this is exactly the covariance kernel of standard Brownian motion.

For standard Brownian motion $B(t)$,

$$
\mathbb E[B(s)B(t)]=\min(s,t).
$$

Consequently, the same kernel describes both the RKHS in your lecture and a Gaussian process.

In the RKHS viewpoint, minimizing

$$
\int_0^1|f'(t)|^2\,dt
$$

under interpolation constraints produces a piecewise linear function.

In the GP viewpoint, conditioning Brownian motion on its values at specified times produces a posterior mean that is piecewise linear between observed times.

These are two routes to the same estimator.

### One subtle but important distinction

This does not mean that a typical Brownian motion sample path belongs to the RKHS.

In fact, standard Brownian motion sample paths almost surely do not belong to the associated RKHS: they are almost surely nowhere differentiable, whereas the RKHS consists of absolutely continuous functions with square-integrable derivatives and $f(0)=0$.

So the RKHS describes a useful geometry associated with the Gaussian process—not necessarily the set containing its typical random realizations.

This distinction becomes important in theoretical work on Gaussian measures, inverse problems, and Bayesian inference.

## 9. How is this used in real research?

Three applications illustrate why the connection matters.

**Bayesian optimization.** Imagine that evaluating $f(x)$ requires running an expensive simulation. A GP surrogate can predict the function at unobserved inputs and estimate uncertainty. An acquisition rule then chooses the next point to evaluate, balancing promising predictions and uncertainty. This is used for expensive black-box optimization and hyperparameter tuning.

**System identification.** Suppose an unknown dynamical system is modeled as

$$
x_{t+1}=g(x_t,u_t)+\varepsilon_t.
$$

A GP can be used to learn the unknown transition function $g$ from observed trajectories. Its posterior variance can indicate which state–input combinations are poorly explored. In practical systems, safety and uncertainty calibration require additional care.

**Inverse problems and scientific machine learning.** Suppose we observe measurements

$$
y_i=L_i(f)+\varepsilon_i
$$

where the $L_i$ are bounded linear functionals, such as integrals, averages, or suitable point evaluations.

The Riesz representation ideas from Lecture 2 generalize GP regression beyond point samples. With a GP covariance kernel, one can form

$$
\operatorname{Cov}(L_i(f),L_j(f))=L_i^{(x)}L_j^{(x')}K(x,x')
$$

when these operations are well-defined.

This allows inference from indirect observations, which is central to many scientific inverse problems.

## 10. What should you take away from Lectures 1–3?

The chain of connections is:

$$
\boxed{\begin{gathered}
\text{Hilbert-space geometry}\\
\Downarrow\\
\text{Riesz representation and RKHS}\\
\Downarrow\\
\text{Finite kernel expansion}\\
\Downarrow\\
\text{Minimum-norm interpolation}\\
\Downarrow\\
\text{Kernel regularization and GP regression}
\end{gathered}}
$$

The first four stages are substantially covered by your lectures. The last stage adds statistical learning, Gaussian conditioning, and noise models.

For a mathematically rigorous MSc research direction, this framework is valuable because it connects functional analysis, optimization, probability, and algorithms. In particular, the theory provides several different ways to study a predictor: approximation error, worst-case recovery error, regularization, and posterior uncertainty.

The single most important connection to remember is

$$
\boxed{\underbrace{P_X(x)^2}_{\text{deterministic power function}}=\underbrace{\operatorname{Var}(f(x)\mid y)}_{\text{noise-free GP posterior variance}}}
$$

under the corresponding kernel assumptions.

It is a concrete example of deterministic approximation theory and Bayesian inference arriving at the same mathematical expression from different starting points.
