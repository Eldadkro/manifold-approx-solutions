# What is a Gaussian process?

A **Gaussian process (GP)** is a probabilistic model for an unknown function. Instead of selecting one function immediately, it describes uncertainty about the values that function might take.

For example, suppose $f(t)$ is an unknown temperature curve. A GP lets us express which curves are plausible before observing temperatures, then update those beliefs after measurements.

## 1. From a Gaussian variable to a random function

A scalar Gaussian random variable models one uncertain number:

$$
Z\sim\mathcal N(\mu,\sigma^2).
$$

A multivariate Gaussian models a finite vector of uncertain numbers:

$$
z\sim\mathcal N(m,\Sigma).
$$

Here $m$ contains the means, while $\Sigma$ contains variances and covariances. Covariance describes how two quantities vary together.

A GP extends this idea to a collection of random variables indexed by inputs $x\in\Omega$:

$$
\{f(x):x\in\Omega\}.
$$

**Definition.** This collection is a Gaussian process if, for every finite selection $x_1,\ldots,x_n$, the vector

$$
\begin{pmatrix}f(x_1)\\ \vdots\\ f(x_n)\end{pmatrix}
$$

has a multivariate Gaussian distribution, allowing degenerate Gaussian distributions.

The inputs can be times, spatial locations, feature vectors, or points on another domain. A GP is not restricted to time series.

## 2. Mean and covariance kernel

We write

$$
f\sim\mathcal{GP}(m,K),
$$

where

$$
m(x)=\mathbb E[f(x)]
$$

and

$$
K(x,x')=\operatorname{Cov}(f(x),f(x')).
$$

For any finite input set $X=(x_1,\ldots,x_n)$, define

$$
m_X=(m(x_1),\ldots,m(x_n))^T,\qquad G_{ij}=K(x_i,x_j).
$$

Then

$$
f_X\sim\mathcal N(m_X,G).
$$

In particular:

- $m(x)$ is the expected function value at $x$.
- $K(x,x)$ is its prior variance.
- $K(x,x')$ describes the dependence between values at two inputs.

For a real-valued GP, $K$ must be symmetric and positive semidefinite: every finite Gram matrix $G$ must satisfy $a^TGa\geq0$ for every $a\in\mathbb R^n$.

Conversely, a mean function and such a kernel specify a GP through consistent finite-dimensional Gaussian distributions. Additional kernel assumptions are needed to establish properties such as continuous sample paths.

## 3. Example: the squared-exponential kernel

For inputs in $\mathbb R^d$, a common choice is

$$
K(x,x')=\tau^2\exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right),
\qquad \tau>0,\quad\ell>0.
$$

- $\tau^2$ sets the prior variance of the function at each input.
- $\ell$ is the length scale: it controls how quickly correlation decreases with distance.

Nearby inputs have strongly correlated function values. A larger $\ell$ favors variation over longer distances; a smaller $\ell$ allows more rapid variation. This kernel encodes very strong smoothness assumptions, regardless of the particular positive length scale.

The kernel is a modeling choice, not a fact inferred automatically from the definition of a GP.

## 4. Prior, observations, and posterior

Before observing data, the GP is a **prior** over the unknown function.

Suppose we observe

$$
y_i=f(x_i)+\varepsilon_i,\qquad
\varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2),
$$

with noise independent of $f$. Conditioning on these data produces a **posterior** GP. See [What is a posterior?](what-is-a-posterior.md).

Let

$$
y=(y_1,\ldots,y_n)^T,\qquad
k(x)=(K(x_1,x),\ldots,K(x_n,x))^T,\qquad
A=G+\sigma^2I_n.
$$

Assuming $A$ is invertible, the posterior mean is

$$
\boxed{m_D(x)=m(x)+k(x)^TA^{-1}(y-m_X).}
$$

The posterior covariance kernel is

$$
\boxed{K_D(x,x')=K(x,x')-k(x)^TA^{-1}k(x').}
$$

Thus,

$$
f\mid D\sim\mathcal{GP}(m_D,K_D),
$$

where $D=\{(x_i,y_i)\}_{i=1}^n$ and the inputs are treated as fixed.

For $\sigma^2>0$, $A$ is invertible even when $G$ is singular. With zero noise, these formulas require invertible $G$; redundant noiseless observations need separate treatment.

In code, solve linear systems involving $A$ rather than explicitly computing its inverse.

## 5. What does a GP prediction contain?

At a new input $x_*$, the latent function value satisfies

$$
f(x_*)\mid D\sim\mathcal N\bigl(m_D(x_*),K_D(x_*,x_*)\bigr).
$$

The posterior mean is a point prediction. The variance describes remaining uncertainty under the model.

If we instead predict a new noisy observation $y_*=f(x_*)+\varepsilon_*$, with independent noise of the same variance, then

$$
y_*\mid D\sim\mathcal N\bigl(m_D(x_*),K_D(x_*,x_*)+\sigma^2\bigr).
$$

**Uncertainty about the function and uncertainty about a future measurement are different.**

With fixed kernel and noise parameters, posterior covariance depends on observation locations, not observed values. If those parameters are learned from the outputs, uncertainty can depend on the outputs through the fitted parameters as well.

## 6. Connection to RKHS

The same positive-semidefinite kernel defines an RKHS $H_K$ and a GP covariance model, but these are different mathematical objects:

| RKHS | Gaussian process |
| --- | --- |
| A Hilbert space of deterministic functions | A collection of jointly Gaussian random variables |
| Its norm expresses a notion of function complexity | Its covariance expresses probabilistic dependence |
| Used for interpolation and regularized optimization | Used for Bayesian prediction and uncertainty quantification |

For a zero-mean GP, its posterior mean equals kernel ridge regression with objective

$$
\sum_{i=1}^n(y_i-f(x_i))^2+\lambda\|f\|_{H_K}^2
$$

when $\lambda=\sigma^2>0$ and both use the same kernel. If the loss is averaged by dividing by $n$, the matching convention is $n\lambda=\sigma^2$ instead.

This agreement does **not** mean GP sample paths belong to the RKHS. For example, Brownian motion has covariance $K(s,t)=\min(s,t)$, but its sample paths almost surely do not belong to the associated RKHS of absolutely continuous functions with square-integrable derivatives and value zero at the origin.

For the detailed derivation, see [From RKHS to kernel machine learning and Gaussian processes](from-rkhs-to-kernel-machine-learning-and-gaussian-processes.md).

## Takeaway

A GP models an unknown function by specifying joint Gaussian distributions for every finite collection of its values. Its kernel determines how information at observed inputs influences predictions elsewhere. Conditioning gives both a predicted function and a model-dependent description of uncertainty.
