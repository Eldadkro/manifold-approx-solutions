# What is a posterior Gaussian process?

A **posterior Gaussian process (GP)** is a probability model for an unknown function **after conditioning on observed data**. In ordinary GP regression with Gaussian observation noise, the posterior is itself a GP, with an updated mean function and covariance kernel.

The key distinction is:

> The posterior mean is one prediction function. The posterior GP describes uncertainty about the entire unknown function, including how its values at different inputs vary together.

This note develops the standard Gaussian-noise regression case. It assumes fixed kernel and noise parameters; integrating over uncertain parameters need not yield a Gaussian process.

Related notes: [What is a Gaussian process?](what-is-a-gaussian-process.md), [What is a posterior?](what-is-a-posterior.md), and [Kernel ridge regression](kernel-ridge-regression.md).

## 1. Start with a prior over functions

Let $\mathcal X$ be the input domain and let $f:\mathcal X\to\mathbb R$ be the unknown function. Write

$$
f\sim\operatorname{GP}(m,k),
$$

where

$$
m(x)=\mathbb E[f(x)],
\qquad
k(x,x')=\operatorname{Cov}(f(x),f(x')).
$$

Here $k$ is a symmetric positive-semidefinite kernel. For any finite collection of inputs $x_1,\ldots,x_n$, the vector of function values is jointly Gaussian:

$$
\begin{pmatrix}f(x_1)\\ \vdots\\ f(x_n)\end{pmatrix}
\sim\mathcal N(m_X,K),
$$

where $m_X=(m(x_1),\ldots,m(x_n))^T$ and $K_{ij}=k(x_i,x_j)$.

This is what “a Gaussian distribution over functions” means: every finite set of evaluations has a compatible multivariate Gaussian distribution. It does not require a density over functions with respect to an infinite-dimensional analogue of Lebesgue measure.

The prior specifies both:

- What values we expect before observing the current data, through $m$.
- How deviations at different inputs are related, through $k$.

For example, an RBF (Radial basis function) kernel models strong correlation between values at nearby inputs. This correlation is what allows a measurement at one location to inform predictions elsewhere.

## 2. Observations are not necessarily exact function values

Suppose we observe the dataset $D=(X,y)$, where $X=(x_1,\ldots,x_n)$ and

$$
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2),
\qquad \sigma^2>0.
$$

Assume the observation noise is independent of $f$, and treat the inputs as fixed. Then

$$
y\mid f_X\sim\mathcal N(f_X,\sigma^2I_n),
\qquad
f_X=(f(x_1),\ldots,f(x_n))^T.
$$

After averaging over the prior uncertainty in $f_X$,

$$
y\sim\mathcal N(m_X,C),
\qquad C=K+\sigma^2I_n.
$$

### What does “averaging over prior uncertainty in $f_X$” mean?

Before observing $y$, we do not know the latent vector $f_X$. The prior assigns probabilities to its possible values:

$$
f_X\sim\mathcal N(m_X,K).
$$

If we temporarily knew that $f_X=u$, the observation model would be

$$
y\mid f_X=u\sim\mathcal N(u,\sigma^2I_n).
$$

But we do not know $u$. To obtain the distribution of $y$ alone, we combine these conditional observation distributions, weighting each possible $u$ by its prior probability. This operation is called **marginalization**, or **integrating out $f_X$**.

When $K$ is positive definite, so that the prior has an ordinary density on $\mathbb R^n$, this is

$$
\boxed{
 p(y)=\int_{\mathbb R^n}p(y\mid f_X=u)\,p_{f_X}(u)\,du.
}
$$

More generally, including singular $K$, we can write the same operation as an expectation with respect to the prior distribution:

$$
p(y)=\mathbb E_{f_X}\!\left[p(y\mid f_X)\right].
$$

For each fixed candidate observation vector $y$, we average its conditional density over the possible latent vectors. We are **not** averaging the observed measurements, and we are **not** simply replacing $f_X$ by its mean.

To see the distinction, replacing $f_X$ by $m_X$ would give $\mathcal N(m_X,\sigma^2I_n)$ and discard uncertainty about the latent function. Marginalization retains that uncertainty and gives $\mathcal N(m_X,K+\sigma^2I_n)$ instead.

#### Why do the covariances add?

Since $y=f_X+\varepsilon$, with independent Gaussian vectors $f_X$ and $\varepsilon$, their sum is Gaussian. Its mean follows from the law of total expectation:

$$
\mathbb E[y]
=\mathbb E_{f_X}\!\left[\mathbb E[y\mid f_X]\right]
=\mathbb E[f_X]
=m_X.
$$

The law of total covariance gives

$$
\begin{aligned}
\operatorname{Cov}(y)
&=\mathbb E_{f_X}\!\left[\operatorname{Cov}(y\mid f_X)\right]
 +\operatorname{Cov}_{f_X}\!\left(\mathbb E[y\mid f_X]\right)\\
&=\sigma^2I_n+K.
\end{aligned}
$$

The first term is the measurement variability remaining even if the function values were known. The second is our prior uncertainty about those function values themselves. Gaussianity of the sum, together with these moments, establishes the stated distribution of $y$.

For a scalar example, suppose $f(x_1)\sim\mathcal N(20,4)$ and the measurement noise has variance $1$. If the latent value were known to be $22$, then $y_1\mid f(x_1)=22\sim\mathcal N(22,1)$. Before knowing that latent value, however, averaging over its prior gives

$$
y_1\sim\mathcal N(20,5).
$$

The variance is $5$, not $1$, because both the unknown latent value and the measurement noise contribute.

This distribution of $y$ before conditioning on its realized value is called the **prior predictive distribution**. Its density evaluated at the observed data is also called the **marginal likelihood** or **evidence**. The next step—forming the posterior—uses the observed $y$ to update our beliefs about $f$, rather than integrating those beliefs out to predict $y$.

### The resulting observation covariance

The matrix $C$ is positive definite even if $K$ is singular. Its two terms represent different sources of variation:

- $K$: variation in the latent function under the prior.
- $\sigma^2I_n$: independent measurement noise.

Conditioning changes our beliefs about $f$; it does not change the already observed values of $y$.

## 3. The posterior GP formulas

For any input $x$, define

$$
k_X(x)=\begin{pmatrix}k(x_1,x)\\ \vdots\\ k(x_n,x)\end{pmatrix}\in\mathbb R^n.
$$

Under the assumptions above,

$$
\boxed{f\mid D\sim\operatorname{GP}(m_D,k_D),}
$$

where

$$
\boxed{m_D(x)=m(x)+k_X(x)^TC^{-1}(y-m_X)}
$$

and

$$
\boxed{k_D(x,x')=k(x,x')-k_X(x)^TC^{-1}k_X(x').}
$$

These formulas define a mean and covariance for every pair of inputs, not only the training locations.

### Reading the mean formula

The posterior mean is

$$
\text{prior mean}+\text{correlation-weighted correction from the data}.
$$

The residual vector $y-m_X$ records how the measurements differ from prior expectations. The solve $C^{-1}(y-m_X)$ accounts for both correlations among observations and measurement noise. Finally, $k_X(x)$ determines how relevant those observations are to the test input $x$.

The correction is not generally a convex average: its coefficients can be negative and need not sum to one.

### Reading the covariance formula

The posterior covariance is

$$
\text{prior covariance}-\text{covariance explained by the observations}.
$$

At a single input,

$$
s_D^2(x):=k_D(x,x)
=k(x,x)-k_X(x)^TC^{-1}k_X(x).
$$

In exact arithmetic,

$$
0\le s_D^2(x)\le k(x,x).
$$

Thus conditioning reduces, or leaves unchanged, the marginal variance under this fixed Gaussian model. More strongly, the covariance reduction on any finite test set is positive semidefinite.

## 4. Why does the posterior remain Gaussian?

Choose $q$ test inputs $Z=(z_1,\ldots,z_q)$ and define

$$
f_Z=(f(z_1),\ldots,f(z_q))^T.
$$

Let $m_Z\in\mathbb R^q$ contain the prior means, $K_{XZ}\in\mathbb R^{n\times q}$ have entries $k(x_i,z_j)$, and $K_{ZZ}\in\mathbb R^{q\times q}$ have entries $k(z_i,z_j)$. Write $K_{ZX}=K_{XZ}^T$.

Because GP evaluations and independent Gaussian noise are jointly Gaussian,

$$
\begin{pmatrix}y\\ f_Z\end{pmatrix}
\sim\mathcal N\left(
\begin{pmatrix}m_X\\m_Z\end{pmatrix},
\begin{pmatrix}
C & K_{XZ}\\
K_{ZX} & K_{ZZ}
\end{pmatrix}
\right).
$$

Define the residual after a linear prediction from $y$:

$$
r=f_Z-m_Z-K_{ZX}C^{-1}(y-m_X).
$$

Its cross-covariance with $y$ is

$$
\operatorname{Cov}(r,y)
=K_{ZX}-K_{ZX}C^{-1}C=0.
$$

Since $r$ and $y$ are jointly Gaussian, zero cross-covariance implies independence. Moreover,

$$
\operatorname{Cov}(r)
=K_{ZZ}-K_{ZX}C^{-1}K_{XZ}.
$$

After conditioning on $y$, the linear prediction is fixed and the independent residual retains its Gaussian distribution. Therefore,

$$
f_Z\mid D\sim\mathcal N\left(
 m_Z+K_{ZX}C^{-1}(y-m_X),
 K_{ZZ}-K_{ZX}C^{-1}K_{XZ}
\right).
$$

This works for every finite test set, giving precisely the posterior GP mean and kernel above.

**Why the Gaussian noise assumption matters:** With a non-Gaussian likelihood, such as binary classification observations, the posterior is generally not a GP. Gaussian approximations may still be useful, but they are approximations.

## 5. A worked example with one noisy observation

Consider a synthetic one-dimensional example with zero prior mean and RBF kernel

$$
k(x,x')=\tau^2\exp\left(-\frac{(x-x')^2}{2\ell^2}\right),
\qquad \tau^2>0,\quad \ell>0.
$$

### What is the RBF kernel?

**RBF** stands for **radial basis function**. The kernel above is also called the **squared-exponential kernel** or **Gaussian kernel**. “Radial” means that its value depends only on the distance between inputs, not their absolute locations or direction. Although RBF can describe a broader family of radial functions, in GP regression “the RBF kernel” usually means this squared-exponential form.

For inputs $x,x'\in\mathbb R^d$, its isotropic version is

$$
k(x,x')=\tau^2\exp\left(-\frac{\|x-x'\|_2^2}{2\ell^2}\right),
\qquad \tau^2>0,\quad \ell>0.
$$

This is a symmetric positive-semidefinite kernel, so it can serve both as a GP covariance kernel and as an RKHS reproducing kernel. It is not a probability density: there is no requirement that its integral equal one.

The parameters have distinct roles:

- **Signal variance $\tau^2$:** Since $k(x,x)=\tau^2$, this is the prior variance of the latent function at each input. The prior standard deviation is $\tau$. It is separate from the measurement-noise variance $\sigma^2$.
- **Length scale $\ell$:** Controls how quickly correlation decays with distance. A large $\ell$ makes deviations from the mean strongly correlated over longer distances; a small $\ell$ permits more rapid variation.

In particular, the prior correlation is

$$
\operatorname{Corr}(f(x),f(x'))
=\frac{k(x,x')}{\sqrt{k(x,x)k(x',x')}}
=\exp\left(-\frac{\|x-x'\|_2^2}{2\ell^2}\right).
$$

Thus inputs separated by distance $\ell$ have correlation $e^{-1/2}\approx0.607$; at distance $2\ell$ it is $e^{-2}\approx0.135$; and at distance $3\ell$ it is about $0.011$. Nearby observations therefore tend to be informative about one another under this prior. These are correlations of deviations from the prior mean, not a requirement that the function itself be positive.

**Smoothness assumption:** The squared-exponential kernel represents a very smooth function model. Its GP has mean-square derivatives of every order; on Euclidean domains it admits a version with infinitely differentiable sample paths. Changing $\ell$ changes the scale of variation, not this smoothness class. This can be a poor assumption for functions with abrupt changes or rough behavior; other kernels, such as Matérn kernels, allow different smoothness assumptions.

**Input units matter:** Euclidean distance can be misleading when coordinates have different units or scales. One option is to scale the inputs using training data only. Another is to assign a separate length scale to each coordinate:

$$
k(x,x')=\tau^2\exp\left(
-\frac12\sum_{j=1}^d\frac{(x_j-x_j')^2}{\ell_j^2}
\right),
\qquad \ell_j>0.
$$

Finally, an ordinary Euclidean RBF kernel does not automatically account for manifold geometry: two inputs can be close in ambient Euclidean distance without being close along an underlying manifold.

### Conditioning on one observation

Observe one value $y_0$ at $x_0$. Define

$$
\rho(x)=\exp\left(-\frac{(x-x_0)^2}{2\ell^2}\right).
$$

Then $C=\tau^2+\sigma^2$ is a scalar, so

$$
m_D(x)=\frac{\tau^2\rho(x)}{\tau^2+\sigma^2}y_0,
$$

$$
s_D^2(x)=\tau^2-\frac{\tau^4\rho(x)^2}{\tau^2+\sigma^2}.
$$

For example, let $\tau^2=1$, $\sigma^2=1/4$, and $y_0=2$. At the observed input,

$$
m_D(x_0)=1.6,
\qquad
s_D^2(x_0)=0.2.
$$

The predicted latent value is not exactly $2$: the model allows the measurement to contain noise. Nor does its uncertainty vanish after one noisy measurement.

Far from $x_0$, $\rho(x)\to0$, so

$$
m_D(x)\to0,
\qquad
s_D^2(x)\to1.
$$

The prediction returns to the prior mean and prior variance because this particular kernel makes distant values weakly correlated. This behavior is kernel-dependent, not a universal property of all GPs.

## 6. Three different things called a prediction

### A. Posterior mean

The deterministic function $m_D(x)$ is a point prediction. At each input, it minimizes posterior expected squared error:

$$
m_D(x)=\operatorname*{arg\,min}_{a\in\mathbb R}
\mathbb E[(f(x)-a)^2\mid D].
$$

It is not the entire posterior.

### B. Posterior distribution of the latent function value

At one input,

$$
f(x)\mid D\sim\mathcal N(m_D(x),s_D^2(x)).
$$

A pointwise approximately $95\%$ posterior credible interval is

$$
m_D(x)\pm1.96\,s_D(x).
$$

“Pointwise” matters: intervals with $95\%$ posterior probability at each individual input do not automatically form a band containing the whole function with $95\%$ probability.

### C. Posterior predictive distribution of a new measurement

For a new independent noise realization,

$$
y_*=f(x_*)+\varepsilon_*,
\qquad \varepsilon_*\sim\mathcal N(0,\sigma^2),
$$

we have

$$
y_*\mid D\sim\mathcal N\left(m_D(x_*),s_D^2(x_*)+\sigma^2\right).
$$

The additional $\sigma^2$ represents noise in the future measurement. Even if the latent function becomes well known, that noise remains.

In the one-observation example at $x_0$, the latent posterior variance is $0.2$, whereas a new measurement has predictive variance $0.2+0.25=0.45$.

## 7. Why the covariance between test inputs matters

Knowing only the individual means and variances is not enough to describe uncertainty about a function. The posterior kernel specifies

$$
\operatorname{Cov}(f(x),f(x')\mid D)=k_D(x,x').
$$

For example, uncertainty in a difference is

$$
\operatorname{Var}(f(x)-f(x')\mid D)
=s_D^2(x)+s_D^2(x')-2k_D(x,x').
$$

Strong positive covariance can make the difference much more certain than either value considered separately.

To draw a posterior function on a finite grid $Z$, sample the entire vector jointly from

$$
\mathcal N(m_D(Z),K_D(Z,Z)).
$$

Drawing each grid value independently from its marginal distribution would discard the posterior correlations and generally produce the wrong kind of function sample.

## 8. What controls posterior uncertainty?

With fixed kernel and noise parameters, the posterior covariance depends on the input locations and model parameters, but **not on the observed target values** $y$.

Consequently, two datasets with the same inputs but different targets have the same posterior covariance under the same fixed model, although their posterior means differ.

This is an exact property of Gaussian conditioning—not a statement that prediction difficulty never depends on the measured values. In practice:

- Kernel and noise parameters are often fitted using $y$, which indirectly changes the covariance.
- A misspecified kernel can produce overconfident predictions.
- Plug-in parameter estimates omit uncertainty about the parameters themselves.
- Bayesian credible intervals do not guarantee frequentist coverage for arbitrary real-world data.

If parameters are integrated out, the posterior is generally a mixture of conditional GPs rather than a single GP. Its covariance can also include uncertainty in the conditional mean.

## 9. Connection to RKHS and kernel ridge regression

Let $\mathcal H_k$ be the RKHS associated with the same kernel $k$, and first take $m=0$. Consider

$$
\widehat f=\operatorname*{arg\,min}_{g\in\mathcal H_k}
\left\{\frac1n\sum_{i=1}^n(y_i-g(x_i))^2
+\lambda\|g\|_{\mathcal H_k}^2\right\},
\qquad \lambda>0.
$$

The representer argument yields

$$
\widehat f(x)=k_X(x)^T(K+n\lambda I_n)^{-1}y.
$$

Comparing this with the posterior mean gives

$$
\boxed{n\lambda=\sigma^2\quad\Longrightarrow\quad\widehat f=m_D.}
$$

Thus the same function arises from two different constructions:

- Penalized optimization in an RKHS.
- Conditional expectation under a probabilistic model.

For a nonzero prior mean, apply the regularized formulation to the correction $g=f-m$ using residual observations $y-m_X$. The correction belongs to $\mathcal H_k$; the chosen prior mean need not.

The GP posterior covariance is additional probabilistic structure, not something supplied automatically by the KRR point estimate.

**Important caveat:** GP sample functions are not generally elements of the associated RKHS. For common infinite-dimensional kernels they lie outside it almost surely. The finite kernel expansion of the posterior mean does not imply the same property for random posterior draws. The equality above is a statement about the posterior mean, not an identification of the GP with an RKHS-valued random variable.

## 10. Noiseless interpolation as a limiting case

If $K$ is invertible and $\sigma^2\to0$, then

$$
m_D(x)\to m(x)+k_X(x)^TK^{-1}(y-m_X).
$$

At the observed inputs, this gives

$$
m_D(x_i)\to y_i,
\qquad
s_D^2(x_i)\to0.
$$

Exact observations determine the function at the observed locations, but generally not everywhere else. Away from those locations, posterior uncertainty can remain.

If $K$ is singular, this inverse formula does not apply directly; exact conditioning requires attention to consistency and degeneracy. Positive observation noise avoids that particular invertibility issue.

## 11. Computing the posterior stably

The formulas use $C^{-1}$ for exposition. Implementations should use linear solves, typically via a Cholesky factorization:

1. Form $C=K+\sigma^2I_n$ and factor $C=LL^T$.
2. Solve $L a=y-m_X$ and $L^T\alpha=a$.
3. Compute $m_D(Z)=m_Z+K_{ZX}\alpha$.
4. Solve $LV=K_{XZ}$.
5. Compute $K_D(Z,Z)=K_{ZZ}-V^TV$.

Here $V\in\mathbb R^{n\times q}$. Standard dense training costs $O(n^3)$ time and $O(n^2)$ storage, before accounting for test-set computations.

Numerical jitter is a small diagonal addition used to stabilize a factorization. It should be documented separately from modeled measurement noise: both affect the matrix, but their intended meanings differ.

## 12. Main takeaway

A posterior GP is **an updated distribution over functions**, not just a fitted curve:

$$
\boxed{
\text{prior GP}+\text{Gaussian observation model}+\text{data}
\ \longrightarrow\ \operatorname{GP}(m_D,k_D).
}
$$

The mean tells us what to predict under squared loss. The covariance tells us how uncertain the latent function remains and how that uncertainty is coupled across inputs. Under matching conventions, the mean is also an RKHS-regularized predictor, giving a precise bridge between kernel machine learning and Bayesian inference.

*Scope note: This is a self-contained background derivation, not a verified mapping to specific course theorems. Formal references and lecture alignment should be added when preparing the manuscript.*
