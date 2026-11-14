# What is a posterior?

A **posterior distribution** describes uncertainty about an unknown quantity **after conditioning on observed data**.

The unknown quantity might be a scalar parameter, a vector, a class label, or an entire function. In Gaussian process regression, it is the unknown function and its values at inputs of interest.

## 1. Prior, likelihood, and posterior

Let $\theta$ denote an unknown quantity and $D$ the observed data.

- **Prior** $p(\theta)$: uncertainty about $\theta$ before using these data.
- **Likelihood** $p(D\mid\theta)$: how the observation model scores the observed data for each possible $\theta$.
- **Posterior** $p(\theta\mid D)$: uncertainty about $\theta$ after observing $D$.

Bayes' rule connects them:

$$
\boxed{p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)}.}
$$

For a continuous parameter with suitable densities,

$$
p(D)=\int p(D\mid\theta)p(\theta)\,d\theta.
$$

For a discrete parameter, replace the integral by a sum. The denominator, called the **evidence** or **marginal likelihood**, normalizes the posterior so its total probability is one, assuming it is finite and positive.

A useful shorthand is

$$
\boxed{\text{posterior}\ \propto\ \text{likelihood}\times\text{prior}.}
$$

The likelihood is a function of $\theta$ with the observations held fixed. It is not, by itself, a probability distribution over $\theta$.

## 2. A simple numerical example

Suppose $\theta$ is an unknown constant temperature. Before taking a measurement, assume

$$
\theta\sim\mathcal N(20,4).
$$

Here the second argument is the variance: our prior mean is $20$ and prior standard deviation is $2$.

We take one noisy measurement:

$$
y\mid\theta\sim\mathcal N(\theta,1).
$$

Suppose we observe $y=23$. The posterior is

$$
\theta\mid y=23\sim\mathcal N(22.4,0.8).
$$

The posterior mean lies between the prior mean and the measurement, closer to the measurement because the measurement has smaller variance.

### Derivation

More generally, let

$$
\theta\sim\mathcal N(\mu_0,\tau^2),\qquad
y\mid\theta\sim\mathcal N(\theta,\sigma^2),
\qquad\tau^2,\sigma^2>0.
$$

Bayes' rule gives

$$
p(\theta\mid y)\propto
\exp\left[-\frac12\left(
\frac{(\theta-\mu_0)^2}{\tau^2}
+\frac{(y-\theta)^2}{\sigma^2}
\right)\right].
$$

Collecting the quadratic and linear terms in $\theta$ and completing the square yields

$$
\theta\mid y\sim\mathcal N(\mu_D,v_D),
$$

where

$$
\boxed{v_D=\left(\frac1{\tau^2}+\frac1{\sigma^2}\right)^{-1}}
$$

and

$$
\boxed{\mu_D=v_D\left(\frac{\mu_0}{\tau^2}+\frac{y}{\sigma^2}\right)
=\frac{\sigma^2}{\tau^2+\sigma^2}\mu_0
+\frac{\tau^2}{\tau^2+\sigma^2}y.}
$$

The reciprocal of variance is called **precision**. In this model, posterior precision is the sum of prior and measurement precisions.

Substituting $\mu_0=20$, $\tau^2=4$, $\sigma^2=1$, and $y=23$ gives $v_D=0.8$ and $\mu_D=22.4$.

## 3. A posterior is not just an estimate

A posterior is a full distribution. Different summaries answer different questions:

- **Posterior mean:** $\mathbb E[\theta\mid D]$, a point estimate optimal for squared-error loss when the relevant moments exist.
- **Posterior variance:** remaining spread around that mean.
- **Maximum a posteriori (MAP) estimate:** a maximizer of the posterior density, when one exists.
- **Credible interval:** an interval containing a specified amount of posterior probability.

For a scalar Gaussian posterior $\mathcal N(\mu_D,v_D)$, an approximately 95% credible interval is

$$
[\mu_D-1.96\sqrt{v_D},\ \mu_D+1.96\sqrt{v_D}].
$$

Its interpretation is conditional on the observations and assumed model: the posterior assigns approximately 95% probability to that interval.

A frequentist confidence interval has a different definition, based on repeated-sampling coverage of an interval-producing procedure. Neither interpretation automatically guarantees that the assumed model is appropriate.

## 4. Posterior versus posterior predictive distribution

The **posterior** concerns the unknown quantity $\theta$.

The **posterior predictive distribution** concerns a future observation $y_*$. Assuming the future observation is conditionally independent of $D$ given $\theta$,

$$
\boxed{p(y_*\mid D)=\int p(y_*\mid\theta)p(\theta\mid D)\,d\theta.}
$$

This averages predictions over uncertainty about $\theta$ rather than substituting a single estimate.

For the temperature example, a new independent measurement with noise variance $1$ satisfies

$$
y_*\mid D\sim\mathcal N(22.4,1.8).
$$

The variance is $0.8+1$: uncertainty about the true temperature plus new measurement noise.

## 5. What is the posterior in a Gaussian process?

A [Gaussian process](what-is-a-gaussian-process.md) places a prior on an unknown function:

$$
f\sim\mathcal{GP}(m,K).
$$

Suppose the observations are

$$
y_i=f(x_i)+\varepsilon_i,\qquad
\varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2),
$$

independently of $f$. Treat the input locations as fixed and define

$$
G_{ij}=K(x_i,x_j),\quad
m_X=(m(x_1),\ldots,m(x_n))^T,\quad
k_*=(K(x_1,x_*),\ldots,K(x_n,x_*))^T.
$$

Here $G\in\mathbb R^{n\times n}$ and $m_X,k_*,y\in\mathbb R^n$.

With $A=G+\sigma^2I_n$ invertible, Gaussian conditioning gives

$$
f(x_*)\mid D\sim\mathcal N(\mu_*,s_*^2),
$$

where

$$
\boxed{\mu_*=m(x_*)+k_*^TA^{-1}(y-m_X)}
$$

and

$$
\boxed{s_*^2=K(x_*,x_*)-k_*^TA^{-1}k_*.}
$$

The prior mean is corrected using the residuals $y-m_X$. The covariance kernel determines how those corrections propagate from measured inputs to the prediction location.

Conditioning all finite collections of function values gives the posterior GP, not merely separate unrelated Gaussian predictions.

For a new noisy measurement at $x_*$,

$$
y_*\mid D\sim\mathcal N(\mu_*,s_*^2+\sigma^2).
$$

These formulas assume fixed kernel and noise parameters. Plugging in estimated parameters does not generally account for uncertainty about those parameters.

### Pointwise is not simultaneous

A GP plot often shades

$$
\mu(x)\pm1.96s(x).
$$

These are pointwise posterior credible intervals for latent function values. They do not generally form a band containing the entire function with 95% posterior probability.

## 6. Connection to our RKHS paper

The posterior mean of a zero-mean GP agrees with the kernel ridge regression solution when the kernel and regularization/noise conventions match. But the interpretation is different:

- RKHS optimization selects a function by balancing data fit and a norm penalty.
- Bayesian conditioning produces a posterior distribution, from which a mean function can be extracted.

For noiseless observations and invertible Gram matrix, the posterior variance equals the squared RKHS power function:

$$
\operatorname{Var}(f(x)\mid D)=K(x,x)-k(x)^TG^{-1}k(x)=P_X(x)^2.
$$

This identity connects probabilistic uncertainty with deterministic interpolation geometry. It does not turn a credible interval into a universal error guarantee for arbitrary true functions.

See [From RKHS to kernel machine learning and Gaussian processes](from-rkhs-to-kernel-machine-learning-and-gaussian-processes.md) for the broader connection.

## Takeaway

**Prior:** what the model says before these observations.  
**Likelihood:** how observations arise from the unknown quantity.  
**Posterior:** what the model says about the unknown quantity after conditioning on observations.  
**Posterior predictive:** what the updated model says about a future observation.
