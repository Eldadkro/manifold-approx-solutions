# Real-world uses of RKHS theory, kernel learning, and Gaussian processes

Companion to [From RKHS to kernel machine learning and Gaussian processes](from-rkhs-to-kernel-machine-learning-and-gaussian-processes.md).

> **Scope and evidence.** This note explains application settings and how the mathematics in the companion note supports them. Cited research examples are distinguished from illustrative workflows. No experiments were run and no dataset or paper topic is selected here. Lecture numbering and alignment with the course remain unverified pending the actual lecture notes.

## 1. What practical problem does the theory solve?

Many scientific and engineering tasks have the same structure: an unknown function is expensive or impossible to observe everywhere, but a finite number of measurements are available. We want to predict its values elsewhere, avoid fitting measurement noise, and sometimes decide where to measure next.

Let the input domain be $\Omega$, with observations

$$
y_i=f_{\mathrm{true}}(x_i)+\varepsilon_i,\qquad x_i\in\Omega,\quad i=1,\ldots,n.
$$

A real, symmetric positive-semidefinite kernel $K:\Omega\times\Omega\to\mathbb R$ defines an RKHS $H_K$. Positive semidefiniteness means that every finite Gram matrix $G\in\mathbb R^{n\times n}$, with $G_{ij}=K(x_i,x_j)$, is positive semidefinite. The kernel is a function on pairs of inputs; $G$ is its matrix on the observed inputs.

Kernel ridge regression (KRR) solves

$$
\min_{f\in H_K}\left\{\sum_{i=1}^n(y_i-f(x_i))^2+\lambda\|f\|_{H_K}^2\right\},\qquad \lambda>0.
$$

Because evaluation is continuous in an RKHS, orthogonal projection onto the span of the observed kernel sections preserves the data fit and cannot increase the norm. Consequently, the unique minimizing function has a finite expansion:

$$
f_\lambda(x)=\sum_{i=1}^n\alpha_iK(x,x_i),\qquad
\alpha=(G+\lambda I_n)^{-1}y.
$$

Thus the RKHS is not merely abstract background: it explains why an infinite-dimensional learning problem reduces to a linear system.

For a zero-mean GP with covariance $K$ and independent Gaussian observation noise of variance $\sigma^2>0$, independent also of the GP, the same calculation gives

$$
\mu(x)=k(x)^T(G+\sigma^2I_n)^{-1}y,
$$

$$
s^2(x)=K(x,x)-k(x)^T(G+\sigma^2I_n)^{-1}k(x),
\qquad k(x)=(K(x,x_1),\ldots,K(x,x_n))^T.
$$

The KRR prediction equals the GP posterior mean when $\lambda=\sigma^2$, using the same kernel and the **sum-of-squares** objective above. If the loss is instead averaged over observations, the matching convention is $n\lambda=\sigma^2$. Kernel amplitude is part of $K$ and must also match.

The GP adds model-based uncertainty. Here $s^2(x)$ concerns the latent function; a future independent noisy observation has conditional variance $s^2(x)+\sigma^2$.

## 2. Environmental mapping and sensor placement

### The real-world task

Suppose a monitoring network measures a pollutant at a limited number of stations. Decision-makers need a map at locations without sensors and must decide where additional monitoring would be useful.

An illustrative regression setup is:

- **Input:** location, possibly augmented with time, weather, and land-use variables.
- **Target:** pollutant concentration, or a suitably transformed concentration.
- **Observations:** noisy station readings.
- **Output:** a predicted field and, with a GP, pointwise uncertainty.

For a purely spatial example, take $\Omega\subset\mathbb R^2$ and a squared-exponential kernel

$$
K(x,x')=a^2\exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right),\qquad a,\ell>0.
$$

This assumes correlations decay with spatial distance and imposes strong smoothness. A Matérn covariance is an alternative when less smooth fields are appropriate. Distances should use suitable physical coordinates, rather than treating raw latitude and longitude as Cartesian coordinates without justification.

### What the theory contributes

The posterior mean combines measurements through kernel similarities. Regularization or observation noise prevents the prediction from having to reproduce every sensor error. GP regression is closely related to spatial prediction methods known as kriging [1].

There is also a direct connection to the companion note's power function. For noiseless data and invertible $G$,

$$
P_X(x)^2=K(x,x)-k(x)^TG^{-1}k(x)=s^2(x).
$$

If a deterministic target belongs to $H_K$ and satisfies $\|f\|_{H_K}\le B$, its minimum-norm interpolant $s_Xf$ obeys

$$
|f(x)-s_Xf(x)|\le B P_X(x).
$$

A large power function therefore signals potential worst-case interpolation error, while the same quantity is a posterior standard deviation under the noiseless GP model. These are different interpretations with different assumptions—not interchangeable guarantees about the actual environment.

### Practical decision and evaluation

A candidate sensor site can be chosen to reduce uncertainty, for example by reducing integrated posterior variance over a region. Merely choosing the largest pointwise variance need not be optimal when installation costs, accessibility, and correlations between proposed sites matter.

Evaluate using held-out geographic regions or future time periods. Randomly splitting nearby readings can make performance look unrealistically good. Weather changes, barriers, miscalibrated sensors, and nonstationarity can invalidate a simple spatial kernel.

**Status:** the pollution-monitoring workflow is illustrative, not a reported experiment.

## 3. Predicting molecular properties with kernel ridge regression

### The real-world task

Quantum-chemical calculations can be expensive. A learned surrogate can estimate properties of candidate molecules before deciding which ones deserve a more costly calculation.

A documented example is the prediction of molecular atomization energies by Rupp and collaborators [2]. Their work uses molecular information derived from nuclear charges and atomic positions, including a Coulomb-matrix representation, in kernel-based regression.

An application formulation is:

- **Input:** a numerical molecular descriptor $x\in\Omega\subseteq\mathbb R^d$.
- **Target:** an energy computed with a specified reference method.
- **Kernel:** a positive-semidefinite similarity kernel on descriptors.
- **Objective:** regularized squared prediction error.

For a chosen Euclidean descriptor, a possible kernel is

$$
K(x,x')=\exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right).
$$

The exact descriptor, distance, and invariance treatment must be specified in an implementation; this formula is not a complete reproduction of [2].

### What the theory contributes

The RKHS norm controls complexity relative to the chosen notion of molecular similarity. The representer argument yields a predictor using only similarities to training molecules:

$$
\widehat E(x)=\sum_{i=1}^n\alpha_iK(x,x_i).
$$

No explicit list of all nonlinear features of the RKHS is needed. This is useful when a linear model in the original descriptor coordinates is too restrictive.

### Practical decision and evaluation

The surrogate can support preliminary screening. Report errors in physical units and distinguish prediction of a computational reference from prediction of experimentally measured energy.

Descriptor design must address physically irrelevant changes such as translation, rotation, and atom indexing. Evaluate chemical novelty explicitly: random splits may be much easier than predicting new molecular families. KRR alone does not provide calibrated predictive uncertainty, and low test error does not imply reliable extrapolation outside the training chemistry.

## 4. Bayesian optimization of expensive experiments

### The real-world task

Suppose each function evaluation requires training a machine-learning model, running a simulator, or conducting a laboratory experiment. The aim is to find a good input using relatively few evaluations—not necessarily to predict accurately everywhere.

A documented example is Bayesian optimization for machine-learning hyperparameter selection [3]. Laboratory optimization is another possible application setting, but no particular laboratory deployment is claimed here.

- **Input:** feasible configuration $x\in\Omega$, such as learning rate and regularization strength.
- **Target:** validation loss or another expensive scalar objective.
- **Model:** a GP surrogate for that objective.
- **Decision:** which configuration to evaluate next.

### What the theory contributes

The GP mean identifies promising settings, while its variance identifies uncertainty under the model. An acquisition function converts these two quantities into a sequential decision.

For a noise-free minimization problem, expected improvement can be written

$$
\operatorname{EI}(x)=\mathbb E\left[(f_{\mathrm{best}}-f(x))_+\mid\mathcal D\right],
\qquad (z)_+=\max(z,0),
$$

where $\mathcal D$ is the current data and $f_{\mathrm{best}}$ is the best observed value. The next input approximately maximizes EI. For noisy observations, the incumbent and improvement criterion require additional treatment; the smallest noisy reading need not represent the best latent value.

The practical loop is:

1. Evaluate an initial set of configurations.
2. Fit or update the GP.
3. Optimize an acquisition function over feasible inputs.
4. Evaluate the proposed input and add the result to the data.
5. Repeat until the evaluation budget is exhausted.

### Why the RKHS–GP connection matters

KRR and the GP may produce the same point prediction, but the sequential decision also uses uncertainty. The power-function connection explains why observation placement changes how well kernel sections cover the relevant prediction directions.

### Limitations and evaluation

Uncertainty-driven search can fail under a badly chosen covariance model. High-dimensional, categorical, or changing objectives require additional design. A standard GP does not ensure that an experiment is safe.

For hyperparameter tuning, use validation data for the search and reserve a separate test set for final assessment. Compare optimization methods under matched evaluation budgets and multiple random seeds.

## 5. Learning dynamics for robotics and control

### The real-world task

A robot may have imperfectly known dynamics because of friction, changing loads, or uncertain actuator behavior. Learning a transition model can support planning without requiring an exact analytical dynamics model.

Write

$$
z_{t+1}=g(z_t,u_t)+\varepsilon_t,
$$

where $z_t\in\mathbb R^p$ is the state and $u_t\in\mathbb R^q$ is the control. Regression inputs are $(z_t,u_t)\in\mathbb R^{p+q}$; targets can be individual components of the next state or state increment.

A documented example is PILCO, which uses probabilistic dynamics models in model-based policy search [4].

### What the theory contributes

For a simple model, fit one scalar GP per output component. Each posterior mean is a finite kernel expansion, while its posterior variance reflects model uncertainty about that component under the specified assumptions. Independent output GPs do not capture all cross-output correlations; a joint model would require additional structure.

The kernel determines which state–action pairs share information. The uncertainty estimate helps planning account for regions where the dynamics are poorly learned. However, propagating uncertainty through multiple nonlinear transitions requires additional methods beyond the single-step regression formulas in the companion note.

### Practical decision and evaluation

Evaluate on held-out trajectories, not only shuffled individual transitions. Measure both one-step prediction error and multi-step rollout behavior. A model with small one-step error can still accumulate substantial long-horizon error.

GP variance is not a safety certificate. Reliable deployment requires constraints, robust fallback behavior, and validation under relevant operating conditions. Out-of-distribution behavior can be much less predictable than a fitted GP suggests.

## 6. Reconstructing a field from indirect measurements

### The real-world task

Sensors often measure an average, integral, or other indirect quantity rather than the field value at one point. Examples include an average temperature over a sensor footprint or an integral of a field along a measurement path.

An illustrative model is

$$
y_i=L_i(f)+\varepsilon_i,
$$

where $L_i:H_K\to\mathbb R$ is a bounded linear functional. For example,

$$
L_i(f)=\int_\Omega w_i(t)f(t)\,dt,
$$

provided the kernel and weight make this a bounded functional.

### What the theory contributes

By Riesz representation, there is an $r_i\in H_K$ satisfying

$$
L_i(f)=\langle f,r_i\rangle_{H_K}.
$$

The same orthogonal-projection argument now gives

$$
\widehat f=\sum_{i=1}^n\alpha_i r_i,
\qquad
\alpha=(A+\lambda I_n)^{-1}y,
\qquad A_{ij}=\langle r_j,r_i\rangle_{H_K}.
$$

This minimizes $\sum_i(y_i-L_i(f))^2+\lambda\|f\|_{H_K}^2$. Point measurements are the special case $r_i=K(\cdot,x_i)$.

On the GP side, if the corresponding random measurements exist and the covariance operations are valid,

$$
A_{ij}=\operatorname{Cov}(L_i(f),L_j(f))
=L_i^{(x)}L_j^{(x')}K(x,x').
$$

Writing $c_i(x)=\operatorname{Cov}(L_i(f),f(x))$, the posterior mean becomes

$$
\mu(x)=c(x)^T(A+\sigma^2I_n)^{-1}y.
$$

Thus the central recipe survives: replace point-evaluation representers by measurement representers. This is particularly relevant to inverse problems, where measurements only partially constrain the unknown field.

### Limitations

Boundedness on the RKHS and existence of a random GP measurement should both be justified; typical GP paths need not belong to the RKHS. Derivative measurements require appropriate kernel regularity. Nonlinear measurement operators are not covered by this linear Gaussian formula.

Regularization selects a reconstruction but does not make an unidentifiable inverse problem fully observable. Components invisible to the measurements remain constrained mainly by the chosen model.

**Status:** this is a mathematical application template, not a claim about a particular imaging system or dataset.

## 7. Smoothing a noisy time signal: the Brownian kernel

The companion note's most explicit kernel is

$$
K(s,t)=\min(s,t),\qquad s,t\in[0,1].
$$

Its RKHS consists of absolutely continuous functions with $f(0)=0$ and square-integrable derivative. The regularized learning problem is therefore

$$
\min_f\left\{\sum_i(y_i-f(t_i))^2+
\lambda\int_0^1|f'(t)|^2\,dt\right\}.
$$

An illustrative use is reconstructing a slowly varying sensor offset relative to a known zero baseline. The loss rewards agreement with noisy readings, while the derivative penalty discourages rapid changes.

The representer theorem gives

$$
\widehat f(t)=\sum_i\alpha_i\min(t,t_i),
$$

which is piecewise linear, with possible changes of slope at observation times. A Brownian-motion GP gives the same posterior mean with the matching noise convention.

This is an unusually transparent example: one can interpret the RKHS norm directly as a roughness penalty rather than treating it as an abstract complexity measure.

The boundary condition matters. An unknown initial offset requires a separate mean/intercept model or another kernel. Beyond the last observation, this expansion is constant, so it does not automatically extrapolate a trend. Brownian sample paths are almost surely not members of this RKHS; a regular posterior mean should not be confused with a typical prior realization.

## 8. Which use case best highlights which result?

| Application | Main result from the companion note | Practical contribution | Additional material needed |
| --- | --- | --- | --- |
| Environmental mapping | GP conditioning; power function | Predict between sensors and plan measurements | Spatial validation and suitable covariance assumptions |
| Molecular property prediction | Representer theorem; KRR | Nonlinear surrogate from molecular similarities | Molecular descriptors and invariances |
| Bayesian optimization | Posterior mean and variance | Choose the next expensive evaluation | Acquisition functions and sequential optimization |
| Robotics and control | GP regression | Learn uncertain dynamics | Multi-step uncertainty propagation and control |
| Indirect measurements | Riesz representation | Reconstruct from averages or integrals | Measurement operators and identifiability |
| Time-signal smoothing | Brownian RKHS | Explicit derivative regularization | Boundary conditions and noise modeling |

For a short assignment, **spatial reconstruction** offers a direct connection between the power function and uncertainty, while **Brownian-kernel smoothing** offers the most explicit function-space derivation. **Molecular prediction** provides a documented scientific KRR application. These are alternatives for the author to consider, not topic commitments.

## 9. Common practical cautions

- **Kernel choice is a modeling assumption.** Similarity in a chosen feature space must correspond to useful similarity in the target. A small RKHS norm is kernel-dependent, not a universal definition of simplicity.
- **Uncertainty is conditional on a model.** Pointwise intervals $\mu(x)\pm1.96s(x)$ are latent-function credible intervals under the specified Gaussian model, not simultaneous confidence bands or guaranteed real-world coverage. Fitting hyperparameters and then treating them as fixed omits their uncertainty.
- **Validate the intended generalization task.** Use spatial, temporal, trajectory, or chemical-family splits when those reflect deployment. Fit preprocessing and choose hyperparameters using training/validation information only.
- **Dense computation has limits.** Standard dense training requires $O(n^3)$ factorization work and $O(n^2)$ storage. Use linear solves, usually Cholesky factorization, rather than explicitly forming matrix inverses. Large datasets need additional approximations or structure.
- **There is no automatic manifold result.** An ambient Euclidean kernel does not by itself exploit intrinsic manifold geometry. A manifold claim needs an explicit domain model and a justified kernel or approximation method adapted to that geometry.

## References and source status

1. Carl Edward Rasmussen and Christopher K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006. [Official book website and freely available chapters](https://gaussianprocess.org/gpml/). General background for GP regression, covariance design, model selection, and its relationship to other methods.
2. Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert Müller, and O. Anatole von Lilienfeld. “Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning.” *Physical Review Letters* **108**, 058301 (2012). [Publisher and DOI](https://doi.org/10.1103/PhysRevLett.108.058301). Documented molecular prediction example; no numerical performance claims from this paper are reproduced here.
3. Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. “Practical Bayesian Optimization of Machine Learning Algorithms.” *Advances in Neural Information Processing Systems* **25** (2012). [Official proceedings page](https://proceedings.neurips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html). Documented hyperparameter-optimization example.
4. Marc Peter Deisenroth and Carl Edward Rasmussen. “PILCO: A Model-Based and Data-Efficient Approach to Policy Search.” *Proceedings of the 28th International Conference on Machine Learning*, pp. 465–472 (2011). [Conference paper](https://icml.cc/Conferences/2011/papers/323_icmlpaper.pdf). Documented probabilistic dynamics and policy-search example.

The book and Bayesian-optimization proceedings pages were retrieved; molecular and PILCO bibliographic/application details were checked against web search results pointing to publisher or conference sources. This note is not a full close reading or reproduction of those studies. The remaining workflows are explicitly illustrative, and their empirical suitability has not been tested.
