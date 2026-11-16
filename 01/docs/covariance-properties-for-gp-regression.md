# Covariance properties used in GP regression

This note collects the elementary covariance rules behind the Gaussian-process section of [the manuscript](../paper/main.tex) (compiled as [main.pdf](../paper/main.pdf)) and the companion probability notes. It is background material, not a claim about which results were covered in the course.

All random variables are real and have finite second moments. Vectors are column vectors. Inputs and model parameters are fixed; expectations are over the random function and observation noise, not over input locations.

## 1. Scalar covariance: definition and basic rules

For random variables $U,V$, define

$$
\operatorname{Cov}(U,V)
=\mathbb E[(U-\mathbb EU)(V-\mathbb EV)]
=\mathbb E[UV]-\mathbb E[U]\mathbb E[V].
$$

The second expression follows by expanding the product and using linearity of expectation. In particular,

$$
\operatorname{Var}(U)=\operatorname{Cov}(U,U)
=\mathbb E[(U-\mathbb EU)^2]\geq0.
$$

If both means are zero, covariance simplifies to $\mathbb E[UV]$. Without that assumption, the mean-product term must be retained.

For deterministic scalars $a,b,c,d$,

$$
\operatorname{Cov}(aU+b,cV+d)=ac\operatorname{Cov}(U,V).
$$

Indeed, centering $aU+b$ gives $a(U-\mathbb EU)$: constant shifts disappear. Covariance is symmetric and additive in each argument:

$$
\operatorname{Cov}(U,V)=\operatorname{Cov}(V,U),
\qquad
\operatorname{Cov}(U+V,W)=\operatorname{Cov}(U,W)+\operatorname{Cov}(V,W).
$$

Consequently,

$$
\boxed{\operatorname{Var}(U\pm V)
=\operatorname{Var}(U)+\operatorname{Var}(V)
\pm2\operatorname{Cov}(U,V).}
$$

**Common trap:** variance is not linear. In particular, $\operatorname{Var}(aU)=a^2\operatorname{Var}(U)$, and variances of sums add only when the cross-covariance vanishes.

## 2. Independence versus zero covariance

If $U$ and $V$ are independent, then $\mathbb E[UV]=\mathbb E[U]\mathbb E[V]$, so their covariance is zero. The converse is generally false.

For example, let $U\sim\mathcal N(0,1)$ and $V=U^2$. Symmetry gives $\mathbb E[U]=\mathbb E[U^3]=0$, hence

$$
\operatorname{Cov}(U,U^2)=0.
$$

Nevertheless, $V$ is completely determined by $U$, and they are not independent.

The special rule used in the paper is:

> **Jointly Gaussian random vectors with zero cross-covariance are independent.**

It is not enough for each variable separately to have a Gaussian distribution; the combined vector must be jointly Gaussian. One justification is that the characteristic function of a Gaussian vector with block-diagonal covariance factors into the characteristic functions of the two blocks. This also applies to singular Gaussian distributions.

## 3. Vector covariance and the transpose rule

For $U\in\mathbb R^p$ and $V\in\mathbb R^q$, define

$$
\operatorname{Cov}(U,V)
=\mathbb E[(U-\mathbb EU)(V-\mathbb EV)^\top]
\in\mathbb R^{p\times q}.
$$

Its $(i,j)$ entry is $\operatorname{Cov}(U_i,V_j)$. Write $\operatorname{Cov}(U)=\operatorname{Cov}(U,U)$. Then

$$
\operatorname{Cov}(V,U)=\operatorname{Cov}(U,V)^\top.
$$

Thus vector cross-covariance changes orientation when the arguments are reversed. In particular, $\operatorname{Cov}(y,f(x))$ is an $n\times1$ column, whereas $\operatorname{Cov}(f(x),y)$ is a $1\times n$ row.

For deterministic matrices $A\in\mathbb R^{r\times p}$ and $B\in\mathbb R^{s\times q}$ and deterministic shifts $a,b$ of matching sizes,

$$
\boxed{\operatorname{Cov}(AU+a,BV+b)
=A\operatorname{Cov}(U,V)B^\top.}
$$

To derive this, center the two vectors and move the deterministic matrices outside the expectation:

$$
\mathbb E[A(U-\mathbb EU)(V-\mathbb EV)^\top B^\top]
=A\operatorname{Cov}(U,V)B^\top.
$$

Useful special cases are

$$
\operatorname{Cov}(AU)=A\operatorname{Cov}(U)A^\top,
\qquad
\operatorname{Var}(a^\top U)=a^\top\operatorname{Cov}(U)a.
$$

For vectors $U,V$ of the same dimension,

$$
\operatorname{Cov}(U+V)
=\operatorname{Cov}(U)+\operatorname{Cov}(V)
+\operatorname{Cov}(U,V)+\operatorname{Cov}(V,U).
$$

The last two terms vanish if the vectors are independent (or merely uncorrelated).

## 4. Why covariance matrices and kernels are positive semidefinite

A covariance matrix $\Sigma=\operatorname{Cov}(U)$ is symmetric. For every deterministic vector $a$,

$$
a^\top\Sigma a=\operatorname{Var}(a^\top U)\geq0.
$$

This proves positive semidefiniteness, not necessarily invertibility: a linear combination can have zero variance.

For a random function $f$ on an input domain $\mathcal X$, its covariance kernel is

$$
k(x,x')=\operatorname{Cov}(f(x),f(x')).
$$

For any finite inputs $x_1,\ldots,x_n$, the matrix $K_{ij}=k(x_i,x_j)$ is the covariance matrix of $f_X=(f(x_1),\ldots,f(x_n))^\top$, so it is positive semidefinite. This establishes the kernel's positive-semidefinite property for every finite input collection. The function $k$ and the sample-dependent matrix $K$ are different objects.

Cauchy–Schwarz applied to centered random variables also gives

$$
|\operatorname{Cov}(U,V)|^2
\leq\operatorname{Var}(U)\operatorname{Var}(V),
\qquad
|k(x,x')|^2\leq k(x,x)k(x',x').
$$

Off-diagonal covariances can be negative; positive semidefiniteness does not mean that every entry is nonnegative.

## 5. Application: adding observation noise

Use the paper's notation and assumptions:

$$
f\sim\operatorname{GP}(0,k),\qquad
y=f_X+\varepsilon,\qquad
\varepsilon\sim\mathcal N(0,\sigma^2I_n),\qquad \sigma^2>0,
$$

where $\varepsilon$ is independent of the whole process $f$. Define

$$
k_x=(k(x_1,x),\ldots,k(x_n,x))^\top\in\mathbb R^n.
$$

The sum rule and independence give

$$
\operatorname{Cov}(y)=K+\sigma^2I_n=:C.
$$

Entry by entry, $\operatorname{Cov}(y_i,y_j)=k(x_i,x_j)+\sigma^2\delta_{ij}$, where $\delta_{ij}$ is one when $i=j$ and zero otherwise. Independent observation noise adds only to the diagonal.

The cross-covariance with a latent function value is unchanged:

$$
\operatorname{Cov}(y,f(x))
=\operatorname{Cov}(f_X,f(x))+\operatorname{Cov}(\varepsilon,f(x))
=k_x.
$$

Hence the joint covariance is

$$
\operatorname{Cov}\!\begin{pmatrix}y\\f(x)\end{pmatrix}
=\begin{pmatrix}C&k_x\\k_x^\top&k(x,x)\end{pmatrix}.
$$

These covariance calculations require no Gaussian assumption. Gaussianity additionally identifies the joint distribution as multivariate normal, which is essential for the independence argument below.

Since $K$ is positive semidefinite,

$$
v^\top Cv=v^\top Kv+\sigma^2\|v\|_2^2>0\quad(v\ne0).
$$

Therefore $C$ is invertible, and $C^{-1}$ is symmetric and positive definite, even if $K$ is singular.

## 6. Application: the residual used in Gaussian conditioning

Define the deterministic column $a_x=C^{-1}k_x$ and the scalar residual

$$
r(x)=f(x)-a_x^\top y.
$$

Because $C^{-1}$ is symmetric, $a_x^\top=k_x^\top C^{-1}$. The linear-transformation rule gives

$$
\begin{aligned}
\operatorname{Cov}(r(x),y)
&=\operatorname{Cov}(f(x),y)-a_x^\top\operatorname{Cov}(y)\\
&=k_x^\top-k_x^\top C^{-1}C=0.
\end{aligned}
$$

This is a $1\times n$ zero row. The residual and $y$ are jointly Gaussian because they are a linear transformation of $(f(x),y)$. Thus they are independent. Since $\mathbb E[r(x)]=0$,

$$
\mathbb E[f(x)\mid y]
=a_x^\top y+\mathbb E[r(x)\mid y]
=k_x^\top C^{-1}y.
$$

For two inputs, expand **all four terms** of the residual covariance:

$$
\begin{aligned}
\operatorname{Cov}(r(x),r(x'))
&=\operatorname{Cov}(f(x),f(x'))
-\operatorname{Cov}(f(x),a_{x'}^\top y)\\
&\quad-\operatorname{Cov}(a_x^\top y,f(x'))
+\operatorname{Cov}(a_x^\top y,a_{x'}^\top y)\\
&=k(x,x')-k_x^\top a_{x'}-a_x^\top k_{x'}+a_x^\top C a_{x'}\\
&=k(x,x')-k_x^\top C^{-1}k_{x'}.
\end{aligned}
$$

The two negative terms and the positive term contain the same scalar, so their net contribution is one negative copy.

To pass to conditional covariance, note that the **pair** $(r(x),r(x'))$ is jointly Gaussian with $y$ and has zero cross-covariance with it. The pair is therefore independent of $y$. Conditional on $y$, each $a_x^\top y$ is a constant, so

$$
\boxed{\operatorname{Cov}(f(x),f(x')\mid y)
=k(x,x')-k_x^\top C^{-1}k_{x'}.}
$$

Setting $x'=x$ yields the posterior variance, with

$$
0\leq\operatorname{Var}(f(x)\mid y)
=k(x,x)-k_x^\top C^{-1}k_x\leq k(x,x).
$$

The first inequality holds because this is a variance; the second follows from positive definiteness of $C^{-1}$.

## 7. Conditional covariance and future noisy measurements

Conditional covariance uses conditional centering:

$$
\operatorname{Cov}(U,V\mid y)
=\mathbb E[(U-\mathbb E[U\mid y])(V-\mathbb E[V\mid y])^\top\mid y].
$$

In general it can depend on the observed value of $y$. Its independence from response values in the GP formula above is special to this Gaussian model with fixed inputs and parameters. Parameters estimated from responses can introduce indirect dependence.

For a **new** observation $y_*=f(x)+\varepsilon_*$, assume the new noise has variance $\sigma^2$ and is independent of the entire process and training noise. It remains independent of $f(x)$ conditional on $y$. Thus

$$
\operatorname{Var}(y_*\mid y)
=\operatorname{Var}(f(x)\mid y)+\sigma^2.
$$

This extra term belongs to prediction of a new noisy measurement, not prediction of the latent function. Even at an existing training input, a new measurement has fresh noise; it is not the already observed random variable $y_i$.

## Related notes

- [Deriving the GP predictive distribution from kernel ridge regression](deriving-the-gp-predictive-distribution-from-krr.md) — the complete conditioning argument and connection to KRR.
- [Random functions, covariance kernels, and the connection to KRR](random-functions-mean-covariance-and-krr.md) — interpretation of random functions and covariance kernels. That note uses $K(x,z)$ for the kernel; here $k$ is the kernel and $K$ its Gram matrix, as in the manuscript.
- [What is a posterior Gaussian process?](what-is-a-posterior-gp.md) — interpretation of the posterior mean and covariance functions.
- [Documentation index](index.md).
