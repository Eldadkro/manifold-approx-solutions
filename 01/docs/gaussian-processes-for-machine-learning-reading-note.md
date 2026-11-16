# Reading note: Gaussian Processes for Machine Learning

**Status:** Suggested background reading; the author has not read this book. It is not currently cited in the manuscript.

Carl Edward Rasmussen and Christopher K. I. Williams, *Gaussian Processes for Machine Learning*, The MIT Press, 2006.

- [Official book website and freely available chapters](https://gaussianprocess.org/gpml/)
- [Chapter 2: Regression (PDF)](https://gaussianprocess.org/gpml/chapters/RW2.pdf) — Section 2.2 presents GP regression, including the posterior mean and covariance under Gaussian observation noise.
- [Chapter 6: Relationships between GPs and Other Models (PDF)](https://gaussianprocess.org/gpml/chapters/RW6.pdf) — Sections 6.1–6.2 discuss RKHSs, regularization, and the connection with GP prediction.

## Relevance to this paper

The book is a possible future reference for the interpolation → KRR → GP posterior mean narrative. When reading, compare its regularization normalization with the manuscript's averaged squared loss: using the same kernel, our matching convention is $n\lambda=\sigma^2$.

Section 6.1 also discusses the distinction between a GP's sample functions and its associated RKHS. The manuscript currently makes only the narrower observation that membership of the posterior mean in the RKHS does not establish membership of GP sample functions; it does not state the almost-sure nonmembership result without a reference or proof.

## Citation decision

Book citations and the printed bibliography were removed from `paper/main.tex` at the author's request because the author has not read the source. The unused entry in `paper/references.bib` is retained as bibliographic metadata for possible later use, not as a current manuscript reference. Reintroduce citations only after the author has reviewed the relevant material. Course-specific references still need checking against the lecture notes.
