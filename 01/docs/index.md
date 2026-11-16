# Documentation index

Entry point for the project's mathematical notes, application ideas, and paper planning. Links below cover all current Markdown documents in `docs/` other than this index. Editor configuration under `.obsidian/` is not documentation.

## Overview and paper planning

- [From RKHS to kernel machine learning and Gaussian processes](from-rkhs-to-kernel-machine-learning-and-gaussian-processes.md)
- [Real-world uses of RKHS theory, kernel learning, and Gaussian processes](real-world-use-cases-of-rkhs-kernel-learning-and-gaussian-processes.md)
- [Plan: connecting RKHS, machine learning, and Gaussian processes](rkhs-ml-gp-paper-plan.md)

## Kernel ridge regression

- [Kernel ridge regression](kernel-ridge-regression.md)
- [Why kernel ridge regression stays in the kernel span](why-kernel-ridge-regression-stays-in-the-kernel-span.md) — why an arbitrarily small positive norm penalty still excludes components perpendicular to the training-kernel span.

## Gaussian processes and probability

- [Deriving the GP predictive distribution from kernel ridge regression](deriving-the-gp-predictive-distribution-from-krr.md) — expands Section 5 of the RKHS-to-GP overview, deriving Gaussian observations, the joint distribution, and the conditional mean and variance.

- [What is a Gaussian process?](what-is-a-gaussian-process.md)
- [What is a posterior?](what-is-a-posterior.md)
- [What is a posterior Gaussian process?](what-is-a-posterior-gp.md)
- [Random functions, covariance kernels, and the connection to KRR](random-functions-mean-covariance-and-krr.md)

## Maintenance

Keep this index synchronized whenever documents are added, renamed, moved, deleted, or substantially revised. Update link targets, titles, grouping, and descriptions as needed. Include documents in subdirectories, but exclude editor configuration and generated artifacts. Check that every document is listed and every local document link resolves.
