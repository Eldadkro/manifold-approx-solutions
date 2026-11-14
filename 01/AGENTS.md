# Project: RKHS and Machine Learning

## Purpose
Write a small paper serving as a homework solution for the course **Manifold Approximations**. Present a real-world application of theory covered in the course so far, focusing on the connection between **reproducing kernel Hilbert spaces (RKHS)** and **machine learning (ML)**.

## Scope and direction
- Prioritize a clear, mathematically grounded exposition over a broad survey.
- Connect definitions and results from the course to a concrete ML problem, explaining what the theory contributes in practice.
- Keep the manuscript appropriate for a short course assignment rather than a full research paper.
- The manuscript will be written in LaTeX and compiled to PDF. The application, dataset, length, and exact course results to use are not yet decided. Ask the author before committing to these choices.
- Kernel ridge regression is a possible running example, not a settled topic. Other kernel methods may be considered if they better fit the material taught.

## Collaboration
- Treat the author as the owner of the mathematical argument and topic choices. Help develop, check, and polish the paper without silently expanding its scope.
- Ask for the assignment instructions and relevant lecture notes or covered theorems before claiming alignment with the course.
- Distinguish established course material, additional background, proposed interpretations, and empirical observations.
- Do not assume that RKHS methods automatically exploit manifold geometry; explain and justify any claimed connection to manifolds or manifold approximation.

## Mathematical writing
- Define notation and assumptions before using them, and keep notation consistent.
- State kernel properties and the domain explicitly. Distinguish a positive-semidefinite kernel from its finite-sample Gram matrix.
- Explain the reproducing property and how it links function evaluation to the Hilbert-space inner product.
- When using a representer theorem, state the relevant assumptions and explain the reduction from a function-space optimization problem to finitely many coefficients.
- Track normalization constants, regularization conventions, and matrix dimensions in derivations.
- Clearly distinguish exact statements, informal intuition, and claims requiring proof or citation.
- Give enough intermediate steps for a reader familiar with the course to follow the argument.

## Application and evidence
- Use a concrete, motivated problem rather than an unexplained toy calculation. A synthetic example may illustrate theory but should be labeled as synthetic.
- Explain the inputs, prediction target, kernel choice, learning objective, and evaluation procedure.
- If experiments are included, document data provenance, preprocessing, train/validation/test splits, hyperparameter selection, metrics, and random seeds where relevant.
- Avoid data leakage and distinguish training fit from generalization.
- Never invent experimental results, references, or claims about a dataset. Label proposed experiments and unverified claims explicitly.
- Keep empirical work small and reproducible, with only the dependencies needed for the chosen example.

## Suggested paper structure
Adapt this outline to the assignment once its requirements are available:
1. Introduction: the real-world problem and its connection to the course.
2. Mathematical background: relevant RKHS definitions and results.
3. Learning formulation: how the theory yields a practical ML method.
4. Worked application: data, implementation, and results, if required.
5. Discussion: interpretation, limitations, and the relationship to manifold approximation.
6. References.

## Folder structure
- `docs/`: Markdown knowledge notes and background material used to write the paper.
- `sim/`: Python code for simulations and experiments.
- `paper/`: LaTeX manuscript sources, references, and the compiled final PDF.
- `results/`: Simulation and experiment outputs, including metrics, tables, and figures. Keep generated results separate from simulation source code.

## Python and simulations
- Use `uv` to manage the Python environment and dependencies, with `pyproject.toml` and `uv.lock` at the project root once the Python project is initialized.
- Run simulations and experiments through `uv` from the project root, for example: `uv run python sim/<script>.py`.
- Add dependencies with `uv add <package>` rather than installing them with standalone `pip`.
- Save simulation and experiment outputs under `results/`; reference those outputs from the paper as needed.

## Files and workflow
- Follow the folder structure above for all new material.
- Prefer targeted edits and preserve the author's existing writing and notation.
- Verify bibliographic details and use primary sources when practical.
- Run relevant mathematical, code, and document-build checks after changes when tools are available. Report what was checked and what remains unverified.
