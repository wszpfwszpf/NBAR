# Event Stream Denoising for Optical Flow Evaluation

This repository contains an experimental pipeline for **event stream denoising**, with the primary goal of **improving optical flow quality** rather than preserving perfect event-level structures.

The project is currently in a **refactoring and stabilization phase**.

---

## Project Goal

The core objective of this project is:

> To investigate whether lightweight event stream denoising strategies can **stably improve downstream optical flow estimation**, even if some true events are removed.

This is **not** intended to be a structure-preserving denoiser or a generic event enhancement framework.

---

## Pipeline Overview

The current conceptual pipeline is:

Raw Events

↓

S1 (optional, aggressive filtering)

↓

S2 (optional, compensation / refinement)

↓

Optical Flow Evaluation


- **S1**:  
  May remove true events. This is acceptable as long as optical flow quality improves overall.
- **S2**:  
  Intended as a compensation or refinement stage. It is not necessarily structure-preserving.

---

## Key Design Principles

- Event representation is fixed as **(t, x, y, p)**.
- **Real-time suitability** is prioritized over optimal denoising quality.
- The project currently avoids:
  - FFT or heavy frequency-domain analysis
  - Large or complex neural models
  - Offline or global optimization modules
- The focus is on **engineering robustness and empirical validation**, not algorithmic complexity.

---

## Current Status

- The codebase is under **structural refactoring**.
- No new denoising algorithms are introduced at this stage.
- The goal of refactoring is to:
  - Stabilize the pipeline structure
  - Enable plug-and-play S1 / S2 modules
  - Ensure reproducible experiments

---

## Usage (Temporary)

This section will be updated after refactoring is complete.

At the current stage, scripts may be experimental and interfaces are subject to change.

---

## Notes

- Intermediate visualizations, images, and generated data are intentionally excluded from version control.
- This repository prioritizes **clarity of experimental logic** over completeness.

