# Changelog

Notable changes to this project, newest first.

## 0.1.0 (2026-11-17)

First tagged release. It contains everything below.

## 2026-11-16

- Package versions pinned in `requirements.lock`, with a `make install-locked` target.
- README: the Bayesian line for a revenue metric is 1 - p/2 of the Welch test and adds no information.

## 2026-11-15

- First version: power and MDE, z and Welch tests, SRM check, CUPED, Beta-Binomial, group-sequential and mSPRT, each checked by simulation.
- Fixed: the noncentral-t lower tail returned NaN on SciPy 1.18; it is now computed by reflection.
