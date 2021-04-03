![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)

# Awesome Reference in Statistics and Machine Learning

This repository contains a curated list of awesome open source libraries that will help you deploy, monitor, version, scale, and secure your production machine learning.

## Tags

| | | | |
|-|-|-|-|
| :golf: Methodology (METH) | :blue_book: Learning Theory (LT) | :dart: Optimization (OPT) | :mag_right: Statistical Inference (INF) |
| :computer: Software (SW) | :unlock: Explainable AI (XAI) | :cherries: Biostatistics (BIO) | :keyboard: Empirical Studies (ES) |
|:dart: Deep Learning (DL) |

## Must-Read-Refs

[METH][OPT] Breiman, L. (2001). [Random forests](https://link.springer.com/article/10.1023/A:1010933404324). Machine Learning, 45(1), 5-32.

- **keywords**: random forest, assemble methods, bias-variance trade-off

[SW] [How to upload your python package to PyPi](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56).

- **keywords**: Python package/library, Pypi, twine

## Reading

[LT] Garnham, A. L., & Prendergast, L. A. (2013). [A note on least squares sensitivity in single-index model estimation and the benefits of response transformations](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-7/issue-none/A-note-on-least-squares-sensitivity-in-single-index-model/10.1214/13-EJS831.full). Electronic Journal of Statistics, 7, 1983-2004.

- **keywords**: sliced inverse regression (SIR), sufficient dimension reduction (SDR), OLS, single-index model
- **summary**: in a single-index model, when Cov(X,Y) is nonzero, OLS is able to recover a minimal sufficient reduction space, yet it fails when Cov(X,Y) = 0.

[ES][DL] Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016). [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf). arXiv preprint arXiv:1611.03530.

- **keywords**: deep learning, generalization, random labels
- **summary**: deep neural networks easily fit random labels. Figure 1: training errors for true labels, random labels, shuffled pixels, random pixels, are all converge to zeros. Yet the testing error would affect by the label corruption.

[BIO][METH][OPT][SW] Mak, T. S. H., Porsch, R. M., Choi, S. W., Zhou, X., & Sham, P. C. (2017). [Polygenic scores via penalized regression on summary statistics. Genetic epidemiology](https://onlinelibrary.wiley.com/doi/epdf/10.1002/gepi.22050), 41(6), 469-480. [R package: LassoSum](https://github.com/tshmak/lassosum)

- **keywords**: Summary statistics, sparse regression, invalid IVs, lasso, elastic net
- **summary**: solve LASSO and elastic net based on summary statistics: coordinate descent for Lasso (elastic net) only require summary data.