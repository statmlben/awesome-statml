![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)

# Awesome Reference in Statistics and Machine Learning

This repository contains a curated list of awesome references for statistics and machine learning.

## Tags

| | | |
|-|-|-|
| :golf: Methodology (METH) | :blue_book: Learning Theory (LT) | :dart: Optimization (OPT) | 
| :mag_right: Statistical Inference (INF) | :computer: Software (SW) | :unlock: Explainable AI (XAI) | 
| :cherries: Biostatistics (BIO) | :keyboard: Empirical Studies (ES) | :globe_with_meridians: Deep Learning (DL) | 
| :bar_chart: Dataset (DATA) | :arrow_right: Causal Inference (CI) | :spiral_notepad: Natural Language Learning (NLP) | 

## Reference

[METH] Fisher, R. A. (1922). [On the mathematical foundations of theoretical statistics](https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.1922.0009). Philosophical transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character, 222(594-604), 309-368.

[METH][OPT] Breiman, L. (2001). [Random forests](https://link.springer.com/article/10.1023/A:1010933404324). Machine Learning, 45(1), 5-32.

- **keywords**: random forest, assemble methods, bias-variance trade-off

[METH][LT] Bartlett, P. L., Jordan, M. I., & McAuliffe, J. D. (2006). [Convexity, classification, and risk bounds](https://doi.org/10.1198/016214505000000907). Journal of the American Statistical Association, 101(473), 138-156.

- **keywords**: Bayes rule, Fisher consistency, convex optimization; empirical process theory; excess risk bounds

[OPT] Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). [Distributed optimization and statistical learning via the alternating direction method of multipliers](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf). Foundations and Trends® in Machine learning, 3(1), 1-122.

- **keywords**: Convex Optimization, Proximity, Smooth objective function

[SW] [How to upload your python package to PyPi](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56).

- **keywords**: Python package/library, Pypi, twine

[SW] [Basic Tutorial for Cython](https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html)

- **keywords**: Python package/library, Cython, C/C++
- **memo**: Cython is Python with C data types, to speed up the Python loops.

[METH][LT][INF] Muandet, K., Fukumizu, K., Sriperumbudur, B., & Schölkopf, B. (2016). [Kernel mean embedding of distributions: A review and beyond](https://arxiv.org/pdf/1605.09522.pdf). arXiv preprint arXiv:1605.09522.

- **keywords**: Kernel method, RKHS, MMD
- **memo**: Overview of kernel methods, properties of RKHS and kernel-based MMD.

[LT] Garnham, A. L., & Prendergast, L. A. (2013). [A note on least squares sensitivity in single-index model estimation and the benefits of response transformations](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-7/issue-none/A-note-on-least-squares-sensitivity-in-single-index-model/10.1214/13-EJS831.full). Electronic Journal of Statistics, 7, 1983-2004.

- **keywords**: sliced inverse regression (SIR), sufficient dimension reduction (SDR), OLS, single-index model
- **memo**: in a single-index model, when Cov(X,Y) is nonzero, OLS is able to recover a minimal sufficient reduction space, yet it fails when Cov(X,Y) = 0.

[ES][DL] Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016). [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf). arXiv preprint arXiv:1611.03530.

- **keywords**: deep learning, generalization, random labels
- **memo**: deep neural networks easily fit random labels. Figure 1: training errors for true labels, random labels, shuffled pixels, random pixels, are all converge to zeros. Yet the testing error would affect by the label corruption.

[BIO][METH][OPT][SW] Mak, T. S. H., Porsch, R. M., Choi, S. W., Zhou, X., & Sham, P. C. (2017). [Polygenic scores via penalized regression on memo statistics.](https://onlinelibrary.wiley.com/doi/epdf/10.1002/gepi.22050) Genetic epidemiology, 41(6), 469-480. [R package: LassoSum](https://github.com/tshmak/lassosum)

- **keywords**: memo statistics, sparse regression, invalid IVs, lasso, elastic net
- **memo**: solve LASSO and elastic net based on memo statistics: coordinate descent for Lasso (elastic net) only require memo data.

[METH][DATA][SW] Bhatia, K. and Dahiya, K. and Jain, H. and Kar, P. and Mittal, A. and Prabhu, Y. and Varma, M. (2016). [The extreme classification repository: multi-label datasets & code.](http://manikvarma.org/downloads/XC/XMLRepository.html)

- **keywords**: extreme classification, multi-label classification
- **memo**: The objective in extreme multi-label classification is to learn feature architectures and classifiers that can automatically tag a data point with the most relevant subset of labels from an extremely large label set. This repository provides resources that can be used for evaluating the performance of extreme multi-label algorithms including datasets, code, and metrics.

[METH][DATA] Covington, P., Adams, J., & Sargin, E. (2016). [Deep neural networks for youtube recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf). In Proceedings of the 10th ACM conference on recommender systems (pp. 191-198).

- **keywords**: recommender systems, extreme classification, ranking, candidate set
- **memo**: A two-stage recommender system: first detail a deep candidate generation model and then describe a separate deep ranking model.

[OPT][INF] Stegle, O., Lippert, C., Mooij, J. M., Larence, N. D., & Borgwardt, K. (2011). [Efficient inference in matrix-variate Gaussian models with iid observation noise](https://proceedings.neurips.cc/paper/2011/file/a732804c8566fc8f498947ea59a841f8-Paper.pdf). In Proceedings of the Advances in Neural Information Processing Systems 24 (NIPS 2011).

- **keywords**: inverse, inference, matrix-variate Gaussian models
- **memo**: In equation (5), it could effectively compute the inverse of a diagonal matrix plus a Kronecker product.

[LT][OPT] Andersen Ang, [Slides: Nuclear norm is the tightest convex envelop of rank function within the unit ball](https://angms.science/doc/LA/NuclearNorm_cvxEnv_rank.pdf). 

- **keywords**: nuclear norm, rank, convex envelop
- **memo**: Find/prove nuclear norm is the tightest convex envelop of rank. The same argument can be used for other nonconvex and discontinuous regularization.

[OPT][SW] Ge, J., Li, X., Jiang, H., Liu, H., Zhang, T., Wang, M., & Zhao, T. (2019). [Picasso: A Sparse Learning Library for High Dimensional Data Analysis in R and Python.](https://www.jmlr.org/papers/volume20/17-722/17-722.pdf) J. Mach. Learn. Res., 20(44), 1-5. [ [Github](https://github.com/jasonge27/picasso) + [Docs](https://hmjianggatech.github.io/picasso/index.html) ]

- **keywords**: sparse regression, scad, MCP
- **memo**: A Python/R library for sparse regression, including Lasso, SCAD, and MCP.

[OPT] Zou, H., & Li, R. (2008). [One-step sparse estimates in nonconcave penalized likelihood models](https://arxiv.org/pdf/0808.1012.pdf). Annals of statistics, 36(4), 1509.

- **keywords**: SCAD, local linear approximation (LLA)
- **memo**: Solve the SCAD by repeatedly solving Lasso in (2.7).

[BIO][METH][CI] Windmeijer, F., Farbmacher, H., Davies, N., & Davey Smith, G. (2019). [On the use of the lasso for instrumental variables estimation with some invalid instruments](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1498346). Journal of the American Statistical Association, 114(527), 1339-1350.

- **keywords**: adaptive lasso, 2SLS, causal inference, invalid IV
- **memo**: Use adaptive lasso to select invalid IVs in 2SLS.

[CI][METH] Egami, N., Fong, C. J., Grimmer, J., Roberts, M. E., & Stewart, B. M. (2018). [How to make causal inferences using texts](https://scholar.princeton.edu/sites/default/files/bstewart/files/ais.pdf). arXiv preprint arXiv:1802.02163.

- **keywords**: text, causal inference
- **memo**: Causal inference based on textual data, text could be treatment or outcome.

[METH][LT] Natarajan, N., Dhillon, I. S., Ravikumar, P. K., & Tewari, A. (2013). [Learning with noisy labels](https://papers.nips.cc/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf). Advances in neural information processing systems, 26, 1196-1204.

- **keywords**: noisy labels, unbalanced-loss
- **memo**: Model the noisy labels by class-conditional random noise model (CCN). Based on CCN, the authors find that the minimizer of classification with noisy labels is drifted Bayes rule: which coincides with the Bayes rule of unbalanced loss.

[METH][DL] Jeremy Jordan, 2018. [An overview of semantic image segmentation.](https://www.jeremyjordan.me/semantic-segmentation/#loss) 

- **keywords**: image segmentation, Dice loss
- **memo**: A introduction for image segmentation, including background, existing methods and loss functions. 

[ES][DATA][BIO] Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., ... & Menze, B. H. (2021). [clDice-a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://arxiv.org/pdf/2003.07311.pdf). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 16560-16569).

- **keywords**: image segmentation, Dice loss, topology-preservation
- **memo**: A novel Dice-based loss function for medical image segmentation. The motivation is topology-preservation and skeleta of vessels in medical image. Moreover, the trackable computing losses are proposed with an ad-hoc manner. 

[ES][DL] Peng, H., Mou, L., Li, G., Chen, Y., Lu, Y., & Jin, Z. (2015). [A comparative study on regularization strategies for embedding-based neural networks](https://aclanthology.org/D15-1252.pdf). arXiv preprint arXiv:1508.03721.

- **keywords**: regularization, embedding
- **memo**: A comparative empirical study (Experiment A and B) for different regularization in embedding-based neural networks, including (i) l2-reg for other layers (**BOTH WORK**); (ii) l2-reg for an embedding layer (**A WORKS**); (iii) re-embedding words: l2-reg in difference on an embedding layer and a pre-trained layer (**NO WORKS**); (iv) Dropout for other layers (**BOTH WORK**). 

[CI][INF] Feder, A., Keith, K. A., Manzoor, E., Pryzant, R., Sridhar, D., Wood-Doughty, Z., ... & Yang, D. (2021). [Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond](https://arxiv.org/pdf/2109.00725.pdf). arXiv preprint arXiv:2109.00725.

- **keywords**: causal inference, NLP, **survey**
- **memo**: (i) Background of CI; (ii) Text as treatment, outcome, or confounder; (iii) CI -> ML prediction;

[METH][LT] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). Deep sets. arXiv preprint arXiv:1703.06114.

- **keywords**: permutation invariance; learning with set
- **memo**: (i) permutation invariance iff the learning model can be express as a sum function;

[METH][LT] Cheng, J., Levina, E., Wang, P., & Zhu, J. (2014). [A sparse Ising model with covariates](https://doi.org/10.1111/biom.12202). Biometrics, 70(4), 943-953.

- **keywords**: Ising model; label dependence
- **memo**: (i) extend the dependence in Ising model to be a function of features;

[LT][DL] Bartlett, P., Foster, D. J., & Telgarsky, M. (2017). [Spectrally-normalized margin bounds for neural networks](https://arxiv.org/pdf/1706.08498.pdf). arXiv preprint arXiv:1706.08498.

- **keywords**: covering number, Rademacher complexity, estimation error bound
- **memo**: The estimation error bounds for neural networks based on covering number and Rademacher complexity

[LT][DL] Bauer, B., & Kohler, M. (2019). [On deep learning as a remedy for the curse of dimensionality in nonparametric regression](https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-4/On-deep-learning-as-a-remedy-for-the-curse-of/10.1214/18-AOS1747.full). The Annals of Statistics, 47(4), 2261-2285.

- **keywords**: regret bound, estimation error, approximation error
- **memo**: Both estimation error and approximation error (Theorems 2-3) are provided in the paper.

[LT][DL] Guo, Z. C., Shi, L., & Lin, S. B. (2019). [Realizing data features by deep nets](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8924927). IEEE Transactions on Neural Networks and Learning Systems, 31(10), 4036-4048.

- **keywords**: covering number, estimation error bound
- **memo**: VC-type covering number for neural networks

[LT][METH] Mazumder, R., Hastie, T., & Tibshirani, R. (2010). [Spectral regularization algorithms for learning large incomplete matrices](https://www.jmlr.org/papers/volume11/mazumder10a/mazumder10a.pdf). The Journal of Machine Learning Research, 11, 2287-2322.

- **keywords**: soft-impute, low-rank, nuclear norm
- **memo**: (i) Solving low-rank regression by soft-thresholded SVD. (ii) Relation between low-rank regression and latent factor model or matrix factorization in Section 8 (Theorem 3) is quite interesting.

[LT][METH][INF] Alex Stephenson, [Standard Errors and the Delta Method](https://www.alexstephenson.me/post/2022-04-02-standard-errors-and-the-delta-method/).

- **keywords**: delta method, asymptotic distribution, standard error
- **memo**: Asymptotic distribution of functions over a random variable; when the asymptotic behavior of this random variable is obtained.

[INF] Aaron Mishkin, [Instrumental Variables, DeepIV, and Forbidden Regressions](https://cs.stanford.edu/~amishkin/assets/slides/instrumental_variables.pdf)

- **keywords**: instrumental Variables, causal inference

[INF] Vovk, V., & Wang, R. (2020). [Combining p-values via averaging](https://doi.org/10.1093/biomet/asaa027). Biometrika, 107(4), 791-808.

- **keywords**: Combining p-values; 

[OPT] Iusem, A. N. (2003). [On the convergence properties of the projected gradient method for convex optimization](https://www.scielo.br/j/cam/a/jkfWkT6CJb9G3bDqH5SLpMy/?format=pdf&lang=en). Computational & Applied Mathematics, 22, 37-52.

- **keywords**: projected gradient method, convex optimization
- **memo**: Proposition 4 shows that projected GD convergence to stationary point (if the cluster point exists) when objective function is continuously differentiable, and the feasible domain is convex.

[METH][OPT] [Some tensor algebra](https://staffwww.dcs.shef.ac.uk/people/H.Lu/MSL/MSLbook-Chapter3.pdf)

- **keywords**: Mode-product, tensor-matrix multiplication, tensor-vector multiplication

[OPT] Andersen, M., Dahl, J., Liu, Z., Vandenberghe, L., Sra, S., Nowozin, S., & Wright, S. J. (2011). [Interior-point methods for large-scale cone programming](http://www.seas.ucla.edu/~vandenbe/publications/mlbook.pdf). Optimization for machine learning, 5583.

- **keywords**: cone programming, QP, interior-point methods

[OPT] Tibshirani, R. J., [Coordinate Descent](https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/22-coord-desc.pdf).

- **keywords**: Coordinate Descent
- **memo**: The CD algorithm and its convergence rate under different assumptions.

[METH] Jansche, M. (2007, June). [A maximum expected utility framework for binary sequence labeling](https://aclanthology.org/P07-1093.pdf). In Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics (pp. 736-743).

- **keywords**: F-score, label-dependence

[LT] Pillai, I., Fumera, G., & Roli, F. (2017). [Designing multi-label classifiers that maximize F measures: State of the art](https://www.sciencedirect.com/science/article/pii/S0031320316302217). Pattern Recognition, 61, 394-404.

- **keywords**: F-score, decision-theoretic approach
- **memo**: A recent survey for F-score maximization

[OPT] Stephen Boyd and Jon Dattorro, [Alternating Projections](https://web.stanford.edu/class/ee392o/alt_proj.pdf).

- **keywords**: alternating projection, 
- **memo**: AP is an algorithm computing a point in the intersection of some convex sets (or smallest distance of two sets).

[INF][METH] Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). [A kernel two-sample test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf?ref=https://githubhelp.com). Journal of Machine Learning Research, 13(1), 723-773.

- **keywords**: MMD, kernel methods, two-sample test, integral probability metric, hypothesis testing
- **memo**: non-parametric distribution discrepancy test based on MMD.

[METH][DL] Sundeep, T. [Knowledge Distillation: Principles, Algorithms, Applications](https://neptune.ai/blog/knowledge-distillation). neptuneblog.

- **keywords**: knowledge distillation
- **memo**: introduction for knowledge distillation: online/offline knowledge distillation; existing methods

[OPT] Powell, Michael JD. "On search directions for minimization algorithms." Mathematical programming 4, no. 1 (1973): 193-201.

- **keywords**: BCD, block coordinate descent
- **memo**: When F is nonconvex, BCD may cycle and stagnate

[METH][OPT] Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018). [DAGs with NO TEARS: Continuous Optimization for Structure Learning](https://arxiv.org/pdf/1803.01422.pdf) Advances in Neural Information Processing Systems, 31.

- **keywords**: DAG, matrix power
- **memo**: convert DAG constrains as one matrix power equality constraint. 

[OPT] Shalev-Shwartz, S., & Zhang, T. (2012). [Stochastic dual coordinate ascent methods for regularized loss minimization](http://tongzhang-ml.org/papers/jmlr13-sdca.pdf). Journal of Machine Learning Research.

- **keywords**: stochastic dual coordinate ascent, liblinear, coordinate descent
- **memo**: simultaneously update primal/dual variables; generalize the algorithm from liblinear

[OPT] Glasmachers, T., & Dogan, U. (2013, October). [Accelerated coordinate descent with adaptive coordinate frequencies](http://proceedings.mlr.press/v29/Glasmachers13.pdf). In Asian Conference on Machine Learning (pp. 72-86). PMLR.

- **keywords**: coordinate descent
- **memo**: Using Adaptive Coordinate Frequencies to update coordinates

[OPT] Zimmert, J., de Witt, C. S., Kerg, G., & Kloft, M. (2015, December). [Safe screening for support vector machines](http://opt-ml.org/papers/OPT2015_paper_38.pdf). In NIPS 2015 Workshop on Optimization in Machine Learning (OPT).

- **keywords**: screening, shrinking, coordinate descent
- **memo**: screening the shrinking variables for boxed QP in coordinate desecent: when the gradient is non-zeros then the variable is in the boundary.

[OPT][LT] Raginsky, M., Rakhlin, A., & Telgarsky, M. (2017, June). [Non-convex learning via stochastic gradient langevin dynamics: a nonasymptotic analysis.](http://proceedings.mlr.press/v65/raginsky17a/raginsky17a.pdf) In Conference on Learning Theory (pp. 1674-1703). PMLR.

- **keywords**: SGD, global solution, langevin process, SDE

[CI][METH][OPT][LT] Kang, H., Zhang, A., Cai, T. T., & Small, D. S. (2016). [Instrumental variables estimation with some invalid instruments and its application to Mendelian randomization](https://arxiv.org/pdf/1401.5755.pdf). Journal of the American statistical Association, 111(513), 132-144.

- **keywords**: 2SLS, invalid IV, sparse regression
- **memo**: introduce sparse regression in the second stage of 2SLS in separate the effect from invalid IVs

[METH][LT] Dalalyan, A. S., & Tsybakov, A. B. (2007). [Aggregation by exponential weighting and sharp oracle inequalities](http://imagine.enpc.fr/~dalalyan/Download/DTcolt07.pdf). COLT 2007, San Diego, CA, USA; June 13-15, 2007. Proceedings 20 (pp. 97-111). Springer Berlin Heidelberg.

- **keywords**: model aggregate, bagging
- **memo**: Theorem 2 indicates model aggregate is close to the best performance based on model selection

[LT] Chen, J. (2017). [Consistency of the MLE under mixture models](https://projecteuclid.org/journals/statistical-science/volume-32/issue-1/Consistency-of-the-MLE-under-Mixture-Models/10.1214/16-STS578.full).

[LT] Chen, J., & Tan, X. (2009). [Inference for multivariate normal mixtures](https://doi.org/10.1016/j.jmva.2008.12.005). Journal of Multivariate Analysis, 100(7), 1367-1383.

- **keywords**: Nonparametric MLE, identfiability, penalized MLE
- **memo**: Conditions for consistency of Nonparametric MLE under Mixture Model. (i) Identifiability of Mixture Model is a necessary condition. (ii) Most existing general approaches do NOT apply to normal mixture models. (A) Theorem 3.1: Under the finite normal mixture model with *equal variance* and #Group is known, the MLE is strongly consistent. (B) Section 3.2: Under the finite normal mixture model with *unequal variance* and #Group is known, the MLE may NOT consistent. Yet, this issue can be solved via *penalzied* (on variance) MLE. (C) Section 3.3: The proper estimation of the mixing distribution under a finite mixture model requires a very large sample size when the *subpopulations are not well separated*.

[LT] Steinwart, I. (2005). [Consistency of support vector machines and other regularized kernel classifiers](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1377497). IEEE transactions on information theory, 51(1), 128-142.

- **keywords**: universal consistency
- **memo**: Proposition 3.3: (Classification) Fisher consistency and classification calibration are equivalent.

[CI][INF] Fabian Dablanderm [An introduction to Causal inference](https://fabiandablander.com/r/Causal-Inference.html)

- **keywords**: causal inference

[CI] Lal, A., Lockhart, M., Xu, Y., & Zu, Z. (2023). [How much should we trust instrumental variable estimates in political science? Practical advice based on over 60 replicated studies](https://arxiv.org/pdf/2303.11399.pdf). arXiv preprint arXiv:2303.11399.

- **keywords**: causal inference, instrumental variable

[LT][OPT][METH] Loh, P. L., & Wainwright, M. J. (2013). [Regularized M-estimators with nonconvexity: Statistical and algorithmic theory for local optima](https://www.jmlr.org/papers/volume16/loh15a/loh15a.pdf). JMLR.

- **keywords**: high-dimensional statistics, M-estimation, model selection, nonconvex optimization, nonconvex regularization
- **memo**: ``local minimizer = global minimizer'': any stationary point of the composite objective function will lie within statistical precision of the underlying parameter vector.

[METH][OPT] S. Sawyer (2004). [Linear Rank Regression](https://www.math.wustl.edu/~sawyer/handouts/RankRegress.pdf)

- **memo**: motivation: the correlation btw `rank of res` and `res` should be close to zero.
