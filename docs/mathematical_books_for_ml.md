# Mathematical Books for Machine Learning

*For someone with a Bachelor's degree in Mathematics*

---

## Tier 1: Core Theory (Start Here)

### 1. Understanding Machine Learning: From Theory to Algorithms
*Shalev-Shwartz & Ben-David* (2014)

| | |
|---|---|
| **Why** | The most rigorous introduction to computational learning theory |
| **Covers** | PAC learning, VC dimension, Rademacher complexity, boosting theory, SVM, neural networks |
| **Style** | Definition → Theorem → Proof (pure math style) |
| **Free PDF** | [cs.huji.ac.il/~shais/UnderstandingMachineLearning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf) |

> *"Rigorous but principled... builds from foundational theoretical ideas through to practical algorithmic paradigms"* — Cambridge University Press

---

### 2. High-Dimensional Statistics: A Non-Asymptotic Viewpoint
*Martin J. Wainwright* (2019)

| | |
|---|---|
| **Why** | Modern ML deals with p >> n; this is THE book for that |
| **Covers** | Concentration inequalities, empirical processes, random matrices, sparse models, LASSO theory |
| **Style** | Graduate-level measure theory + probability |
| **Level** | Advanced (assumes real analysis, probability theory) |

> *"A must-read book for all graduate students in both mathematical statistics and mathematical machine learning"* — Larry Wasserman

---

### 3. Convex Optimization
*Boyd & Vandenberghe* (2004)

| | |
|---|---|
| **Why** | Every ML algorithm optimizes something; this explains HOW |
| **Covers** | Convex sets/functions, duality, gradient descent, Newton's method, interior-point methods |
| **Style** | Rigorous with complete proofs |
| **Free PDF** | [stanford.edu/~boyd/cvxbook](https://stanford.edu/~boyd/cvxbook/) |

> *"One of the best optimization textbooks... a gentle, but rigorous, introduction"*

---

## Tier 2: Deep Foundations

### 4. Statistical Learning Theory
*Vladimir Vapnik* (1998, Wiley)

| | |
|---|---|
| **Why** | THE foundational text by the inventor of VC theory and SVMs |
| **Covers** | VC dimension proofs, structural risk minimization, kernel methods |
| **Style** | Dense, theorem-heavy, complete proofs |
| **Note** | Start with "The Nature of Statistical Learning Theory" (Springer) for overview first |

---

### 5. Elements of Information Theory
*Cover & Thomas* (2nd ed, 2006)

| | |
|---|---|
| **Why** | Cross-entropy loss, KL divergence, mutual information — all from here |
| **Covers** | Entropy, channel capacity, rate-distortion, source coding |
| **Style** | Classic textbook, rigorous proofs |
| **Connection** | Explains WHY we use log-loss, cross-entropy in classification |

---

### 6. The Elements of Statistical Learning (ESL)
*Hastie, Tibshirani, Friedman* (2nd ed, 2009)

| | |
|---|---|
| **Why** | The "bible" — covers everything from linear regression to boosting |
| **Covers** | Regression, classification, trees, boosting, SVMs, neural nets, ensemble methods |
| **Style** | Statistical perspective, less theorem-proof, more derivations |
| **Free PDF** | [stanford.edu/~hastie/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/) |

> Chapter 10 (Boosting) explains the math behind XGBoost/CatBoost

---

## Tier 3: Advanced/Specialized

### 7. Probabilistic Machine Learning: Advanced Topics
*Kevin P. Murphy* (2023)

| | |
|---|---|
| **Why** | Most comprehensive modern treatment (1200+ pages) |
| **Covers** | Deep learning, Bayesian inference, generative models, causality, decision theory |
| **Style** | Rigorous with extensive references |
| **Free Draft** | [probml.github.io/pml-book/book2](https://probml.github.io/pml-book/book2.html) |

> *"Kevin Murphy has distilled the vast and confusing literature... into a beautifully written and extremely clear textbook"* — Geoffrey Hinton

---

### 8. Pattern Recognition and Machine Learning (PRML)
*Christopher Bishop* (2006)

| | |
|---|---|
| **Why** | Best Bayesian/probabilistic treatment |
| **Covers** | Graphical models, EM algorithm, variational inference, kernel methods |
| **Style** | Mathematically rigorous, Bayesian throughout |

---

## Topic → Book Mapping

| What You Want to Understand | Book |
|-----------------------------|------|
| **Why AUC, precision, recall work** | ESL Ch. 7, Shalev-Shwartz Ch. 26 |
| **Theory behind boosting (XGBoost, CatBoost)** | ESL Ch. 10, Shalev-Shwartz Ch. 10 |
| **Why we use cross-entropy loss** | Cover & Thomas Ch. 2, 5 |
| **How gradient descent converges** | Boyd & Vandenberghe Ch. 9 |
| **VC dimension, generalization bounds** | Vapnik, Shalev-Shwartz Ch. 3-6 |
| **Why regularization prevents overfitting** | Wainwright Ch. 7, Shalev-Shwartz Ch. 13 |
| **LASSO, sparse models theory** | Wainwright Ch. 7 |
| **Bayesian inference, priors** | Bishop, Murphy |
| **Neural network theory** | Shalev-Shwartz Ch. 20, Murphy |

---

## Recommended Reading Order

For someone with a Math Bachelor's:

```
1. Understanding Machine Learning (Shalev-Shwartz)
   └── Rigorous foundation, PAC learning, VC theory

2. Convex Optimization (Boyd) — Ch. 1-5, 9
   └── How optimization actually works

3. Elements of Statistical Learning — Ch. 7, 10, 15
   └── Practical algorithms, boosting, ensemble methods

4. Elements of Information Theory (Cover) — Ch. 2, 5
   └── Why cross-entropy, KL divergence

5. High-Dimensional Statistics (Wainwright)
   └── Modern theory for high-dimensional settings

6. Vapnik (if you want the deepest theory)
   └── Original proofs of fundamental theorems
```

---

## Free Resources Summary

| Book | Free PDF |
|------|----------|
| Understanding Machine Learning | [Link](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf) |
| Convex Optimization | [Link](https://stanford.edu/~boyd/cvxbook/) |
| Elements of Statistical Learning | [Link](https://hastie.su.domains/ElemStatLearn/) |
| Probabilistic ML (Murphy) | [Link](https://probml.github.io/pml-book/) |

---

## Sources

- [Understanding Machine Learning - Cambridge](https://www.cambridge.org/core/books/understanding-machine-learning/3059695661405D25673058E43C8BE2A6)
- [High-Dimensional Statistics - Cambridge](https://www.cambridge.org/core/books/highdimensional-statistics/8A91ECEEC38F46DAB53E9FF8757C7A4E)
- [Convex Optimization - Stanford](https://stanford.edu/~boyd/cvxbook/)
- [Statistical Learning Theory - Vapnik](https://www.wiley.com/en-us/Statistical+Learning+Theory-p-9780471030034)
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/)
- [The Blunt Guide to Mathematically Rigorous ML](https://medium.com/technomancy/the-blunt-guide-to-mathematically-rigorous-machine-learning-c53263d45c7b)
