# DDA2001 Introduction to Data Science - Statictics 

> **Course**: DDA2001 Introduction to Data Science, CUHK(SZ)
>
> **Instructor**: Shuang Li
>
> **Scope**: Lecture 9-12

---
## Contents

0. [Concept Map](#0-concept-map)
1. [The Basic Idea of Statistics](#1-the-basic-idea-of-statistics)
2. [Point Estimation](#2-point-estimation)
3. [Maximum Likelihood Estimation(MLE)](#3-maximum-likelihood-estimation)
4. [MLE for Some Common Distribution](#4-mle-for-some-common-distribution)
5. [Regression Analysis](#5-regression-analysis)
6. [Linear Regression and MLE](#6-linear-regression-and-mle)
7. [Residual Analysis](#7-residual-analysis)
8. [Confidence Interval and Central Limit Theorem](#8-confidence-interval-and-central-limit-theorem)
9. [Formula Sheet](#9-formula-sheet)
10. [Final Remarks](#10-final-remarks)

---

## 0. Concept Map

The central workflow is:

```text
Data
  ↓
Candidate Models
  ↓
Criterion / Likelihood / Loss
  ↓
Estimator
  ↓
Point Estimation
  ↓
Uncertainty Quantification
  ↓
Confidence Interval
```

---

# 1. The Basic Idea of Statistics

## 1.1 Probability vs Statistics

### Probability

If the true model is known, we can predict the behavior of random samples.

Example:

If $X \sim Bernoulli(0.8)$, then

$$
P(X=1)=0.8
$$

### Statistics

If samples are observed, we try to infer the underlying true model.

Example:

If $100$ trials produce $81$ successes, a natural estimate for the success probability is

$$
\hat p = \frac{81}{100}=0.81
$$

### Intuition

Probability is like knowing how a machine works and predicting its output.

Statistics is like observing the output and inferring how the machine works.

---

## 1.2 What Does Statistics Need?

A statistical problem usually needs three ingredients:

1. Samples
2. A set of possible models
3. A criterion for measuring model performance

Notation:

- Samples: $X_1,X_2,\dots,X_n$
- Unknown parameter: $\theta$
- Parameter space: $\Theta$
- Candidate models: $\{P_\theta:\theta\in\Theta\}$

The main question is:

> Which value of  $\theta$  is most reasonable given the observed samples?

---

## 1.3 Parameter, Estimator, and Estimate

### Parameter

A parameter is an unknown constant in a model.

Examples:

- $p$ in a Bernoulli distribution
- $\mu$ and $\sigma^2$ in a normal distribution
- $\lambda$ in an exponential distribution
- $\beta_0,\beta_1$ in linear regression

### Estimator

An estimator is a function of the samples:

$$
\hat\theta=g(X_1,X_2,\dots,X_n)
$$

Example:

$$
\bar X=\frac{1}{n}\sum_{i=1}^nX_i
$$

is an estimator for the population mean $\mu$.
  
### Estimate

An estimate is the numerical value obtained after plugging in observed data.

If the data are $2,4,6$, then

$$
\bar X=4
$$

Here, $\bar X$ is the estimator, and $4$ is the estimate.

---

## 1.4 Statistic

A statistic is any function of the samples that extracts useful information.

Examples:

- Sample mean
- Sample variance
- Sample proportion
- Regression slope
- Regression intercept

---

# 2. Point Estimation

## 2.1 What Is Point Estimation?

Point estimation uses a single number to estimate an unknown parameter.

Examples:

$$
\bar X \quad \text{estimates } \mu
$$

$$
\hat p=\frac{M}{N} \quad \text{estimates } p
$$

$$
\hat\beta_0,\hat\beta_1 \quad \text{estimate regression coefficients}
$$

### Advantage

It is simple and easy to compute.

### Limitation

It does not quantify uncertainty.

Example:

- Drug 1: 9 successes out of 10 trials, $\hat p=0.9$
- Drug 2: 8000 successes out of 10000 trials, $\hat p=0.8$

The first point estimate is larger, but it is much less reliable because it is based on only 10 trials.

This motivates confidence intervals.

---

## 2.2 Desirable Properties of Estimators

Although the lectures mainly focus on MLE and confidence intervals, it is helpful to understand what makes an estimator good.

### Unbiasedness

$$
E[\hat\theta]=\theta
$$

On average, the estimator hits the true parameter.

### Small Variance

The estimator should not fluctuate too much across samples.

### Consistency

As the sample size grows,

$$
\hat\theta \rightarrow \theta
$$

This captures the idea that more data should lead to more accurate estimation.

---

# 3. Maximum Likelihood Estimation

## 3.1 Core Idea

Maximum Likelihood Estimation asks:

> Which parameter value makes the observed data most likely?

Given samples

$$
X_1,X_2,\dots,X_n
$$

and a parameter

$$
\theta \in \Theta
$$

define the likelihood function

$$
L(\theta)=P(X_1,X_2,\dots,X_n\mid\theta)
$$

The MLE is

$$
\hat\theta=\arg\max_{\theta\in\Theta}L(\theta)
$$

---

## 3.2 Likelihood vs Probability

The expression

$$
P(Data\mid \theta)
$$

can be interpreted in two ways.

### Probability

The parameter $\theta$ is fixed, and the data are random.

### Likelihood

The data are fixed, and the expression is viewed as a function of $\theta$.

This distinction is conceptually important.

MLE does not compute $P(\theta\mid Data)$. It maximizes $P(Data\mid\theta)$.

---
  
## 3.3 Why Do We Multiply Densities or Probabilities?

If the samples are **independent**, then the joint probability is a product.

### Discrete case

$$
L(\theta)=\prod_{i=1}^nP(X_i\mid\theta)
$$

### Continuous case

$$
L(\theta)=\prod_{i=1}^nf(X_i\mid\theta)
$$

where $f$ is the probability density function.

---

## 3.4 Log-Likelihood

Products are often difficult to differentiate. Therefore, we use the logarithm:

$$
\ell(\theta)=\log L(\theta)
$$

Since $\log$ is increasing,
 
$$
\arg\max L(\theta)=\arg\max \ell(\theta)
$$

The logarithm converts products into sums:

$$
\log\prod_i a_i=\sum_i\log a_i
$$
---

## 3.5 General MLE Procedure

```text
Step 1: Specify the model.
Step 2: Write the likelihood L(θ).
Step 3: Write the log-likelihood l(θ).
Step 4: Differentiate with respect to θ.
Step 5: Set the derivative equal to zero.
Step 6: Check whether the solution maximizes the likelihood.
Step 7: Report the MLE.
```

---

# 4. MLE for Some Common Distribution

## 4.1 MLE for the Bernoulli Distribution

### 4.1.1 Setup

A drug experiment can be modeled as Bernoulli trials.

- Success: $X=1$
- Failure: $X=0$
- Success probability: $p$
- Number of trials: $N$
- Number of successes: $M$

$$
X_i\sim Bernoulli(p)
$$

---

### 4.1.2 Likelihood

If there are $M$ successes and $N-M$ failures,

$$
L(p)=p^M(1-p)^{N-M}
$$

Log-likelihood:

$$
\ell(p)=M\log p+(N-M)\log(1-p)
$$

---

### 4.1.3 Derivation

$$
\ell'(p)=\frac{M}{p}-\frac{N-M}{1-p}
$$

Set it equal to zero:
   
$$
\frac{M}{p}=\frac{N-M}{1-p}
$$

$$
M(1-p)=p(N-M)
$$

$$
M=Np
$$

Therefore,

$$
\hat p=\frac{M}{N}
$$

---

### 4.1.4 Interpretation

The MLE of a Bernoulli success probability is the sample success rate.

If 81 out of 100 trials are successful,

$$
\hat p=0.81
$$

---

### 4.1.5 MLE with Additional Information

Suppose the parameter space is discrete:

- Drug 1: $p=0.8$
- Drug 2: $p=0.9$

Observed data:

$$
M=81,\quad N=100
$$

Compute

$$
\ell(p)=81\log p+19\log(1-p)
$$

Compare $\ell(0.8)$ and $\ell(0.9)$.

The model with the larger likelihood is more likely to have generated the data.

### Key Point

If the parameter space is discrete, compare candidate likelihoods directly instead of differentiating.

---

## 4.2 MLE for the Exponential Distribution

### 4.2.1 Model

Suppose light bulb lifetimes follow an exponential distribution:

$$
X_i\sim Exponential(\lambda)
$$

with PDF

$$
f(x\mid\lambda)=\lambda e^{-\lambda x},\quad x\ge0
$$

---

### 4.2.2 Likelihood

$$
L(\lambda)=\prod_{i=1}^n\lambda e^{-\lambda X_i}
$$

$$
L(\lambda)=\lambda^n e^{-\lambda\sum_iX_i}
$$

Log-likelihood:

$$
\ell(\lambda)=n\log\lambda-\lambda\sum_{i=1}^nX_i
$$

---

### 4.2.3 Derivation

$$
\ell'(\lambda)=\frac{n}{\lambda}-\sum_iX_i
$$

Set it equal to zero:

$$
\frac{n}{\lambda}=\sum_iX_i
$$

$$
\hat\lambda=\frac{n}{\sum_iX_i}=\frac{1}{\bar X}
$$

---

### 4.2.4 Interpretation

For an exponential distribution,

$$
E[X]=\frac{1}{\lambda}
$$

A larger sample mean implies a smaller rate $\lambda$.

---

## 4.3 MLE for the Mean of a Normal Distribution

### 4.3.1 Model

Assume

$$
X_i\sim N(\mu,1)
$$

where the variance is known and the unknown parameter is $\mu$.

---

### 4.3.2 Likelihood

$$
f(x_i\mid\mu)=\frac{1}{\sqrt{2\pi}}\exp\left[-\frac{(x_i-\mu)^2}{2}\right]
$$

$$
L(\mu)=\prod_{i=1}^n
\frac{1}{\sqrt{2\pi}}\exp\left[-\frac{(x_i-\mu)^2}{2}\right]
$$

Log-likelihood:

$$
\ell(\mu)=C-\frac12\sum_{i=1}^n(x_i-\mu)^2
$$

where $C$ does not depend on $\mu$.

---

### 4.3.3 Maximizing the Likelihood

Maximizing the log-likelihood is equivalent to minimizing

$$
\sum_{i=1}^n(x_i-\mu)^2
$$

Differentiate:

$$
\frac{d}{d\mu}\sum_i(x_i-\mu)^2=-2\sum_i(x_i-\mu)
$$

Set equal to zero:

$$
\sum_i(x_i-\mu)=0
$$

$$
\hat\mu=\bar X
$$

---

### 4.3.4 Key Takeaway

For a normal distribution with known variance, the MLE of the mean is the sample mean.

However, the MLE is not always the sample mean for every distribution.

> Think about a question: If the variance of a normal distribution is unknown, what is the MLE.
>
> You will learn more about this kind of situation in course STA2002

---

## 4.4 MLE for the Uniform Distribution

### 4.4.1 Model

Let

$$
X\sim Uniform(0,\theta)
$$

The PDF is

$$
f(x\mid\theta)=
\begin{cases}
1/\theta, & 0\le x\le\theta\\
0, & otherwise
\end{cases}
$$

---

### 4.4.2 Likelihood

If

$$
\theta < \max_iX_i
$$

then at least one observed sample is impossible, so

$$
L(\theta)=0
$$

If

$$
\theta\ge\max_iX_i
$$

then

$$
L(\theta)=\frac{1}{\theta^n}
$$

Since $1/\theta^n$ decreases as $\theta$ increases, the maximum occurs at the smallest feasible value:

$$
\hat\theta=\max(X_1,\dots,X_n)
$$

---

### 4.4.3 Why Calculus May Fail

The maximum occurs at a boundary or discontinuity, not at an interior point where the derivative is zero.

This example shows that MLE is not always solved by setting $\ell'(\theta)=0$.

--- 

## 4.5 Summary of MLE

| Model | Unknown Parameter | MLE |
|---|---|---|
| Bernoulli(p) | $p$ | $M/N=\bar X$ |
| Exponential($\lambda$) | $\lambda$ | $1/\bar X$ |
| Normal($\mu,1$) | $\mu$ | $\bar X$ |
| Uniform(0,$\theta$) | $\theta$ | $\max_iX_i$ |

---

## 4.6 Common Pitfalls

### Pitfall 1

MLE maximizes

$$
P(Data\mid\theta)
$$

not

$$
P(\theta\mid Data)
$$

### Pitfall 2

The product form of likelihood assumes independent samples.

### Pitfall 3

Do not ignore parameter boundaries.

### Pitfall 4

If the parameter space is discrete, compare likelihood values directly.

---

# 5. Regression Analysis

## 5.1 What Is Regression?

Regression analysis studies the relationship between $X$ and $Y$.

It answers questions such as:

- Does $Y$ increase or decrease as $X$ increases?
- By how much does $Y$ change when $X$ increases by one unit?
- Can we predict $Y$ from $X$?

A simple linear model is

$$
Y=\beta_0+\beta_1X
$$

---

## 5.2 Correlation vs Regression

### Correlation

Measures the strength and direction of linear association.

### Regression

Builds a predictive model.

Regression is not only about whether variables move together, but also about estimating the size of the relationship.

---

## 5.3 Simple Linear Regression Model

The lecture uses the model

$$
Y\mid X\sim N(\beta_0+\beta_1X,\sigma^2)
$$

This means that for a given $X=x$,

$$
E[Y\mid X=x]=\beta_0+\beta_1x
$$

and the noise is normal:

$$
Y-\beta_0-\beta_1X\sim N(0,\sigma^2)
$$

---

## 5.4 Interpretation of Parameters

### Intercept $\beta_0$

The predicted mean of $Y$ when $X=0$.

### Slope $\beta_1$

The expected change in $Y$ when $X$ increases by one unit.

- $\beta_1>0$: positive linear trend
- $\beta_1<0$: negative linear trend
- $\beta_1=0$: no linear trend

---

# 6. Linear Regression and MLE

## 6.1 Probabilistic Model

$$
Y_i=\beta_0+\beta_1X_i+\epsilon_i
$$

where

$$
\epsilon_i\sim N(0,\sigma^2)
$$

Thus,

$$
Y_i\mid X_i\sim N(\beta_0+\beta_1X_i,\sigma^2)
$$

---

## 6.2 Likelihood

For observations

$$
(X_1,Y_1),\dots,(X_n,Y_n)
$$

the likelihood is

$$
L(\beta_0,\beta_1)=\prod_{i=1}^n
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left[
-\frac{(Y_i-\beta_0-\beta_1X_i)^2}{2\sigma^2}
\right]
$$

The log-likelihood is

$$
\ell(\beta_0,\beta_1)
=C-\frac{1}{2\sigma^2}\sum_{i=1}^n(Y_i-\beta_0-\beta_1X_i)^2
$$

---

## 6.3 MLE Equals Least Squares

Maximizing the log-likelihood is equivalent to minimizing

$$
SSE=\sum_{i=1}^n(Y_i-\beta_0-\beta_1X_i)^2
$$

Therefore, under the normal error model,

$$
\text{MLE}=\text{Least Squares Estimator}
$$

---

## 6.4 Normal Equations

The first-order conditions are

$$
\sum_i(Y_i-\beta_0-\beta_1X_i)=0
$$

$$
\sum_i(Y_i-\beta_0-\beta_1X_i)X_i=0
$$

Solving gives

$$
\hat\beta_1=
\frac{\sum_i(X_i-\bar X)(Y_i-\bar Y)}
{\sum_i(X_i-\bar X)^2}
$$

$$
\hat\beta_0=\bar Y-\hat\beta_1\bar X
$$

---

## 6.5 Intuition for the Slope

The slope is a covariance-like quantity divided by a variance-like quantity.

If large $X$ tends to occur with large $Y$, the slope is positive.

If large $X$ tends to occur with small $Y$, the slope is negative.

---

# 7. Residual Analysis

## 7.1 Residual

The residual is

$$
e_i=Y_i-\hat Y_i
$$

where

$$
\hat Y_i=\hat\beta_0+\hat\beta_1X_i
$$

Thus,

$$
e_i=Y_i-\hat\beta_0-\hat\beta_1X_i
$$

---

## 7.2 Why Residual Analysis Matters

Fitting a line does not guarantee that the model assumptions are valid.

Residual analysis checks whether the fitted regression model is reasonable.

---

## 7.3 Key Assumptions

### Linearity

The relationship between $X$ and the mean of $Y$ should be approximately linear.

If residuals show a curved pattern, the linearity assumption may fail.

### Constant Variance / Homoscedasticity

The variance of the error should not depend on $X$.

If residuals form a funnel shape, the constant variance assumption may fail.

---

## 7.4 How to Interpret Residual Plots

- Random scatter around zero: reasonable fit
- Curved pattern: nonlinearity
- Funnel shape: non-constant variance
- Extreme points: possible outliers

---

# 8. Confidence Interval and Central Limit Theorem

## 8.1 Confidence Interval

A confidence interval is a random interval constructed from data:

$$
[a(X_1,\dots,X_n),b(X_1,\dots,X_n)]
$$

A 95% confidence interval has the following repeated-sampling interpretation:

> If we repeated the sampling process many times and constructed an interval each time, about 95% of those intervals would contain the true parameter.

---

## 8.2 Important Interpretation

For a particular computed interval, the true parameter is either inside or outside.

Strictly speaking, the probability statement is about the random procedure, not about the fixed parameter.

A practical interpretation is:

> We are 95% confident that the interval contains the true value.

---

## 8.3 Central Limit Theorem

If $X_1,\dots,X_n$ have mean $\mu$ and variance $\sigma^2$, then for large $n$,

$$
\bar X\approx N\left(\mu,\frac{\sigma^2}{n}\right)
$$

Equivalently,

$$
\frac{\sqrt n(\bar X-\mu)}{\sigma}\approx N(0,1)
$$

---

# 9. Formula Sheet

## Basic Statistics

$$
\bar X=\frac{1}{n}\sum_{i=1}^nX_i
$$

$$
\hat p=\frac{M}{N}
$$

---

## MLE

$$
L(\theta)=P(X_1,\dots,X_n\mid\theta)
$$

$$
\ell(\theta)=\log L(\theta)
$$

Discrete independent samples:

$$
L(\theta)=\prod_iP(X_i\mid\theta)
$$

Continuous independent samples:

$$
L(\theta)=\prod_if(X_i\mid\theta)
$$

---

## Bernoulli MLE

$$
L(p)=p^M(1-p)^{N-M}
$$

$$
\hat p=\frac{M}{N}
$$

---

## Exponential MLE

$$
\hat\lambda=\frac{1}{\bar X}
$$

---

## Normal Mean MLE

$$
\hat\mu=\bar X
$$

---

## Uniform MLE

$$
\hat\theta=\max_iX_i
$$

---

## Linear Regression

$$
Y_i=\beta_0+\beta_1X_i+\epsilon_i
$$

$$
\hat\beta_1=
\frac{\sum_i(X_i-\bar X)(Y_i-\bar Y)}
{\sum_i(X_i-\bar X)^2}
$$

$$
\hat\beta_0=\bar Y-\hat\beta_1\bar X
$$

$$
e_i=Y_i-\hat\beta_0-\hat\beta_1X_i
$$

---

## CLT

$$
\bar X\approx N\left(\mu,\frac{\sigma^2}{n}\right)
$$

$$
\frac{\sqrt n(\bar X-\mu)}{\sigma}\approx N(0,1)
$$

---

# Final Remarks

The statistics part of the course is built around three key ideas:

1. **MLE:** Choose the parameter that makes the observed data most likely.
2. **Regression:** Under normal errors, MLE is equivalent to least squares.
3. **Confidence Intervals:** Quantify uncertainty beyond a single point estimate.