# DDA2001 Introduction to Data Science — Probability 

> **Course**: DDA2001 Introduction to Data Science, CUHK(SZ)
> **Instructor**: Shuang Li
> **Scope**: Lecture 3–8, including elementary probability theory, random variables, discrete distributions, continuous random variables, correlation, conditional probability, and Pearson correlation calculation examples.  

---
## Contents

0. [Concept Map](#0-concept-map)
1. [Probability Fundamentals](#1-probability-fundamentals)
2. [Random Variables and Distributions](#2-random-variables-and-distributions)
3. [Expectation and Variance](#3-expectation-and-variance)
4. [Common Discrete Distributions](#4-common-discrete-distributions)
5. [Continuous Distributions](#5-continuous-distributions-uniform-normal-and-monte-carlo)
6. [Joint Distribution, Covariance, and Correlation](#6-joint-distribution-covariance-and-correlation)
7. [Conditional Probability and Bayes Rule](#7-conditional-probability-and-bayes-rule)
8. [Exam Templates](#8-exam-templates)
9. [Foemula Sheet](#9-formula-sheet)
10. [Final Remarks](#10-final-remarks)
11. [Reference Material](#11-reference-material)

---

## 0. Concept Map

The whole probability module can be understood through one chain:

```text
Uncertainty
  ↓
Random Experiment
  ↓
Sample Space Ω / S
  ↓
Event = subset of the sample space
  ↓
Probability Function
  ↓
Random Variable X: maps outcomes to numbers
  ↓
Distribution: PMF / PDF / CDF
  ↓
Expectation: long-run average
  ↓
Variance: fluctuation / risk
  ↓
Common Distributions: Bernoulli / Binomial / Geometric / Uniform / Normal
  ↓
Joint Distribution
  ↓
Covariance and Correlation
  ↓
Conditional Probability and Bayes Rule
  ↓
Data Science Applications: classification, recommendation, risk, Monte Carlo, PCA, Naive Bayes
```

The lectures are not isolated topics. They answer three major questions:

1. **How do we describe uncertainty?**  
   By using sample spaces, events, probabilities, random variables, and distributions.

2. **How do we summarize random quantities?**  
   By using expectation, variance, CDFs, and standard distributions.

3. **How do we describe relationships and information updates?**  
   By using joint distributions, correlation, conditional probability, and Bayes Rule.

---

# 1. Probability Fundamentals 

## 1.1 What is Probability?

Probability is a mathematical tool for measuring uncertainty. It describes how likely an event is to occur.

Examples:

- the probability that it rains tomorrow;
- the probability that a stock earns a positive return;
- the probability that a randomly selected student has a birthday on January 1;
- the probability that a classifier labels an email as spam.

A key intuition in the course is the **frequentist interpretation**:

If an experiment is repeated \(N\) times and event \(A\) occurs \(n(A)\) times, then for large \(N\),

$$
P(A)\approx \frac{n(A)}{N}.
$$

More formally,

$$
P(A)=\lim_{N\to\infty}\frac{n(A)}{N}.
$$

Thus, probability is not about the exact result of a single trial. It is about stable long-run frequencies.

---

## 1.2 Probability vs Statistics

The lectures emphasize an important distinction:

| Field | Input | Output | Example |
|---|---|---|---|
| Probability | Known model / data-generating process | Properties of outcomes | Given a fair die, how often should we see a 2? |
| Statistics | Observed data | Inference about the model | Given many die rolls, is the die fair? |

In short:

```text
Probability: model → data
Statistics:  data  → model
```

Data science uses both. Statistical inference learns models from data, while probability quantifies uncertainty in predictions.

---

## 1.3 Random Experiment

A random experiment is a repeatable procedure that may produce different outcomes even when repeated under the same conditions.

Examples:

- tossing a coin;
- rolling a die;
- randomly selecting a student;
- whether a machine breaks down tomorrow;
- whether a user clicks an advertisement;
- whether a message contains the word “happy”.

A process is a random experiment if:

1. it is repeatable;
2. the outcome is not deterministic.

---

## 1.4 Outcome and Sample Space

### Outcome

An outcome is one possible result of a random experiment, usually denoted by Ω .

For one die roll, Ω in {1,2,3,4,5,6\}.

### Sample Space

The sample space is the set of all possible outcomes, denoted by \(\Omega\) or \(S\).

Examples:

- One die roll:

$$
\Omega=\{1,2,3,4,5,6\}.
$$

- Two coin tosses:

$$
\Omega=\{HH,HT,TH,TT\}.
$$

- Selecting two items from \(\{a,b,c\}\) without replacement, if order matters:

$$
\Omega=\{ab,ac,ba,bc,ca,cb\}.
$$

If order does not matter, the sample space becomes:

$$
\Omega=\{\{a,b\},\{a,c\},\{b,c\}\}.
$$

### Important Exam Point

The sample space depends on whether order matters and whether sampling is with or without replacement.

---

## 1.5 Events

An event is a subset of the sample space.

For two coin tosses,

$$
\Omega=\{HH,HT,TH,TT\}.
$$

The event “at least one head” is

$$
A=\{HH,HT,TH\}.
$$

Events can be described in words, set notation, or random variable notation.

For example, if \(X\) is the sum of two dice, the event “the sum is larger than 10” can be written as

$$
\{\omega:X(\omega)>10\}
$$

or simply

$$
\{X>10\}.
$$

---

## 1.6 Probability Function  

In a discrete sample space, a probability function assigns a probability to every outcome.

It must satisfy:

1. Non-negativity:

$$
P(\omega)\ge 0.
$$

2. Total probability equals 1:

$$
\sum_{\omega\in\Omega}P(\omega)=1.
$$

3. The probability of an event is the sum of probabilities of its outcomes:

$$
P(A)=\sum_{\omega\in A}P(\omega).
$$

Therefore, probabilities can never be negative or greater than 1.

---

## 1.7 Set Operations

Since events are sets, we can apply set operations.

| Concept | Words | Notation | Meaning |
|---|---|---|---|
| Union | A or B | \(A\cup B\) | At least one of A or B occurs |
| Intersection | A and B | \(A\cap B\) | Both A and B occur |
| Complement | not A | \(A^c\) or \(A'\) | A does not occur |

The addition rule is

$$
P(A\cup B)=P(A)+P(B)-P(A\cap B).
$$

The subtraction is needed because the intersection is counted twice.

---

## 1.8 Mutually Exclusive Events

Events \(A\) and \(B\) are mutually exclusive if

$$
A\cap B=\emptyset.
$$

They cannot occur at the same time.

If \(A\) and \(B\) are mutually exclusive,

$$
P(A\cup B)=P(A)+P(B).
$$

Example: in one die roll, “getting 1” and “getting 6” are mutually exclusive.

---

## 1.9 Independence

Events \(A\) and \(B\) are independent if

$$
P(A\cap B)=P(A)P(B).
$$

Intuition: knowing that one event occurred does not change the probability of the other event.

Equivalently, if \(P(B)>0\),

$$
P(A|B)=P(A).
$$

### Independence vs Mutual Exclusiveness

| Concept | Meaning | Formula | Intuition |
|---|---|---|---|
| Mutually exclusive | Cannot occur together | \(A\cap B=\emptyset\) | A excludes B |
| Independent | No probabilistic influence | \(P(A\cap B)=P(A)P(B)\) | A gives no information about B |

Nonzero mutually exclusive events are usually not independent.

---

## 1.10 Birthday Paradox

Question: in a room with \(n\) people, what is the probability that at least two people share a birthday?

Use the complement:

$$
P(\text{at least one shared birthday})=1-P(\text{all birthdays different}).
$$

Ignoring leap years,

$$
P(\text{all different})=\frac{365}{365}\cdot\frac{364}{365}\cdot\frac{363}{365}\cdots\frac{365-n+1}{365}.
$$

Thus,

$$
P(\text{at least one match})=1-\prod_{k=0}^{n-1}\frac{365-k}{365}.
$$

At \(n=23\), the probability is already around 50%. At \(n=100\), it is over 99%.

The birthday paradox mainly tests complement reasoning and multiplication rules.

---

# 2. Random Variables and Distributions

## 2.1 Why Random Variables?

Outcomes are often not numerical, but statistics and data science usually require numerical quantities.

Example: after three coin tosses, the outcome may be \(HTH\), but we may care about the number of heads.

Define

$$
X=\text{number of heads}.
$$

A random variable is a function

$$
X:\Omega\to\mathbb{R}.
$$

It maps each outcome \(\omega\) to a numerical value \(X(\omega)\).

---

## 2.2 Range 

The range of a random variable is the set of values it can take.

Example: if \(X\) is the sum of two dice, then

$$
X\in\{2,3,4,5,6,7,8,9,10,11,12\}.
$$

---

## 2.3 Discrete Random Variables

A random variable is discrete if it takes a finite or countably infinite set of values.

Examples:

- dice outcome;
- number of heads;
- number of customers arriving in a day;
- number of trials until the first success.

---

## 2.4 Probability Mass Function (PMF)

For a discrete random variable \(X\), its PMF is

$$
f(x)=P(X=x).
$$

It must satisfy:

$$
f(x_i)\ge 0
$$

and

$$
\sum_i f(x_i)=1.
$$

Example: for a fair die,

$$
P(X=i)=\frac{1}{6},\quad i=1,2,3,4,5,6.
$$

Example: if \(X\) is the number of heads in two coin tosses,

| X | 0 | 1 | 2 |
|---|---|---|---|
| Probability | 1/4 | 1/2 | 1/4 |

---

## 2.5 Cumulative Distribution Function (CDF)

The CDF is defined as

$$
F_X(x)=P(X\le x).
$$

For a discrete random variable,

$$
F_X(x)=\sum_{x_i\le x}f(x_i).
$$

The CDF is useful for threshold events, such as:

- loss is no more than 1000;
- machine lifetime is less than 3 hours;
- exam score is below 60;
- fewer than three components are operational.

Properties:

1. \(0\le F(x)\le 1\);
2. \(F(x)\) is nondecreasing;
3. \(F(x)\to0\) as \(x\to-\infty\);
4. \(F(x)\to1\) as \(x\to+\infty\).

---

## 2.6 Continuous Random Variables

A continuous random variable can take any value in an interval or a union of intervals.

Examples:

- temperature;
- time;
- height;
- stock return;
- task completion time;
- sensor measurement.

The most important property is

$$
P(X=x)=0.
$$

This does not mean that \(X=x\) is impossible. It means that a single point has zero width and therefore zero probability mass.

For continuous variables, we ask interval questions:

$$
P(0.4\le X\le 0.6),
$$

not point questions such as \(P(X=0.5)\).

---

## 2.7 Probability Density Function (PDF)

A continuous random variable is described by a PDF \(f(x)\).

It must satisfy:

$$
f(x)\ge0
$$

and

$$
\int_{-\infty}^{\infty}f(x)dx=1.
$$

Interval probabilities are computed by integration:

$$
P(a\le X\le b)=\int_a^b f(x)dx.
$$

### PDF is not Probability

A PDF value is density, not probability. Only the area under the PDF over an interval is a probability.

Therefore, a PDF can be greater than 1.

For example,

$$
f(x)=2,\quad x\in[0,0.5]
$$

is valid because

$$
\int_0^{0.5}2dx=1.
$$

---

## 2.8 PMF vs PDF vs CDF

| Item | PMF | PDF | CDF |
|---|---|---|---|
| Used for | Discrete variables | Continuous variables | Both |
| Meaning | Point probability | Density | Cumulative probability |
| Formula | \(P(X=x)\) | \(f(x)\) | \(P(X\le x)\) |
| Probability calculation | Sum | Integral | Difference |
| Can exceed 1? | No | Yes | No |

For continuous variables,

$$
P(a\le X\le b)=F(b)-F(a).
$$

---

# 3. Expectation and Variance

## 3.1 Expectation

Expectation is the long-run average of a random variable.

For a discrete random variable,

$$
E[X]=\sum_x xP(X=x)=\sum_x xf(x).
$$

For a continuous random variable,

$$
E[X]=\int_{-\infty}^{\infty}xf(x)dx.
$$

Example: a fair die game pays 2 dollars if the outcome is greater than 3 and loses 1 dollar otherwise. Let \(X\) be the payoff.

Then

$$
P(X=2)=\frac12,
\quad
P(X=-1)=\frac12.
$$

So

$$
E[X]=2\cdot\frac12+(-1)\cdot\frac12=0.5.
$$

The long-run average gain is 0.5 dollars per game.

---

## 3.2 Linearity of Expectation

One of the most important formulas is

$$
E[X_1+X_2+\cdots+X_n]=E[X_1]+E[X_2]+\cdots+E[X_n].
$$

More generally,

$$
E\left[\sum_i c_iX_i\right]=\sum_i c_iE[X_i].
$$


---


## 3.3 Expectation of \(g(X)\)

If \(g(X)\) is a function of \(X\), then

For discrete \(X\):

$$
E[g(X)]=\sum_xg(x)P(X=x).
$$

For continuous \(X\):

$$
E[g(X)]=\int_{-\infty}^{\infty}g(x)f(x)dx.
$$

### Common Trap

In general,

$$
E[g(X)]\ne g(E[X]).
$$

Example: let \(X=1\) for heads and \(X=-1\) for tails in a fair coin toss.

Then

$$
E[X]=0,
$$

but

$$
E[X^2]=1
$$

while

$$
(E[X])^2=0.
$$

Thus,

$$
E[X^2]\ne(E[X])^2.
$$

---

## 3.4 Variance

Expectation tells us the average level, but not the stability. Variance measures how far a random variable is from its mean on average.

Definition:

$$
Var(X)=E[(X-E[X])^2].
$$

A very useful equivalent formula is

$$
Var(X)=E[X^2]-(E[X])^2.
$$

### Why Square?

If we used \(E[X-E[X]]\), the result would always be zero. Squaring removes signs, pe  nalizes large deviations, and gives a mathematically convenient measure.

---

## 3.5 Standard Deviation

The standard deviation is

$$
SD(X)=\sqrt{Var(X)}.
$$

It has the same unit as the original variable. If returns are measured in dollars, standard deviation is also measured in dollars.

---

## 3.6 Variance Addition

If \(X_1,X_2,\ldots,X_n\) are independent, then

$$
Var(X_1+X_2+\cdots+X_n)=Var(X_1)+Var(X_2)+\cdots+Var(X_n).
$$

Unlike expectation, variance addition generally requires independence.

Also,

$$
Var(aX+b)=a^2Var(X).
$$

---

# 4. Common Discrete Distributions

## 4.1 Bernoulli Distribution

A Bernoulli distribution describes one binary trial: success or failure.

Notation:

$$
X\sim Bernoulli(p).
$$

PMF:

$$
P(X=1)=p,
$$

$$
P(X=0)=1-p.
$$

Mean:

$$
E[X]=p.
$$

Variance:

$$
Var(X)=p(1-p).
$$

Applications:

- whether a customer arrives in a minute;
- whether a machine breaks down on a day;
- whether a user clicks an advertisement;
- whether an email is spam;
- whether a trial succeeds.

Memory: Bernoulli = one yes/no trial.

---

## 4.2 Binomial Distribution

A binomial distribution describes the number of successes in \(N\) independent Bernoulli trials.

Notation:

$$
X\sim Binomial(N,p).
$$

PMF:

$$
P(X=k)=\binom{N}{k}p^k(1-p)^{N-k}.
$$

Mean:

$$
E[X]=Np.
$$

Variance:

$$
Var(X)=Np(1-p).
$$

$$
X=X_1+X_2+\cdots+X_N.
$$

Since each \(X_i\sim Bernoulli(p)\),

$$
E[X]=\sum_iE[X_i]=Np.
$$

Applications:
  
- number of breakdowns in \(N\) days;
- number of users clicking an advertisement;
- number of erroneous bits among \(N\) bits.

Memory: Binomial = total successes in many Bernoulli trials.

---

## 4.3 Geometric Distribution

A geometric distribution describes the trial number of the first success.

Notation:

$$
X\sim Geometric(p).
$$

If \(X\) is the number of the trial on which the first success occurs,

$$
P(X=k)=(1-p)^{k-1}p,
\quad k=1,2,3,\ldots
$$

Mean:

$$
E[X]=\frac{1}{p}.
$$

Variance:

$$
Var(X)=\frac{1-p}{p^2}.
$$

Applications:

- drawing blind boxes until getting the target;
- tossing a coin until the first head;
- waiting for the first customer;
- waiting for the first machine breakdown.

Memory: Geometric = waiting time until first success.
 
---

## 4.4 Relationship among the Three Distributions

| Question | Random Variable | Distribution |
|---|---|---|
| Does a customer arrive in a specific minute? | \(X_i\in\{0,1\}\) | Bernoulli |
| How many customers arrive in the first \(N\) minutes? | \(Y_N=X_1+\cdots+X_N\) | Binomial |
| In which minute does the first customer arrive? | \(Z\) | Geometric |

Similarly for machine breakdowns:

| Question | Distribution |
|---|---|
| Does the machine break down today? | Bernoulli |
| How many breakdowns occur in the first \(N\) days? | Binomial |
| On which day does the first breakdown occur? | Geometric |

---

# 5. Continuous Distributions, Uniform, Normal, and Monte Carlo

## 5.1 Core Idea of Continuous Variables

For a continuous random variable,

$$
P(X=x)=0.
$$

Therefore, we use PDFs and integrals:

$$
P(a\le X\le b)=\int_a^bf(x)dx.
$$

For continuous variables, endpoints do not matter:

$$
P(a<X<b)=P(a\le X\le b)=P(a\le X<b)=P(a<X\le b).
$$

This is because single points have probability zero.

---

## 5.2 Uniform Distribution

A uniform distribution means that all locations in an interval are equally likely in terms of density.

Notation:

$$
X\sim Uniform(a,b).
$$

PDF:

$$
f(x)=\frac{1}{b-a},\quad a\le x\le b,
$$

and \(f(x)=0\) otherwise.

Mean:

$$
E[X]=\frac{a+b}{2}.
$$

Variance:

$$
Var(X)=\frac{(b-a)^2}{12}.
$$

Intuition: the mean is the midpoint of the interval, and the variance grows with interval length.

---

## 5.3 Expectation and Variance for Continuous Variables

The transition from discrete to continuous probability is:

```text
summation → integration
PMF       → PDF
```

Discrete expectation:

$$
E[X]=\sum_xxf(x).
$$

Continuous expectation:

$$
E[X]=\int_{-\infty}^{\infty}xf(x)dx.
$$

Discrete variance:

$$
Var(X)=\sum_x(x-E[X])^2f(x).
$$

Continuous variance:

$$
Var(X)=\int_{-\infty}^{\infty}(x-E[X])^2f(x)dx.
$$

---

## 5.4 Monte Carlo Integration

Suppose we want to approximate

$$
\int_a^b h(x)dx.
$$

Let

$$
X\sim Uniform(a,b).
$$

Then

$$
E[h(X)]=\int_a^b h(x)\frac{1}{b-a}dx.
$$

Thus,

$$
\int_a^b h(x)dx=(b-a)E[h(X)].
$$

Approximate the expectation by sample average:

$$
E[h(X)]\approx \frac{1}{N}\sum_{i=1}^Nh(x_i).
$$

Therefore,

$$
\int_a^b h(x)dx\approx \frac{b-a}{N}\sum_{i=1}^Nh(x_i).
$$

This is the foundation of Monte Carlo integration.

---

## 5.5 Estimating \(\pi\) by Random Sampling

Draw random points uniformly from the square \([-1,1]\times[-1,1]\). The square has area 4, and the unit circle has area \(\pi\).

A point is inside the unit circle if

$$
X^2+Y^2\le1.
$$

The probability of falling inside the circle is approximately

$$
\frac{\pi}{4}.
$$

Therefore,

$$
\pi\approx4\cdot\frac{\text{number of points inside the circle}}{\text{total number of points}}.
$$

---

## 5.6 Normal Distribution

The normal distribution is one of the most important continuous distributions.

Notation:

$$
X\sim Normal(\mu,\sigma^2).
$$

Mean:

$$
E[X]=\mu.
$$

Variance:

$$
Var(X)=\sigma^2.
$$

Normal distributions are widely used to model measurement errors, heights, sample averages, and many aggregated effects.

---

## 5.7 Central Limit Theorem

The course mentions the Central Limit Theorem as the reason why the normal distribution is important.

Let \(X_1,X_2,\ldots,X_n\) be independent and identically distributed with mean \(\mu\) and variance \(\sigma^2\). Define

$$
\bar X=\frac{X_1+\cdots+X_n}{n}.
$$

For large \(n\),

$$
\bar X\approx Normal\left(\mu,\frac{\sigma^2}{n}\right).
$$

Intuition: even if the original data are not normally distributed, the sample average often becomes approximately normal when the sample size is large.

This explains why sample means, estimation errors, and Monte Carlo estimates often behave normally.

---

# 6. Joint Distribution, Covariance, and Correlation

## 6.1 Why Joint Distributions?

A single random variable describes one quantity. In data science, we often care about relationships between two quantities.

Examples:

- temperature \(X\) and air-conditioner usage \(Y\);
- age \(X\) and height \(Y\);
- attendance rate \(X\) and final score \(Y\);
- exercise amount \(X\) and health measure \(Y\).

The realization of one variable may change the distribution of another variable, so we need joint distributions.

---

## 6.2 Joint Distribution

For discrete random variables, the joint PMF is

$$
f(x,y)=P(X=x,Y=y).
$$

It must satisfy

$$
f(x,y)\ge0
$$

and

$$
\sum_x\sum_yf(x,y)=1.
$$

Marginal distributions are obtained by summing out the other variable:

$$
P(X=x)=\sum_yf(x,y),
$$

$$
P(Y=y)=\sum_xf(x,y).
$$

For continuous variables, sums are replaced by integrals.

---

## 6.3 Covariance

Covariance is defined as

$$
Cov(X,Y)=E[(X-E[X])(Y-E[Y])].
$$

An equivalent formula is

$$
Cov(X,Y)=E[XY]-E[X]E[Y].
$$

Intuition:

| Case | \(X-E[X]\) | \(Y-E[Y]\) | Product |
|---|---|---|---|
| X high, Y high | + | + | + |
| X low, Y low | - | - | + |
| X high, Y low | + | - | - |
| X low, Y high | - | + | - |

Positive covariance means the two variables tend to move together. Negative covariance means they tend to move in opposite directions.

---

## 6.4 Correlation

The Pearson correlation coefficient is

$$
\rho_{X,Y}=\frac{Cov(X,Y)}{\sigma_X\sigma_Y},
$$

where

$$
\sigma_X=\sqrt{Var(X)},\quad \sigma_Y=\sqrt{Var(Y)}.
$$

It always satisfies

$$
-1\le\rho\le1.
$$

| Correlation | Meaning |
|---|---|
| 1 | Perfect positive linear relationship |
| close to 1 | Strong positive linear relationship |
| 0 | No clear linear relationship |
| close to -1 | Strong negative linear relationship |
| -1 | Perfect negative linear relationship |

### Important Note

Correlation measures linear relationship only. \(\rho=0\) does not mean there is no relationship at all.

For example, \(Y=X^2\) may have zero linear correlation with \(X\) under a symmetric distribution, but the variables are clearly related.

---

## 6.5 Pearson Correlation Calculation Template

Given data \((x_i,y_i), i=1,\ldots,n\):

1. Compute sample means:

$$
\bar x=\frac{1}{n}\sum_ix_i,
\quad
\bar y=\frac{1}{n}\sum_iy_i.
$$

2. Compute covariance:

$$
Cov(X,Y)=\frac{1}{n}\sum_i(x_i-\bar x)(y_i-\bar y).
$$

3. Compute standard deviations:

$$
SD_X=\sqrt{\frac{1}{n}\sum_i(x_i-\bar x)^2},
$$

$$
SD_Y=\sqrt{\frac{1}{n}\sum_i(y_i-\bar y)^2}.
$$

4. Compute correlation:

$$
r=\frac{Cov(X,Y)}{SD_XSD_Y}.
$$

Example: if \(Y=2X\), such as

| X | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Y | 2 | 4 | 6 | 8 |

then the correlation is 1 because all points lie on a line with positive slope.

---

## 6.6 Correlation Does Not Imply Causation

Correlation does not imply causation.

Example: eating ice cream and using an electric fan may be positively correlated, but neither causes the other. The hidden cause may be hot weather.

This hidden factor is called a confounding variable.

In data science, correlation alone is not enough for causal conclusions. Experiments, randomization, or causal inference are needed.

---

# 7. Conditional Probability and Bayes Rule
> You will leran more details of conditional probability in STA2001H/STA2001

## 7.1 Conditional Probability

Conditional probability describes the probability of an event given that another event has occurred.
 
If \(P(B)>0\), then

$$
P(A|B)=\frac{P(A\cap B)}{P(B)}.
$$

Intuition: once we know \(B\) occurred, the sample space is restricted to \(B\).

---

## 7.2 Multiplication Rule

From the definition,

$$
P(A\cap B)=P(B)P(A|B).
$$

Also,

$$
P(A\cap B)=P(A)P(B|A).
$$

This formula is useful for tree diagrams and joint probability tables.

---

## 7.3 Bayes Rule

Since

$$
P(A\cap B)=P(B|A)P(A)=P(A|B)P(B),
$$

we get

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}.
$$

If \(B\) can occur under multiple cases, use the law of total probability:

$$
P(B)=P(B|A)P(A)+P(B|A^c)P(A^c).
$$

Thus,

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B|A)P(A)+P(B|A^c)P(A^c)}.
$$

---

## 7.4 Medical Testing Example

Suppose

$$
P(Infected)=0.01,
$$

$$
P(Well)=0.99,
$$

$$
P(Positive|Infected)=0.99,
$$

$$
P(Positive|Well)=0.01.
$$

We want \(P(Infected|Positive)\).

Compute joint probabilities:

$$
P(Infected\cap Positive)=0.01\times0.99=0.0099,
$$

$$
P(Well\cap Positive)=0.99\times0.01=0.0099.
$$

So

$$
P(Positive)=0.0099+0.0099=0.0198.
$$

Therefore,

$$
P(Infected|Positive)=\frac{0.0099}{0.0198}=0.5.
$$

Even with a highly accurate test, the posterior probability can be much lower than intuition suggests when the base rate is low.

---

## 7.5 Monty Hall Problem

There are three doors: one car and two goats. You choose one door. The host opens another door with a goat. Should you switch?

Yes.

- Probability of winning if staying: \(1/3\)
- Probability of winning if switching: \(2/3\)

Intuition: your initial choice is wrong with probability \(2/3\). If your initial choice is wrong, switching wins.

---

## 7.6 Conditional Probability and Independence

If \(A\) and \(B\) are independent, then

$$
P(A|B)=P(A).
$$

For random variables, if \(X\) and \(Y\) are independent,

$$
f(y|x)=f(y)
$$

and

$$
f(x,y)=f(x)f(y).
$$

---

# 8. Exam Templates

## 8.1 Sample Space Problems

Steps:

1. Identify the experiment;
2. decide whether order matters;
3. decide whether replacement is allowed;
4. list all outcomes;
5. define events.

Common mistake: confusing an event with the sample space.

---

## 8.2 Event Probability Problems

Steps:

1. Define \(A\) and \(B\);
2. check whether they are mutually exclusive;
3. check whether they are independent;
4. use

$$
P(A\cup B)=P(A)+P(B)-P(A\cap B).
$$

---

## 8.3 PMF Problems

Steps:

1. Find the support of the random variable;
2. compute the probability of each value;
3. check non-negativity;
4. check that probabilities sum to 1.

---

## 8.4 CDF Problems

Steps:

1. Write the PMF or PDF;
2. accumulate probabilities;
3. use summation for discrete variables;
4. use integration for continuous variables;
5. be careful with piecewise definitions.

---

## 8.5 Expectation and Variance Problems

Steps:

1. Compute \(E[X]\);
2. compute \(E[X^2]\);
3. use

$$
Var(X)=E[X^2]-(E[X])^2.
$$

---

## 8.6 Distribution Identification

| Keywords | Distribution |
|---|---|
| yes/no, success/failure | Bernoulli |
| number of successes in \(N\) trials | Binomial |
| first success | Geometric |
| equally likely on an interval | Uniform |
| sample average, error, many small effects | Normal |

---

## 8.7 Conditional Probability Problems

Steps:

1. Determine whether the question asks for \(P(A|B)\) or \(P(B|A)\);
2. use

$$
P(A|B)=\frac{P(A\cap B)}{P(B)};
$$

3. if \(P(B)\) is not directly given, use total probability;
4. watch for base-rate effects.

---

## 8.8 Correlation Problems

Steps:

1. Compute \(\bar x\) and \(\bar y\);
2. compute deviations;
3. compute covariance;
4. compute standard deviations;
5. apply Pearson correlation formula.

---

# 9. Formula Sheet

## Probability

$$
0\le P(A)\le1
$$

$$
P(\Omega)=1
$$

$$
P(A^c)=1-P(A)
$$

$$
P(A\cup B)=P(A)+P(B)-P(A\cap B)
$$

Mutually exclusive:

$$
A\cap B=\emptyset
$$

Independent:

$$
P(A\cap B)=P(A)P(B)
$$

---

## Conditional Probability

$$
P(A|B)=\frac{P(A\cap B)}{P(B)}
$$

$$
P(A\cap B)=P(A)P(B|A)=P(B)P(A|B)
$$

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$

Law of total probability:

$$
P(B)=P(B|A)P(A)+P(B|A^c)P(A^c)
$$

---

## Random Variables

PMF:

$$
f(x)=P(X=x)
$$

CDF:

$$
F(x)=P(X\le x)
$$

PDF interval probability:

$$
P(a\le X\le b)=\int_a^bf(x)dx
$$

---

## Expectation and Variance

$$
E[X]=\sum_xxf(x)
$$

$$
E[X]=\int xf(x)dx
$$

$$
E[g(X)]=\sum_xg(x)f(x)
$$

$$
E[g(X)]=\int g(x)f(x)dx
$$

$$
Var(X)=E[(X-E[X])^2]
$$

$$
Var(X)=E[X^2]-(E[X])^2
$$

$$
E\left[\sum_ic_iX_i\right]=\sum_ic_iE[X_i]
$$

If independent:

$$
Var\left(\sum_iX_i\right)=\sum_iVar(X_i)
$$

---

## Common Distributions

| Distribution | Parameters | Mean | Variance |
|---|---|---|---|
| Bernoulli | \(p\) | \(p\) | \(p(1-p)\) |
| Binomial | \(N,p\) | \(Np\) | \(Np(1-p)\) |
| Geometric | \(p\) | \(1/p\) | \((1-p)/p^2\) |
| Uniform | \(a,b\) | \((a+b)/2\) | \((b-a)^2/12\) |
| Normal | \(\mu,\sigma^2\) | \(\mu\) | \(\sigma^2\) |

---

## Covariance and Correlation

$$
Cov(X,Y)=E[(X-E[X])(Y-E[Y])]
$$

$$
Cov(X,Y)=E[XY]-E[X]E[Y]
$$

$$
\rho=\frac{Cov(X,Y)}{\sigma_X\sigma_Y}
$$

$$
-1\le\rho\le1
$$

---

# 10. Final Remarks

The key to this part of the course is not memorizing formulas mechanically. The key is to identify:

1. what the random experiment is;
2. what the random variable is;
3. whether to use PMF or PDF;
4. whether the question asks for probability, expectation, variance, or conditional probability;
5. whether relationships between two variables are involved.

Once these are clear, most exam questions become direct applications of formulas.

## 11. Reference Material
- Applied Statistics and Probability for Engineers, 3rd Edition, Douglas C. Montgomery & George C. Runger
- Probability and Statistical Inference, 9th edition, Hogg, McKean & Craig
- Havard Stat 110, Joe Blitzstein & Jessica Hwang
