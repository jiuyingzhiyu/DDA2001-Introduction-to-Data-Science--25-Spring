# DDA2001 Introduction to Data Science — Optimization 

> **Course**: DDA2001 Introduction to Data Science, CUHK(SZ)
>
> **Instructor**: Shuang Li
>
> **Scope**: Lecture 14-20, including optimization formulation, convex sets, convex functions, convex optimization, linear programming, newsvendor models, and proof-based questions.

---
## Contents

0. [Concept Map](#0-concept-map)
1. [Optimization Formulation](#1-optimization-formulation)
2. [Feasible Region and Optimality](#2-feasible-region-and-optimality)
3. [Convex Sets](#3-convex-sets)
4. [Convex Functions](#4-convex-functions)
5. [Concave Functions](#5-concave-functions)
6. [Operations Preserving Convexity](#6-operations-preserving-convexity)
7. [Convex Optimization](#7-convex-optimization)
8. [Answer Templates](#8-answer-templates)
9. [Formula Sheet](#9-formula-sheet)
10. [Final Remarks](#10-final-remarks)

---

## 0. Concept Map

The whole probability module can be understood through one chain:

```text
Optimization Formulation
  ↓
Decision Variable / Objective / Constraints
  ↓
Feasible Region
  ↓
Local Optimum vs Global Optimum
  ↓
Convex Set
  ↓
Convex Function
  ↓
Convexity-Preserving Operations
  ↓
Convex Optimization
  ↓
Applications: Linear Regression, Newsvendor, Linear Programming
  ↓
Proof Techniques and Exam Questions
```

The most important theorem is:

> In convex optimization, every local minimum is also a global minimum.

---

# 1. Optimization Formulation

## 1.1 The Three Elements

An optimization problem usually contains:

1. Decision variables
2. Objective function
3. Constraints

Standard form:

$$
\begin{aligned}
\min_{x\in\mathbb{R}^n}\quad & f_0(x)\\
\text{s.t.}\quad & f_i(x)\le0,\quad i=1,\dots,m\\
& h_i(x)=0,\quad i=1,\dots,p
\end{aligned}
$$

where:

- $x$ is the decision variable
- $f_0(x)$ is the objective or cost function
- $f_i(x)\le0$ are inequality constraints
- $h_i(x)=0$ are equality constraints

---

## 1.2 Pricing with Linear Demand

Suppose a company sells a product at price $x$.

Demand is modeled as:

$$
D(x)=ax+b
$$

where:

- $a<0$: price sensitivity
- $b>0$: base demand

Revenue is:

$$
f(x)=xD(x)=ax^2+bx
$$

If \(a=-1,b=10\), then

$$
f(x)=-x^2+10x
$$

The optimization problem is:

$$
\max_{x\ge0}-x^2+10x
$$

### Three Elements

| Element | Content |
|---|---|
| Decision variable | price $x$ |
| Objective | revenue $xD(x)$ |
| Constraint | $x\ge0$ |

---

## 1.3 Maximization vs Minimization

A maximization problem

$$
\max_x f(x)
$$

can be converted into

$$
\min_x -f(x)
$$

Thus:

- maximizing a concave function is equivalent to minimizing a convex function
- minimizing a convex function is the standard convex optimization form

---

# 2. Feasible Region and Optimality

## 2.1 Feasible Region

The feasible region is the set of all points satisfying the constraints:

$$
X=\{x:f_i(x)\le0,\ h_j(x)=0\}
$$

A point in $X$ is called feasible.

---

## 2.2 Global Optimum

A feasible point $x^\star$ is a global minimum if

$$
f(x^\star)\le f(x),\quad \forall x\in X
$$

For maximization, the inequality is reversed.

---

## 2.3 Local Optimum

A feasible point $x^\star$ is a local minimum if there exists a small neighborhood around it in which no feasible point has a smaller objective value.

Formally, there exists $\epsilon>0$ such that

$$
f(x^\star)\le f(x)
$$

for all feasible $x$ satisfying

$$
\|x-x^\star\|\le\epsilon
$$

---

## 2.4 Why Local vs Global Matters

In non-convex problems, local optima may not be global.

In convex optimization:

$$
\text{local minimum}=\text{global minimum}
$$

This is the key reason convexity is important.

---

# 3. Convex Sets

## 3.1 Definition

A set $C$ is convex if the line segment between any two points in $C$ stays inside $C$.

Mathematically, for any $x_1,x_2\in C$ and $\theta\in[0,1]$,

$$
\theta x_1+(1-\theta)x_2\in C
$$

---

## 3.2 Convex Combination

$$
x_\theta=\theta x_1+(1-\theta)x_2
$$

where

$$
0\le\theta\le1
$$

This represents a point on the line segment between $x_1$ and $x_2$.

---

## 3.3 Common Convex Sets

### Empty set, singleton, and whole space

$$
\emptyset,\quad \{x_0\},\quad \mathbb{R}^n
$$

are convex.

### Interval

$$
[a,b]\subset\mathbb{R}
$$

is convex.

### Hyperplane

$$
H=\{x\in\mathbb{R}^n:a^Tx=c\}
$$

is convex.

### Halfspace

$$
H=\{x\in\mathbb{R}^n:a^Tx\le c\}
$$

is convex.

### Ball

$$
B=\{x:\|x\|_2\le r\}
$$

is convex.

---

## 3.4 How to Prove a Set Is Convex

### Method 1: Definition

Pick arbitrary $x_1,x_2\in C$ and prove

$$
\theta x_1+(1-\theta)x_2\in C
$$

### Method 2: Geometry

Useful in one or two dimensions.

### Method 3: Intersection

The intersection of convex sets is convex.

### Method 4: Epigraph

If a set is the epigraph of a convex function, it is convex.

---

## 3.5 Halfspace Proof Template

Let

$$
H=\{(x,y):y\le ax+b\}
$$

Take two points in $H$:

$$
y_1\le ax_1+b,\quad y_2\le ax_2+b
$$

For the convex combination,

$$
y_\theta=\theta y_1+(1-\theta)y_2
$$

Then

$$
y_\theta
\le
\theta(ax_1+b)+(1-\theta)(ax_2+b)
$$

$$
=a[\theta x_1+(1-\theta)x_2]+b
$$

$$
=ax_\theta+b
$$

Thus the halfspace is convex.

---

## 3.6 Intersections Preserve Convexity

If $C_1$ and $C_2$ are convex, then

$$
C_1\cap C_2
$$

is convex.

Proof:

Take $x_1,x_2\in C_1\cap C_2$. Since both sets are convex,

$$
\theta x_1+(1-\theta)x_2\in C_1
$$

and

$$
\theta x_1+(1-\theta)x_2\in C_2
$$

so it belongs to the intersection.

---

## 3.7 Unions Do Not Preserve Convexity

Even if $C_1$ and $C_2$ are convex,

$$
C_1\cup C_2
$$

may not be convex.

Example:

$$
C_1=[0,1],\quad C_2=[2,3]
$$

Their union is not convex because the midpoint of $0.5$ and $2.5$ is $1.5$, which is not in the union.

---

# 4. Convex Functions

## 4.1 Definition

A function $f:\mathbb{R}^n\to\mathbb{R}\cup\{+\infty\}$ is convex if:

1. $\operatorname{dom}(f)$ is convex
2. for any $x_1,x_2\in\operatorname{dom}(f)$ and $\lambda\in[0,1]$,

$$
f(\lambda x_1+(1-\lambda)x_2)
\le
\lambda f(x_1)+(1-\lambda)f(x_2)
$$

---

## 4.2 Geometric Interpretation

A function is convex if the line segment connecting any two points on its graph lies above the function curve.

---

## 4.3 Epigraph

The epigraph of $f$ is

$$
\operatorname{epi}(f)=\{(x,t):t\ge f(x)\}
$$

The key theorem is:

$$
f\text{ is convex}
\iff
\operatorname{epi}(f)\text{ is convex}
$$

---

## 4.4 Absolute Value Is Convex

For $f(x)=|x|$,

$$
|\lambda x_1+(1-\lambda)x_2|
\le
|\lambda x_1|+|(1-\lambda)x_2|
$$

$$
=\lambda|x_1|+(1-\lambda)|x_2|
$$

Therefore $|x|$ is convex.

---

## 4.5 Second-Order Condition

If $f:\mathbb{R}\to\mathbb{R}$ is twice differentiable, then

$$
f\text{ is convex}\iff f''(x)\ge0
$$

Examples:

### Linear function

$$
f(x)=ax+b
$$

$$
f''(x)=0
$$

Thus it is both convex and concave.

### Quadratic function

$$
f(x)=x^2
$$

$$
f''(x)=2>0
$$

Thus it is convex.

### Exponential function

$$
f(x)=e^x
$$

$$
f''(x)=e^x>0
$$

Thus it is convex.

---

## 4.6 Second-Order Condition: Multivariate

For $f:\mathbb{R}^n\to\mathbb{R}$, define

$$
F(t)=f(x+te)
$$

for arbitrary point $x$ and direction $e$.

If

$$
F''(t)\ge0
$$

for every $x,e,t$, then $f$ is convex.

Equivalently, if the Hessian exists,

$$
\nabla^2 f(x)\succeq0
$$

for all $x$.

---

## 4.7 Example

Show

$$
f(x_1,x_2)=x_1^2+3x_2^2
$$

is convex.

Let $e=(e_1,e_2)$ and

$$
F(t)=f(x_1+te_1,x_2+te_2)
$$

Then

$$
F(t)=(x_1+te_1)^2+3(x_2+te_2)^2
$$

$$
F''(t)=2e_1^2+6e_2^2\ge0
$$

Therefore $f$ is convex.

---

# 5. Concave Functions

## 5.1 Definition

A function $f$ is concave if

$$
f(\lambda x_1+(1-\lambda)x_2)
\ge
\lambda f(x_1)+(1-\lambda)f(x_2)
$$

---

## 5.2 Relation with Convexity

$$
f\text{ is concave}
\iff
-f\text{ is convex}
$$

Thus maximizing a concave function is equivalent to minimizing a convex function.

---

## 5.3 Second-Order Condition

For a twice differentiable univariate function,

$$
f\text{ is concave}\iff f''(x)\le0
$$

Examples:

$$
f(x)=-x^2
$$

is concave.

$$
f(x)=\log x
$$

is concave on $(0,\infty)$.

$$
f(x)=\sin x
$$

is concave on $[0,\pi]$.

---

# 6. Operations Preserving Convexity

## 6.1 Positive Scaling

If $f$ is convex and $c>0$, then

$$
cf(x)
$$

is convex.

---

## 6.2 Sums and Nonnegative Weighted Sums

If $f_1,\dots,f_m$ are convex and $w_i\ge0$, then

$$
\sum_{i=1}^m w_i f_i(x)
$$

is convex.

---

## 6.3 Affine Transformation

If $f$ is convex, then

$$
g(x)=f(Ax+b)
$$

is convex.

Vertical shifts do not affect convexity.

---

## 6.4 Pointwise Maximum

If $f_1,\dots,f_m$ are convex, then

$$
f(x)=\max\{f_1(x),\dots,f_m(x)\}
$$

is convex.

The same holds for infinitely many convex functions.

---

# 7. Convex Optimization

## 7.1 Standard Form

$$
\min_{x\in X} f(x)
$$

where:

- $f$ is convex
- $X$ is convex

---

## 7.2 Local Minimum Is Global Minimum

Theorem:

If $f$ is convex and $X$ is convex, then any local minimum is a global minimum.

### Proof Sketch

Suppose $x_0$ is local but not global.

Then there exists $x_1\in X$ such that

$$
f(x_1)<f(x_0)
$$

For $\lambda\in(0,1)$, define

$$
x_\lambda=(1-\lambda)x_0+\lambda x_1
$$

Since $X$ is convex,

$$
x_\lambda\in X
$$

By convexity,

$$
f(x_\lambda)
\le
(1-\lambda)f(x_0)+\lambda f(x_1)
$$

Since $f(x_1)<f(x_0)$,

$$
f(x_\lambda)<f(x_0)
$$

For small $\lambda$, $x_\lambda$ is arbitrarily close to $x_0$, contradicting local optimality.

---

## 7.3 Convex Minimization vs Concave Maximization

If $f$ is convex,

$$
\min_{x\in X}f(x)
$$

is a convex optimization problem.

If $g$ is concave,

$$
\max_{x\in X}g(x)
$$

is also a convex optimization problem because it is equivalent to

$$
\min_{x\in X}-g(x)
$$

---

## 7.4 Linear Regression as Convex Optimization

### 7.4.1 Setup

Given samples

$$
(X_1,Y_1),\dots,(X_N,Y_N)
$$

consider the linear model

$$
Y_i=\beta_1X_i+\beta_0+\epsilon_i
$$

Estimate $\beta_0,\beta_1$ by minimizing

$$
SSE(\beta_0,\beta_1)
=
\sum_i(\beta_1X_i+\beta_0-Y_i)^2
$$

---

### 7.4.2 Convexity of SSE

Take any direction $(e_0,e_1)$. Define

$$
F(t)=\sum_i[(\beta_1+te_1)X_i+(\beta_0+te_0)-Y_i]^2
$$

Then

$$
F''(t)=2\sum_i(e_1X_i+e_0)^2\ge0
$$

Thus the SSE objective is convex.

---

### 7.4.3 Data Science Meaning

Linear regression is a convex optimization problem.

Therefore, local optimization methods can find a global solution, and the model is computationally stable.

---

## 7.5 Newsvendor Problem

### 7.5.1 Setup

A newsvendor or bakery must decide the order quantity before random demand is known.

Decision variable:

$$
q
$$

Random demand:

$$
D
$$

Parameters:

- $p$: selling price
- $c$: purchase or production cost
- $s$: salvage value
- $q$: order quantity

---

### 7.5.2 Sold Quantity and Leftover Quantity

Sold quantity:

$$
\min(q,D)
$$

Leftover quantity:

$$
(q-D)^+=\max(q-D,0)
$$

---

### 7.5.3 Expected Profit

A general profit function is

$$
\pi(q)
=
pE[\min(q,D)]
+sE[(q-D)^+]
-cq
$$

If salvage value is zero,

$$
\pi(q)=pE[\min(q,D)]-cq
$$

The optimization problem is

$$
\max_{q\ge0}\pi(q)
$$

or equivalently

$$
\min_{q\ge0}-\pi(q)
$$

---

### 7.5.4 Why It Is Convex Optimization

The key identity is

$$
-\min(q,D)=\max(-q,-D)
$$

This is the maximum of affine functions in $q$, so it is convex.

Using nonnegative weighted sums and expectations, $-\pi(q)$ is convex.

The constraint $q\ge0$ defines a halfspace, hence the feasible set is convex.

---

## 7.6 Linear Programming as Convex Optimization

### 7.6.1 Example

$$
\begin{aligned}
\max\quad & 7T+5C\\
\text{s.t.}\quad & 3T+4C\le2400\\
& 2T+C\le1000\\
& C\le450\\
& T\ge100\\
& T,C\ge0
\end{aligned}
$$

---

### 7.6.2 Convexity

The objective is linear.

The feasible region is an intersection of halfspaces.

Therefore, the problem is a convex optimization problem.

---

# 8. Answer Templates

## Proving a Set Is Convex

```text
Pick arbitrary x1,x2∈C and θ∈[0,1].
Let xθ=θx1+(1-θ)x2.
Verify that xθ satisfies all defining constraints of C.
Therefore xθ∈C.
Hence C is convex.
```

---

## Proving a Function Is Convex by Definition

```text
First, dom(f) is convex.
Pick arbitrary x1,x2∈dom(f), θ∈[0,1].
Let z=θx1+(1-θ)x2.
Show f(z)≤θf(x1)+(1-θ)f(x2).
Therefore f is convex.
```

---

## Proving a Function Is Convex by Second-Order Condition

```text
Compute f''(x).
Since f''(x)≥0 on the domain, f is convex.
```

Multivariate:

```text
Pick arbitrary point x and direction e.
Let F(t)=f(x+te).
Compute F''(t).
Since F''(t)≥0 for all x,e,t, f is convex.
```

---

## Proving a Problem Is Convex Optimization

```text
The objective function is convex.
The feasible region is convex because it is an intersection of convex sets.
Therefore the problem is a convex optimization problem.
Hence any local minimum is also a global minimum.
```

---

## Writing an Optimization Formulation

```text
Decision variable: ...
Objective function: ...
Constraints: ...

The optimization problem is:

maximize/minimize ...
subject to ...
```

---

# 9. Formula Sheet

## Optimization Standard Form

$$
\begin{aligned}
\min_x\quad & f_0(x)\\
\text{s.t.}\quad & f_i(x)\le0,\quad i=1,\dots,m\\
& h_i(x)=0,\quad i=1,\dots,p
\end{aligned}
$$

---

## Convex Set

$$
x_1,x_2\in C,\ \theta\in[0,1]
\Rightarrow
\theta x_1+(1-\theta)x_2\in C
$$

---

## Convex Function

$$
f(\theta x_1+(1-\theta)x_2)
\le
\theta f(x_1)+(1-\theta)f(x_2)
$$

---

## Concave Function

$$
f(\theta x_1+(1-\theta)x_2)
\ge
\theta f(x_1)+(1-\theta)f(x_2)
$$

---

## Epigraph

$$
\operatorname{epi}(f)=\{(x,t):t\ge f(x)\}
$$

---

## Second-Order Conditions

Univariate convex:

$$
f''(x)\ge0
$$

Univariate concave:

$$
f''(x)\le0
$$

Multivariate convex:

$$
\nabla^2f(x)\succeq0
$$

Directional version:

$$
\frac{d^2}{dt^2}f(x+te)\ge0
$$

---

## SSE

$$
SSE(\beta_0,\beta_1)
=
\sum_i(\beta_1X_i+\beta_0-Y_i)^2
$$

---

## Newsvendor

$$
\pi(q)=pE[\min(q,D)]+sE[(q-D)^+]-cq
$$

$$
(q-D)^+=\max(q-D,0)
$$

---

## Airline Pricing

$$
P(\text{buy})=1-F(p)
$$

$$
D\sim Binomial(n,1-F(p))
$$

$$
E[D]=n(1-F(p))
$$

$$
\max_{p\ge1}(p-c)n(1-F(p))
$$

---

# 10. Final Remarks

The optimization part is built around three levels:

1. **Formulation:** identify the decision variable, objective, and constraints.
2. **Convexity:** determine whether the feasible region is convex and whether the objective is convex or concave.
3. **Optimization conclusion:** if the problem is convex, local minima are global minima.

Most exam questions are proof-based or formulation-based:

- prove a set is convex
- prove a function is convex
- determine whether a problem is convex optimization
- formulate a real-world problem
- explain local and global optimality