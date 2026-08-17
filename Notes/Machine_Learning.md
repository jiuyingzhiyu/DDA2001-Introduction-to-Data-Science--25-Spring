# DDA2001 Introduction to Data Science — Machine Learning

> **Course**: DDA2001 Introduction to Data Science, CUHK(SZ)
>
> **Instructor**: Shuang Li
>
> **Scope**: Lecture 21–24, Including Introduction to Machine Learning, KNN, Logistic Regression, Unsupervised Learning, Clustering, K-means, Model Selection.

---

## Contents

0. [Concept Map](#0-concept-map)
1. [ML Workflow](#1-machine-learning-workflow)
2. [Supervised Learning](#2-supervised-learning)
3. [K Nearest Neighbours(KNN)](#3-k-nearest-neighbors-knn)
4. [Logistic Regression](#4-logistic-regression)
5. [Unsupervised Learning](#5-unsupervised-learning)
6. [Clustering](#6-clustering)
7. [K-means Clustering](#7-k-means-clustering)
8. [Model Selection](#8-model-selection)
9. [Bias-Variance Tradeoff](#9-bias-variance-tradeoff)
10. [Validation Set Approach](#10-validation-set-approach)
11. [K-fold Cross Validation](#11-k-fold-cross-validation)
12. [Formula Sheet](#12-formula-sheet)
13. [Final Remarks](#13-final-remarks)

---

# 0. Concept Map

Machine Learning is a task-driven field of computer science that builds systems capable of learning from data and improving performance through experience, without being explicitly programmed for every possible case.

A standard definition, following Tom Mitchell, is:

> A computer program is said to learn from experience $E$, with respect to some task $T$, and performance measure $P$, if its performance on $T$, as measured by $P$, improves with experience $E$.

The three key ingredients are:

| Component | Meaning |
|---|---|
| Task $T$ | What we want the machine to do |
| Experience $E$ | The data used for learning |
| Performance $P$ | How we evaluate whether the model is good |

Machine learning connects the three major parts of this course:

| Previous Topic | Role in Machine Learning |
|---|---|
| Probability | Models uncertainty, class probabilities, and randomness |
| Statistics | Helps understand data, evaluate models, and reason about generalization |
| Optimization | Trains models by minimizing loss or maximizing likelihood |

In short:

```text
Data
  ↓
Model
  ↓
Loss / Objective
  ↓
Optimization
  ↓
Prediction
  ↓
Evaluation
```

---

# 1. Machine Learning Workflow

A typical machine learning workflow consists of the following steps.

## Step 1: Prepare the Training Data

The model learns from data. The data may be images, words, numerical tables, clicks, or other formats.

Before learning, raw data must usually be processed into a numerical representation.

Examples:

- Image → pixels → feature vector
- Text → word counts / embeddings
- User behavior → numerical features
- Medical record → structured variables

## Step 2: Select a Learning Algorithm

Different tasks require different algorithms.

| Task | Example Algorithms |
|---|---|
| Classification | KNN, Logistic Regression |
| Regression | Linear Regression |
| Clustering | K-means |
| Model Selection | Validation, Cross-validation |

## Step 3: Train the Model

Training means using data to improve model parameters.

Usually this involves optimization:

$$
\min_{\theta} Loss(\theta)
$$

or equivalently, in probabilistic models:

$$
\max_{\theta} Likelihood(\theta)
$$ 

## Step 4: Evaluate and Improve

A model should not only perform well on training data. It should also generalize to unseen data.

Evaluation is usually done using:

- validation set
- test set
- cross-validation
- prediction error
- accuracy

---

# 2. Supervised Learning

## 2.1 Definition

Supervised learning uses labeled data.

The training data have the form:

$$
(x_i,y_i),\quad i=1,\dots,N
$$

where:

- $x_i$ is the input or feature vector
- $y_i$ is the label or target

The goal is to learn a function:

$$
h:X\rightarrow Y
$$

such that for a new input $x$, the prediction $h(x)$ is close to the correct label $y$.

---

## 2.2 Classification

Classification is a supervised learning task where the output is a discrete class label.

Examples:

| Input | Output |
|---|---|
| Email | Spam / Not Spam |
| Image | Cat / Dog |
| Patient data | Disease A / Disease B / Healthy |
| Customer profile | Will buy / Will not buy |

In binary classification:

$$
y\in\{0,1\}
$$

A classifier is a function:

$$
h:X\rightarrow\{0,1\}
$$

---


# 3. K-Nearest Neighbors (KNN)

## 3.1 Main Idea

K-Nearest Neighbors is a simple supervised learning method.

To classify a new point:

1. Compute its distance to all training samples.
2. Find the $K$ nearest training samples.
3. Use majority voting among those $K$ neighbors.
4. Assign the new point to the majority class.

The intuition is:

> Similar points should have similar labels.

---

## 3.2 KNN for Classification

Suppose the training data are:

$$
(x_1,y_1),\dots,(x_N,y_N)
$$

For a new point $x$, find the $K$ points closest to$x$.

Then predict:

$$
h(x)=\text{majority label among the K nearest neighbors}
$$

Example:

If $K=5$, and among the five nearest neighbors:

- 3 are cats
- 2 are dogs

then KNN predicts cat.

---

## 3.3 Distance Functions

KNN depends heavily on the choice of distance function.

Given:

$$
x=(x_1,\dots,x_n)^T
$$

$$
y=(y_1,\dots,y_n)^T
$$

we need a function \(d(x,y)\) measuring how different \(x\) and \(y\) are.

---

## 3.4 Euclidean Distance

The Euclidean distance is:

$$
d(x,y)=\sqrt{\sum_{i=1}^n(x_i-y_i)^2}
$$

It is the ordinary straight-line distance.

For example, if the coordinate differences are 3 and 4, the Euclidean distance is:

$$
\sqrt{3^2+4^2}=5
$$

---

## 3.5 Manhattan Distance

The Manhattan distance is:

$$
d(x,y)=\sum_{i=1}^n |x_i-y_i|
$$

It measures distance as if movement must occur along grid-like city blocks.

For coordinate differences 3 and 4:

$$
3+4=7
$$

---

## 3.6 Minkowski Distance

The Minkowski distance is a general family:

$$
d(x,y)=\left(\sum_{i=1}^n|x_i-y_i|^p\right)^{1/p}
$$

Special cases:

| $p$ | Distance |
|---|---|
| $p=1$ | Manhattan distance |
| $p=2$ | Euclidean distance |
| $p=\infty$ | Maximum coordinate difference |

For $p=\infty$:

$$
d(x,y)=\max_i |x_i-y_i|
$$

---

## 3.7 Choosing K in KNN

The value of $K$ controls model complexity.

### Small K

If $K=1$, the model uses only the  nearest neighbor.

This can lead to overfitting.

Characteristics:

- Decision boundary is very flexible
- Training performance may be very good
- Sensitive to noise
- Test performance may be poor

### Large K

If $K$ is very large, the model averages over many points.

This can lead to underfitting.

Characteristics:

- Decision boundary is too smooth
- Local patterns may be ignored
- Both training and test performance may be poor

### Moderate K

A good $K$ balances flexibility and stability.

Model selection methods such as validation sets and cross-validation are used to choose $K$.

---

# 4. Logistic Regression

## 4.1 Main Idea

Logistic regression is a supervised learning method for classification.

Instead of directly outputting a class label, logistic regression outputs a probability.

For binary classification:

$$
y\in\{0,1\}
$$

Logistic regression models:

$$
P(y=1\mid x)
$$

as a function of the input features.

---

## 4.2 Why Not Use Linear Regression Directly?

A linear model:

$$
\theta^Tx+b
$$

can output any real number:

$$
-\infty < \theta^Tx+b < +\infty
$$

But a probability must lie between 0 and 1.

Therefore, logistic regression passes the linear score through a sigmoid function.

---

## 4.3 Sigmoid Function

The sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Properties:

- It maps any real number to $(0,1)$
- It is increasing
- If $z$ is very large, $\sigma(z)\approx1$
- If $z$ is very negative, $\sigma(z)\approx0$
- If $z=0$, $\sigma(z)=0.5$

---

## 4.4 Logistic Regression Model

Let

$$
z=\theta^Tx+b
$$ 

Then:

$$
P(y=1\mid x,\theta,b)=\sigma(\theta^Tx+b)
$$

That is:

$$
P(y=1\mid x,\theta,b)=
\frac{1}{1+\exp[-(\theta^Tx+b)]}
$$

And:

$$
P(y=0\mid x,\theta,b)=1-P(y=1\mid x,\theta,b)
$$

Equivalently:

$$
P(y=0\mid x,\theta,b)=
\frac{\exp[-(\theta^Tx+b)]}{1+\exp[-(\theta^Tx+b)]}
$$

---

## 4.5 Classification Rule

After estimating $\theta$ and $b$, we can classify a new point using a threshold.

Usually:

$$
\hat y=
\begin{cases}
1, & P(y=1\mid x)\ge0.5\\
0, & P(y=1\mid x)<0.5
\end{cases}
$$

Since $\sigma(z)\ge0.5$ when $z\ge0$, this is equivalent to:

$$
\theta^Tx+b\ge0
$$

So logistic regression has a linear decision boundary.

---

## 4.6 Training Logistic Regression

Given labeled samples:

$$
(x_i,y_i),\quad i=1,\dots,m
$$

we estimate $\theta$ and $b$ by maximum likelihood estimation.

The likelihood is:

$$
L(\theta,b)=\prod_{i=1}^m P(y_i\mid x_i,\theta,b)
$$

The log-likelihood is:

$$
\ell(\theta,b)=\sum_{i=1}^m \log P(y_i\mid x_i,\theta,b)
$$

Training logistic regression means maximizing this quantity, or equivalently minimizing negative log-likelihood.

---

## 4.7 Concavity and Global Optimum

The lecture emphasizes that the logistic regression log-likelihood is concave in $\theta$ and $b$.

This is important because maximizing a concave function is a convex optimization problem.

Therefore, logistic regression training has a single global optimum in the convex optimization sense.

---

# 5. Unsupervised Learning

In unsupervised learning, labels are not available.

Data:

$$
x_i
$$

The corresponding label $y_i$ is unseen or does not exist.

Goal:

Discover patterns or structure in the data.

Examples:

- customer segmentation
- topic discovery
- outlier detection
- recommendation
- clustering images or documents

---

# 6. Clustering
  
## 6.1 What Is Clustering?

Clustering is an unsupervised learning task.

The goal is to divide objects into groups based on similarity or dissimilarity.

A good clustering result should satisfy:

- points within the same cluster are similar
- points across different clusters are not very similar

There is no universal standard for clustering because “similarity” depends on the problem.

---

## 6.2 Similarity and Dissimilarity

A clustering algorithm needs a similarity or dissimilarity function.

A dissimilarity function is often called a distance function.

The choice of distance strongly influences the clustering result.

---

## 6.3 Desired Properties of a Distance Function

### Symmetry

$$
d(x,y)=d(y,x)
$$

### Positive Separability

$$
d(x,y)=0 \iff x=y
$$

### Triangle Inequality

$$
d(x,y)\le d(x,z)+d(z,y)
$$

---

# 7. K-Means Clustering

## 7.1 Main Idea

K-means clustering divides data into $K$clusters.

Given data points:

$$
x_1,x_2,\dots,x_m
$$

we want to find cluster centers:

$$
c_1,c_2,\dots,c_K
$$  

and assign each data point to one cluster:

$$
\pi(i)\in\{1,\dots,K\}
$$

---

## 7.2 K-Means Objective

The goal is to minimize the total within-cluster squared distance:

$$
\min_{c,\pi}
\sum_{i=1}^m \|x_i-c_{\pi(i)}\|^2
$$

where:

- $c_j$ is the center of cluster $j$
- $\pi(i)$ is the cluster assignment of point $i $

The objective says:

> Each point should be close to its assigned cluster center.

---

## 7.3 K-Means Algorithm

### Step 1: Initialize Centers

Choose $K$ initial centers.

### Step 2: Assignment Step

Assign each point to the nearest center:

$$
\pi(i)=\arg\min_j \|x_i-c_j\|
$$

### Step 3: Update Step

Update each center as the mean of the points assigned to it:

$$
c_j=\frac{1}{|C_j|}\sum_{x_i\in C_j}x_i
$$

### Step 4: Repeat

Repeat assignment and update until the cluster assignments or centers stop changing significantly.

---

## 7.4 Important Notes About K-Means

K-means is simple and widely used, but it has limitations.

- The number of clusters $K$ must be chosen before training.
- Different initial centers may lead to different final clusters.
- K-means works best when clusters are roughly compact and spherical.
- Feature scaling matters because K-means depends on distances.

---

# 8. Model Selection

## 8.1 What Is Model Selection?

Model selection asks:

> Which model or hyperparameter should we choose?

Examples:

1. Should we use KNN or logistic regression?
2. What value of $K$ should be used in KNN?
3. How many clusters should be used in K-means?
4. What polynomial degree should be used in regression?

---

## 8.2 Why Model Selection Is Necessary

A model that is too simple may underfit.

A model that is too complex may overfit.

The goal is to find a model that generalizes well to new, unseen data.

---

## 8.3 Overfitting and Underfitting

### 8.3.1 Overfitting

A model overfits when it learns patterns specific to the training data, including noise.

Characteristics:

- Training error is low
- Test error is high
- Model is too complex
- Decision boundary may be too flexible

Example:

In KNN, $K=1$ often overfits because the prediction is determined by a single nearest point.

---

### 8.3.2 Underfitting

A model underfits when it fails to capture the important patterns in the data.

Characteristics:

- Training error is high
- Test error is high
- Model is too simple
- Decision boundary may be too rigid

Example:

In KNN, a very large $K$, such as $K=101$, may underfit because it averages over too many points.

---

### 8.3.3 Appropriate Fitting

A good model captures the main structure of the data without fitting noise.

This is the goal of model selection.

---

# 9. Bias-Variance Tradeoff

Although the lectures emphasize overfitting and underfitting, the underlying idea is the bias-variance tradeoff.

## 9.1 Bias

Bias measures how much the model’s assumptions deviate from the true pattern.

High bias means the model is too simple.

High bias is associated with underfitting.

## 9.2 Variance

Variance measures how sensitive the model is to changes in the training data.

High variance means the model changes a lot when the data change.

High variance is associated with overfitting.

## 9.3 Tradeoff

A very simple model has high bias and low variance.

A very complex model has low bias and high variance.

The goal is to balance bias and variance.

---

# 10. Validation Set Approach

## 10.1 Main Idea

The validation set approach splits data into two parts:

```text
Training data
Validation / Test data
```

For each candidate model:

1. Train the model on the training data.
2. Evaluate prediction error on the validation/test data.
3. Choose the model with the smallest validation/test error.

If candidate models are indexed by $m=1,\dots,M$, choose:

$$
m^*=\arg\min_m Err_m
$$

---

## 10.2 Example: Choosing K in KNN

Candidate models:

$$
K=1,2,\dots,M
$$

For each $K$:

1. Train KNN using the training data.
2. Compute validation error.
3. Choose the $K$ with the smallest validation error.

---

## 10.3 Limitation

The selected model may depend heavily on how the data are split.

A different train/test split may produce a different best model.

This motivates cross-validation.

---

# 11. K-Fold Cross-Validation

## 11.1 Main Idea

K-fold cross-validation reduces dependence on a single data split.

Procedure:

1. Split the data into $K$ folds.
2. For each fold:
   - use that fold as the test/validation set
   - use the remaining folds as training data
   - train the model and compute error
3. Average the errors across folds.

The cross-validation error is:

$$
CV=\frac{1}{K}\sum_{i=1}^K Err_i
$$

Choose the model with the smallest average validation error.

---

## 11.2 5-Fold Cross-Validation

In 5-fold cross-validation:

- Split data into 5 parts.
- Train 5 times.
- Each time, use one fold for testing and the other four for training.
- Average the five test errors.

This gives a more stable estimate of model performance than a single validation split.

---

# 12. Formula Sheet

## Sigmoid Function

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

## Logistic Regression

$$
P(y=1\mid x,\theta,b)=
\frac{1}{1+\exp[-(\theta^Tx+b)]}
$$

$$
P(y=0\mid x,\theta,b)=1-P(y=1\mid x,\theta,b)
$$

## Logistic Regression Log-Likelihood

$$
\ell(\theta,b)=
\sum_{i=1}^m \log P(y_i\mid x_i,\theta,b)
$$

## Euclidean Distance

$$
d(x,y)=\sqrt{\sum_i(x_i-y_i)^2}
$$

## Manhattan Distance

$$
d(x,y)=\sum_i|x_i-y_i|
$$

## Minkowski Distance

$$
d(x,y)=
\left(\sum_i|x_i-y_i|^p\right)^{1/p}
$$

## Infinity Distance

$$
d(x,y)=\max_i|x_i-y_i|
$$

## K-Means Objective

$$
\min_{c,\pi}
\sum_{i=1}^m \|x_i-c_{\pi(i)}\|^2
$$

## K-Fold Cross-Validation Error

$$
CV=\frac{1}{K}\sum_{i=1}^K Err_i
$$

---

# 13. Final Remarks

The Machine Learning part of DDA2001 connects all previous topics.

- Probability provides class probabilities and uncertainty modeling.
- Statistics provides evaluation and generalization ideas.
- Optimization provides the training mechanism.

The main learning methods covered are:

1. **KNN** — classify by nearest labeled examples.
2. **Logistic Regression** — model class probability using a sigmoid function.
3. **K-means** — cluster unlabeled data by minimizing within-cluster distance.
4. **Model Selection** — choose models and hyperparameters using validation and cross-validation.

The most important conceptual goal is not memorizing algorithms, but understanding the complete ML workflow:

```text
Data
  ↓
Features
  ↓
Model
  ↓
Training
  ↓
Evaluation
  ↓
Model Selection
  ↓
Generalization
```
