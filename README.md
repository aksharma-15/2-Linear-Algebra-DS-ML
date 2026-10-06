# 2-Linear-Algebra-DS-ML
Linear algebra foundations for data science and machine learning

# 2-Linear-Algebra-DS-ML
Linear algebra foundations for data science and machine learning

# Linear Algebra Foundations for Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=for-the-badge\&logo=matplotlib\&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge\&logo=jupyter\&logoColor=white)

> A practical, intuition-first exploration of the linear algebra operations that form the foundation of machine learning and deep learning.

---

## Overview

This notebook develops a practical understanding of **linear algebra through NumPy**, with a focus on the operations that repeatedly appear in machine learning systems.

The notebook moves from basic numerical representations to a small recommendation-system example:

```text
Scalars
   ↓
Vectors
   ↓
Matrices
   ↓
Vector operations
   ↓
Dot product
   ↓
Norms and distance
   ↓
Cosine similarity
   ↓
Unit vectors
   ↓
Matrix multiplication
   ↓
Batch predictions
   ↓
User similarity
   ↓
Recommendation
```

The goal is not simply to memorize mathematical formulas.

The goal is to understand **what the operations mean geometrically, how they are implemented computationally, and where they appear in machine learning**.

---

## Notebook Contents

* Scalars, vectors, matrices, and tensors
* Representing tabular data as vectors and matrices
* Vector addition
* Vector subtraction
* Scalar multiplication
* Shape compatibility
* Dot products
* Dot products as weighted sums
* Dot products as directional alignment
* Vector norms
* Euclidean distance
* Cosine similarity
* Unit vectors and normalization
* Matrix multiplication
* Element-wise multiplication vs matrix multiplication
* Matrix transpose
* Pairwise relationships using `A @ A.T`
* Batch prediction using matrix multiplication
* A simple user-based recommendation system

---

## Libraries and Tools

### NumPy

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square\&logo=numpy\&logoColor=white)

Used throughout the notebook for:

* arrays
* vectors
* matrices
* dot products
* matrix multiplication
* norms
* normalization
* random matrix generation
* vectorized computation

```python
import numpy as np
```

### Matplotlib

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square\&logo=matplotlib\&logoColor=white)

Used to visualize:

* vectors
* points in feature space
* distances
* directional relationships
* normalized vectors

```python
import matplotlib.pyplot as plt
```

### Jupyter / Google Colab

The notebook uses the standard Python 3 kernel and contains visual and interactive experiments suitable for Jupyter Notebook or Google Colab.

---

# 1. Scalars, Vectors, Matrices, and Tensors

The notebook introduces the basic hierarchy of numerical structures used in machine learning.

| Structure | Dimensions | Example                     |
| --------- | ---------: | --------------------------- |
| Scalar    |        0-D | `35`                        |
| Vector    |        1-D | `[35, 55, 12]`              |
| Matrix    |        2-D | `[[34, 55, 12], ...]`       |
| Tensor    |        N-D | General N-dimensional array |

A useful mental model is:

```text
Scalar → Vector → Matrix → Tensor
  0-D      1-D       2-D       N-D
```

---

## Scalar

A scalar is a single numerical value.

```python
cust_age = 35
```

In machine learning, scalars can represent individual measurements, scores, losses, biases, probabilities, or hyperparameters.

---

## Vector

A vector is a one-dimensional numerical representation.

```python
customer = np.array([35, 55, 12])
```

For example, the values could represent:

```text
[age, income, purchases]
```

A vector can be interpreted as a point in a feature space.

---

## Matrix

A matrix is a two-dimensional collection of values.

```python
customers = np.array([
    [34, 55, 12],
    [28, 42,  5],
    [45, 68, 20],
])
```

The shape is:

```text
(3, 3)
```

which can be interpreted as:

```text
3 rows    → 3 customers
3 columns → 3 features
```

This is one of the most important conventions in practical machine learning:

```text
rows    → samples / observations
columns → features
```

---

## Tensor

A tensor generalizes these ideas to arbitrary dimensions.

```text
0-D → scalar
1-D → vector
2-D → matrix
N-D → tensor
```

Deep-learning frameworks such as PyTorch and TensorFlow use tensors as their primary data structure.

---

# 2. Data as Points in Feature Space

One of the most important ideas demonstrated in the notebook is that numerical observations can be interpreted geometrically.

Consider:

```python
customer = np.array([35, 55, 12])
```

This vector can be interpreted as a point:

```text
(age, income, purchases)
```

If we visualize two of those features:

```python
draw_points(
    x=customers[:, 0],
    y=customers[:, 2],
    names=["A", "B", "C"],
    xlabel="age",
    ylabel="purchases",
    title="Each customer is a POINT"
)
```

we can think of every customer as a point in a feature space.

This geometric interpretation becomes extremely useful when we later discuss:

* distance
* direction
* similarity
* normalization
* nearest neighbors
* recommendation systems

---

# 3. Feature Vectors in Machine Learning

The notebook also introduces an applicant-loan representation:

```text
[age, income, loan, tenure, coapplicant, education, job]
```

Each applicant can therefore be represented as a feature vector.

For example:

```text
Applicant 1
→ [age, income, loan, tenure, coapplicant, education, job]

Applicant 2
→ [age, income, loan, tenure, coapplicant, education, job]
```

This is the fundamental idea behind tabular machine learning:

```text
One observation
      ↓
Feature vector
      ↓
Model
      ↓
Prediction
```

---

# 4. Vector Addition, Subtraction, and Scaling

The notebook demonstrates three basic vector operations.

```python
last_month = np.array([2, 5, 1])
this_month = np.array([3, 1, 4])
```

## Vector Addition

```python
last_month + this_month
```

Addition happens element by element.

```text
[2, 5, 1]
+
[3, 1, 4]
-----------
[5, 6, 5]
```

This can represent combining corresponding quantities across two vectors.

---

## Vector Subtraction

```python
this_month - last_month
```

Subtraction also happens element by element.

```text
[3, 1, 4]
-
[2, 5, 1]
-----------
[1, -4, 3]
```

Geometrically, subtracting one vector from another can describe a displacement or change.

---

## Scalar Multiplication

```python
2 * this_month
```

produces:

```text
2 × [3, 1, 4]
=
[6, 2, 8]
```

Scalar multiplication changes the magnitude of a vector.

---

# 5. Shape Compatibility

The notebook intentionally includes an example of incompatible vector shapes:

```python
a = np.array([3, 5, 1])
b = np.array([1, 4])
```

Trying:

```python
a + b
```

does not represent a valid element-wise vector addition because the vectors have different lengths.

This introduces an important engineering principle:

> **Tensor shape is part of the meaning of a computation.**

When debugging numerical ML code, always inspect:

```python
x.shape
```

before assuming an operation is valid.

This becomes even more important when working with:

* NumPy
* PyTorch
* TensorFlow
* JAX
* batched data
* broadcasting
* attention tensors

---

# 6. Dot Product

The dot product is one of the most important operations in machine learning.

For two vectors:

```text
a = [a₁, a₂, ..., aₙ]
b = [b₁, b₂, ..., bₙ]
```

the dot product is:

```text
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

The notebook demonstrates three equivalent NumPy approaches:

```python
np.dot(a, b)
```

```python
a.dot(b)
```

```python
a @ b
```

For one-dimensional vectors, these produce the same dot-product result.

---

# 7. Dot Product as a Weighted Sum

The first important ML interpretation of the dot product is:

> A dot product is a weighted sum.

Suppose:

```python
features = np.array([35, 55, 12])
weights = np.array([0.1, 0.5, 2.0])
```

Then:

```python
features @ weights
```

computes:

```text
35 × 0.1
+
55 × 0.5
+
12 × 2.0
```

The result is:

```text
54.9
```

Conceptually:

```text
features → x
weights  → w

prediction score → xᵀw
```

A linear model can be expressed as:

```text
ŷ = xᵀw + b
```

where:

```text
x → input features
w → learned weights
b → bias / intercept
ŷ → predicted output
```

The notebook demonstrates the `xᵀw` portion directly.

---

# 8. Dot Product in a Spam-Scoring Example

The notebook applies the same idea to a simple spam classifier.

```python
email = np.array([1, 5, 0, 3])

weights = np.array([
    1.2,
   -0.9,
    1.8,
   -1.1
])

score = email @ weights
```

The features represent:

```text
free
meeting
winner
report
```

The model weights determine how strongly each feature contributes to the score.

Conceptually:

```text
feature value × learned weight
                    ↓
              contribution
                    ↓
                 sum
                    ↓
               final score
```

The notebook then uses:

```python
"SPAM" if score > 0 else "not spam"
```

to turn the score into a simple binary decision.

### Terminology note

The notebook calls this value a **spam score**.

In a production machine-learning system, the precise terminology would depend on the model. It could be described as a:

* decision score
* raw model score
* logit

It should not automatically be called a probability.

---

# 9. Dot Product as Directional Alignment

The dot product has a second important interpretation.

The relationship is:

```text
a · b = ‖a‖ ‖b‖ cos(θ)
```

where:

```text
‖a‖ → magnitude of a
‖b‖ → magnitude of b
θ   → angle between a and b
```

This means that the dot product depends on:

1. the magnitude of the first vector,
2. the magnitude of the second vector,
3. the angle between them.

---

## Interpreting the Sign

The angle between two vectors determines the sign of their dot product.

```text
Angle < 90°  → positive dot product
Angle = 90°  → zero dot product
Angle > 90°  → negative dot product
```

Therefore:

```text
positive → generally similar direction
zero     → perpendicular / orthogonal
negative → generally opposing direction
```

The notebook demonstrates this using preference vectors:

```text
[electronics spend, groceries spend]
```

For example:

```python
tech = np.array([4, 1])
groceries = np.array([-2, 4])
```

Their dot product is negative, indicating that the vectors point in substantially different directions.

---

# 10. Norm of a Vector

A **norm** measures the magnitude or length of a vector.

The notebook uses:

```python
np.linalg.norm(v)
```

For:

```python
v = np.array([3, 4])
```

the result is:

```text
5
```

because:

```text
‖v‖₂ = √(3² + 4²)
     = √25
     = 5
```

The subscript `₂` indicates the **L2 norm**, also called the **Euclidean norm**.

For a general vector:

```text
x = [x₁, x₂, ..., xₙ]
```

the L2 norm is:

```text
‖x‖₂ = √(x₁² + x₂² + ... + xₙ²)
```

---

# 11. Distance Between Vectors

The notebook computes:

```python
c = b - a
np.linalg.norm(b - a)
```

For:

```python
a = np.array([1, 3])
b = np.array([4, 1])
```

the displacement is:

```text
b - a
=
[3, -2]
```

The distance is then:

```text
‖b - a‖₂
```

or:

```text
√(3² + (-2)²)
```

This is the **Euclidean distance** between the two points.

Geometrically:

```text
a ─────────────────→ b
       distance
```

The notebook's visualization makes this relationship explicit by drawing the arrow between the two points.

---

# 12. Norm vs Distance

The notebook makes an important observation:

> The norm of the difference between two vectors gives their Euclidean distance.

In other words:

```text
distance(a, b)
=
‖a - b‖₂
```

This distinction is worth remembering:

```text
‖a‖₂
```

measures the distance of `a` from the origin.

While:

```text
‖a - b‖₂
```

measures the distance between `a` and `b`.

This distinction appears frequently in machine learning.

---

# 13. Cosine Similarity

Cosine similarity measures the similarity between two vectors based primarily on their **direction**.

Starting from:

```text
a · b = ‖a‖ ‖b‖ cos(θ)
```

we can rearrange it:

```text
cos(θ) = (a · b) / (‖a‖ ‖b‖)
```

The notebook implements this directly:

```python
def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
```

The standard range is:

```text
[-1, +1]
```

Interpretation:

| Value | Meaning               |
| ----: | --------------------- |
|  `+1` | Same direction        |
|   `0` | Orthogonal directions |
|  `-1` | Opposite direction    |

---

# 14. Why Cosine Similarity Ignores Magnitude

The notebook demonstrates:

```python
short_tech = np.array([3, 1])
long_tech  = np.array([30, 10])
```

Notice:

```text
long_tech = 10 × short_tech
```

The vectors have very different magnitudes.

However, they point in exactly the same direction.

Therefore:

```text
cosine(short_tech, long_tech) = 1
```

This is the central intuition behind cosine similarity:

> **Cosine similarity compares direction rather than absolute magnitude.**

This is particularly useful for representations such as:

* text vectors
* embeddings
* user preference vectors
* document representations
* recommendation vectors

---

# 15. Cosine Similarity and Zero Vectors

There is an important mathematical edge case.

Cosine similarity contains:

```text
‖a‖ ‖b‖
```

in the denominator.

If either vector is a zero vector:

```text
‖a‖ = 0
```

then the expression involves division by zero.

Therefore, a production implementation should explicitly handle zero vectors.

The notebook does not add this production-level safeguard because the examples use non-zero vectors.

---

# 16. Unit Vectors

A **unit vector** is a vector whose magnitude is exactly `1`.

The notebook defines:

```python
def unit(vector):
    return vector / np.linalg.norm(vector)
```

For:

```python
v = np.array([3, 4])
```

we have:

```text
‖v‖₂ = 5
```

Therefore:

```text
v̂ = v / ‖v‖₂
```

becomes:

```text
v̂ = [3/5, 4/5]
```

or approximately:

```text
[0.6, 0.8]
```

The resulting vector has:

```text
‖v̂‖₂ = 1
```

---

# 17. Why Normalize a Vector?

Normalization preserves direction while changing magnitude.

Conceptually:

```text
Original vector
      ↓
divide by its magnitude
      ↓
Unit vector
      ↓
same direction
      ↓
magnitude = 1
```

This is particularly important for cosine similarity.

If:

```text
‖a‖ = 1
‖b‖ = 1
```

then:

```text
a · b = cos(θ)
```

Therefore, after L2 normalization, a dot product directly represents cosine similarity.

This relationship becomes extremely important in embedding-based systems.

---

# 18. Matrix Multiplication

The notebook introduces matrix multiplication using:

```python
A = np.random.randint(1, 10, size=(2, 4))
B = np.random.randint(1, 10, (4, 5))
```

Their shapes are:

```text
A → (2, 4)
B → (4, 5)
```

Matrix multiplication is valid because the **inner dimensions match**:

```text
(2 × 4) @ (4 × 5)
      ↑       ↑
      └─ match ─┘
```

The resulting matrix has shape:

```text
(2, 5)
```

The general rule is:

```text
(m × n) @ (n × p) → (m × p)
```

The number of columns in the left matrix must equal the number of rows in the right matrix.

---

# 19. Element-wise Multiplication vs Matrix Multiplication

One of the notebook's useful engineering comparisons is:

```python
A * B
```

versus:

```python
A @ B
```

These are not interchangeable.

## Element-wise multiplication

```python
A * B
```

performs element-by-element multiplication when the shapes are compatible.

This is also called the **Hadamard product**.

Conceptually:

```text
[A₁ A₂]    [B₁ B₂]
   ↓  ×       ↓
[A₁B₁ A₂B₂]
```

---

## Matrix multiplication

```python
A @ B
```

performs the algebraic matrix product.

It combines rows of the first matrix with columns of the second matrix through dot products.

The distinction is fundamental in ML engineering:

```text
*  → element-wise multiplication
@  → matrix multiplication
```

---

# 20. Matrix Transpose

The transpose changes rows into columns and columns into rows.

The notebook uses:

```python
A.T
```

If:

```text
A.shape = (2, 4)
```

then:

```text
A.T.shape = (4, 2)
```

Conceptually:

```text
A      →      Aᵀ

rows          columns
↓             ↓
columns       rows
```

Transpose operations appear constantly in linear algebra and ML because they allow tensors to be oriented correctly for mathematical operations.

---

# 21. A @ Aᵀ

The notebook computes:

```python
A @ A.T
```

If:

```text
A.shape = (2, 4)
```

then:

```text
A.T.shape = (4, 2)
```

and therefore:

```text
(2 × 4) @ (4 × 2)
```

produces:

```text
(2 × 2)
```

Each entry represents a dot product between two rows of `A`.

Conceptually:

```text
A @ Aᵀ
     ↓
pairwise dot products between rows
```

This is a very useful pattern in machine learning.

When the rows of `A` have been normalized to unit length:

```text
A_unit @ A_unitᵀ
```

becomes a matrix of pairwise cosine similarities.

This exact idea is used later in the recommendation-system example.

---

# 22. Matrix Multiplication as Batch Prediction

One of the most important ML connections in the notebook is:

```python
customers @ weights
```

The notebook defines:

```python
customers = np.array([
    [34, 55, 12],
    [28, 42,  5],
    [45, 68, 20],
    [39, 50,  9],
])

weights = np.array([0.1, 0.5, 2.0])
```

The shapes are:

```text
customers → (4, 3)
weights   → (3,)
```

Therefore:

```text
(4 × 3) @ (3,) → (4,)
```

The result is:

```text
[54.9, 33.8, 78.5, 46.9]
```

The important idea is that one weight vector is applied to every customer.

Instead of writing:

```python
customer_1 @ weights
customer_2 @ weights
customer_3 @ weights
customer_4 @ weights
```

we can write:

```python
customers @ weights
```

This is **vectorization**.

---

# 23. From One Prediction to a Batch

A single linear prediction looks like:

```text
ŷ = xᵀw
```

For multiple samples:

```text
Ŷ = Xw
```

where:

```text
X  → feature matrix
w  → weight vector
Ŷ → predictions
```

This transition is fundamental to machine learning.

The same mathematical pattern appears in:

* linear regression
* logistic regression
* dense neural-network layers
* recommendation systems
* embedding operations
* attention mechanisms
* similarity search

---

# 24. Recommendation System

The final section of the notebook brings several of the previous concepts together.

The example contains:

```text
Users
Movies
Ratings
```

The users are:

```python
users = ["Aisha", "Bilal", "Chen", "Divya", "Erik"]
```

The movies are:

```python
movies = [
    "Fight Club",
    "KGF",
    "chand mera dil",
    "Musafir cafe",
    "Jiro dreams of shushi'"
]
```

The rating matrix is:

```python
R = np.array([
    [5, 0, 1, 0, 2],  # Aisha
    [4, 5, 0, 1, 1],  # Bilal
    [0, 1, 5, 4, 2],  # Chen
    [1, 0, 4, 5, 3],  # Divya
    [1, 1, 1, 1, 5],  # Erik
])
```

Its shape is:

```text
(5, 5)
```

which represents:

```text
5 users × 5 movies
```

The rows represent users.

The columns represent movies.

Each row can therefore be treated as a user-preference vector.

---

# 25. User Vectors

For example, Aisha's vector is:

```text
[5, 0, 1, 0, 2]
```

Bilal's vector is:

```text
[4, 5, 0, 1, 1]
```

Chen's vector is:

```text
[0, 1, 5, 4, 2]
```

and so on.

We can therefore reinterpret the recommendation problem as:

```text
User
 ↓
Preference vector
 ↓
Compare vectors
 ↓
Find similar users
 ↓
Use similar user's preferences
 ↓
Recommend an unseen movie
```

This is the core linear-algebra idea behind the example.

---

# 26. Normalizing the User Vectors

The notebook converts every user vector into a unit vector:

```python
R_unit = np.array([unit(user) for user in R])
```

Now every row has approximately:

```text
‖user_vector‖₂ = 1
```

This removes differences in overall magnitude and focuses the comparison on direction.

That makes the next matrix multiplication especially useful.

---

# 27. Computing Pairwise User Similarity

The notebook computes:

```python
S = R_unit @ R_unit.T
```

Because the rows have been normalized:

```text
dot product
     ↓
cosine similarity
```

Therefore:

```text
S[i, j]
```

represents the cosine similarity between user `i` and user `j`.

The resulting matrix is:

```text
          Aisha  Bilal  Chen  Divya  Erik
Aisha      1.00   0.61  0.24  0.38  0.54
Bilal      0.61   1.00  0.25  0.26  0.42
Chen       0.24   0.25  1.00  0.95  0.55
Divya      0.38   0.26  0.95  1.00  0.65
Erik       0.54   0.42  0.55  0.65  1.00
```

---

# 28. Reading the Similarity Matrix

Several properties are immediately visible.

## Diagonal

The diagonal is approximately:

```text
1.00
```

because every user is perfectly similar to themselves.

```text
Aisha ↔ Aisha = 1
Bilal ↔ Bilal = 1
...
```

---

## Symmetry

The matrix is symmetric:

```text
S[i, j] = S[j, i]
```

For example:

```text
Chen ↔ Divya = 0.95
Divya ↔ Chen = 0.95
```

This is expected because cosine similarity is symmetric.

---

## Strongest similarity in the example

Chen and Divya have a similarity of approximately:

```text
0.95
```

which means their preference vectors point in very similar directions.

---

# 29. Finding Aisha's Closest User

The notebook extracts Aisha's similarity scores:

```python
aisha = S[0].copy()
```

Since Aisha is perfectly similar to herself:

```text
Aisha ↔ Aisha = 1.00
```

the notebook prevents self-selection:

```python
aisha[0] = 0
```

Then:

```python
match = np.argmax(aisha)
```

selects the user with the largest remaining similarity score.

The result is:

```text
closest user to Aisha: Bilal
```

because:

```text
Aisha ↔ Bilal = 0.61
```

which is Aisha's highest similarity score among the other users.

---

# 30. Recommendation Through the Nearest User

The notebook then checks:

```python
R[match, 1]
```

and recommends:

```text
KGF
```

The simplified workflow is:

```text
Aisha
  ↓
Aisha's preference vector
  ↓
normalize vector
  ↓
compare with every user
  ↓
cosine similarity
  ↓
find most similar user
  ↓
Bilal
  ↓
inspect Bilal's movie preferences
  ↓
recommend KGF
```

This is a simple example of **user-based collaborative filtering**.

---

# 31. Important Recommendation-System Caveat

The notebook explicitly uses:

```text
0 = movie has not been watched yet
```

This is convenient for demonstrating the mathematics, but it is an important simplification.

In a real recommendation system:

```text
0
```

could mean many different things depending on the data model.

For example:

```text
0 rating
```

is not necessarily equivalent to:

```text
missing rating
```

or:

```text
user has never interacted with the item
```

A production recommender should distinguish between:

* missing values
* explicit ratings
* implicit interactions
* negative feedback
* actual zero-valued measurements

Therefore, this notebook should be viewed as a **linear-algebra demonstration of similarity-based recommendation**, rather than a production recommender implementation.

---

# 32. A Useful Mathematical Pattern

One of the most valuable patterns in the notebook is:

```python
R_unit @ R_unit.T
```

This combines several concepts introduced earlier.

### Step 1 — Represent users as vectors

```text
User → preference vector
```

### Step 2 — Normalize the vectors

```text
preference vector
        ↓
unit vector
```

### Step 3 — Compute dot products

```text
R_unit @ R_unit.T
```

### Step 4 — Interpret those dot products

Because the vectors are normalized:

```text
dot product = cosine similarity
```

### Step 5 — Obtain pairwise similarities

```text
user × user similarity matrix
```

This is an excellent example of how seemingly simple linear-algebra operations can compose into a useful ML workflow.

---

# 33. Machine Learning Connections

The notebook's operations appear repeatedly in modern machine learning.

## Linear Models

A linear model can be represented as:

```text
ŷ = xᵀw + b
```

The notebook directly demonstrates the `xᵀw` component.

---

## Neural Networks

A dense neural-network layer can be represented conceptually as:

```text
Z = XW + b
```

followed by an activation function:

```text
A = f(Z)
```

The matrix multiplication concepts demonstrated in this notebook therefore provide part of the mathematical foundation for neural networks.

---

## Embeddings

Many modern ML systems represent objects as vectors:

```text
words
sentences
documents
users
movies
products
images
```

These vector representations are commonly called **embeddings**.

Once objects are represented as vectors, we can compare them using:

```text
dot product
cosine similarity
Euclidean distance
```

---

## Retrieval Systems

Modern vector-search systems follow a conceptual pipeline such as:

```text
Object
  ↓
Embedding model
  ↓
Vector representation
  ↓
Similarity search
  ↓
Nearest / most relevant vectors
```

The normalization and similarity concepts demonstrated in this notebook are directly relevant to this type of system.

---

# 34. Key Mathematical Relationships

The following formulas summarize the core mathematics demonstrated in the notebook.

## Dot Product

```text
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

---

## Geometric Interpretation

```text
a · b = ‖a‖ ‖b‖ cos(θ)
```

---

## L2 Norm

```text
‖x‖₂ = √(x₁² + x₂² + ... + xₙ²)
```

---

## Euclidean Distance

```text
distance(a, b) = ‖a - b‖₂
```

---

## Unit Vector

```text
x̂ = x / ‖x‖₂
```

---

## Cosine Similarity

```text
cosine(a, b) = (a · b) / (‖a‖₂ ‖b‖₂)
```

---

## Matrix Multiplication

```text
(m × n) @ (n × p) → (m × p)
```

---

## Linear Model

```text
ŷ = xᵀw + b
```

---

## Batch Linear Model

```text
Ŷ = Xw + b
```

---

# 35. Important NumPy Patterns

The notebook demonstrates several NumPy patterns that are worth remembering.

```python
# Create a vector
x = np.array([1, 2, 3])
```

```python
# Create a matrix
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

```python
# Inspect shape
X.shape
```

```python
# Vector addition
a + b
```

```python
# Vector subtraction
a - b
```

```python
# Scalar multiplication
2 * a
```

```python
# Dot product
a @ b
```

```python
# Euclidean norm
np.linalg.norm(a)
```

```python
# Transpose
X.T
```

```python
# Matrix multiplication
A @ B
```

```python
# Normalize a vector
a / np.linalg.norm(a)
```

---

# 36. Why the Visualization Helper Code Is Not the Main Focus

The notebook includes a relatively large visualization helper section.

The notebook itself notes that this code was generated with AI and should not be the main focus of the learning exercise.

That is a useful distinction.

The helper functions are implementation infrastructure used to visualize:

* vectors
* points
* distances
* directions
* similarity-related geometry

The important learning objective is the **mathematics being visualized**, not the internal implementation of the plotting helpers.

For someone learning machine learning, it is more valuable to understand:

```text
Why is this vector pointing here?
Why is the distance this value?
Why is this dot product positive?
Why does normalization change the length?
Why does A @ A.T produce pairwise relationships?
```

than to memorize the plotting code.

---

# 37. Engineering Comments Worth Preserving

Several comments in the notebook are particularly useful from an ML-engineering perspective.

## Shape-related comments

The notebook emphasizes that vectors with incompatible shapes cannot simply be added.

This is worth preserving because shape reasoning is fundamental to numerical programming.

A stronger engineering formulation is:

> Before performing an array operation, verify the shape and axis semantics of every operand.

---

## Dot-product comments

The notebook connects:

```python
features @ weights
```

to a model prediction.

This is an important bridge between linear algebra and machine learning.

The conceptual mapping is:

```text
features → input vector
weights  → model parameters
dot      → weighted sum
output   → model score
```

---

## Matrix multiplication comments

The notebook states:

```python
# Matrix to matrix multiplication - possible only if inner dimensions are matching
```

A more precise formulation is:

> Matrix multiplication is valid when the number of columns in the left matrix equals the number of rows in the right matrix.

For example:

```text
(2 × 4) @ (4 × 5)
```

is valid, while:

```text
(2 × 4) @ (2 × 5)
```

is not.

---

## Self-similarity handling

The recommendation example contains:

```python
aisha[0] = 0
```

This is a small but important algorithmic detail.

Without it, Aisha would always be her own closest match because:

```text
similarity(Aisha, Aisha) = 1
```

This is an example of converting a mathematical similarity matrix into a useful retrieval operation.

---

# 38. Terminology Guide

The following terminology makes the concepts more precise when communicating them professionally. 

| Informal wording                  | More precise terminology                               |
| --------------------------------- | ------------------------------------------------------ |
| One number                        | Scalar                                                 |
| List of numbers                   | Vector / 1-D array                                     |
| Multiple vectors combined         | Matrix / 2-D array                                     |
| N-dimensional array               | Tensor                                                 |
| Length of a vector                | Norm / magnitude                                       |
| Length between two points         | Euclidean distance                                     |
| Weighted sum                      | Linear combination / inner product                     |
| How vectors are aligned           | Directional alignment                                  |
| Similarity based on angle         | Cosine similarity                                      |
| Vector with length 1              | Unit vector                                            |
| Element-by-element multiplication | Hadamard product                                       |
| Matrix multiplication             | Matrix product                                         |
| Model score                       | Decision score / raw score / logit, depending on model |
| Most similar user                 | Nearest neighbor under the chosen similarity metric    |
| Similarity table                  | Pairwise similarity matrix                             |
| User ratings as vectors           | User preference vectors                                |
| Recommend from similar users      | User-based collaborative filtering                     |

Precise terminology becomes increasingly important as these concepts scale into more advanced machine-learning systems.

---

# 39. Engineering Takeaways

The most important lessons from this notebook are the relationships between the mathematical operations.

### 1. Data has geometry

A feature vector can be interpreted as a point or direction in a mathematical space.

---

### 2. Dot products are everywhere

The dot product appears in:

* linear models
* neural networks
* embeddings
* recommendation systems
* similarity search
* attention mechanisms

---

### 3. Norms quantify magnitude

A norm gives us a mathematical way to measure vector magnitude.

From the norm we can construct:

```text
distance
normalization
cosine similarity
```

---

### 4. Normalization changes the comparison

L2 normalization removes magnitude differences while preserving direction.

That is why:

```text
normalized vector · normalized vector
```

is equivalent to cosine similarity.

---

### 5. Matrix multiplication enables vectorization

Instead of calculating:

```python
x1 @ w
x2 @ w
x3 @ w
```

individually, we can calculate:

```python
X @ w
```

for an entire batch.

This is a fundamental pattern in efficient machine-learning computation.

---

### 6. Shape is part of the mathematics

Understanding:

```text
shape
dimension
axis
broadcasting
```

is essential for writing correct numerical ML code.

---

### 7. Simple operations can compose into ML systems

The recommendation example uses a small collection of operations:

```text
vector representation
       ↓
normalization
       ↓
dot product
       ↓
cosine similarity
       ↓
matrix multiplication
       ↓
argmax
       ↓
nearest-user retrieval
       ↓
recommendation
```

This is a powerful demonstration of how foundational mathematics can be composed into a machine-learning workflow.

---

# 40. Limitations and Scope

This notebook is intentionally focused on foundational linear algebra.

It does not attempt to cover the full mathematical toolkit required for advanced machine learning.

Topics that naturally follow include:

* vector spaces
* linear independence
* basis and span
* projections
* orthogonality
* matrix rank
* determinants
* inverse matrices
* eigenvalues
* eigenvectors
* singular value decomposition
* principal component analysis
* covariance matrices
* least-squares optimization
* gradient-based optimization
* tensors in deep-learning frameworks

The recommendation system is also intentionally simplified.

It demonstrates the mathematical principle of similarity-based recommendation rather than providing a production-ready recommender system.

---

# 41. Suggested Learning Path

A natural progression after this notebook is:

```text
Scalars / Vectors / Matrices
            ↓
Vector Operations
            ↓
Dot Product
            ↓
Norms
            ↓
Distance
            ↓
Cosine Similarity
            ↓
Normalization
            ↓
Matrix Multiplication
            ↓
Linear Regression
            ↓
Least Squares
            ↓
Projections
            ↓
Eigenvalues / Eigenvectors
            ↓
SVD / PCA
            ↓
Optimization
            ↓
Neural Networks
            ↓
Embeddings
            ↓
Attention
            ↓
Vector Retrieval
```

The objective should not be to memorize every formula.

The objective should be to recognize these operations when they appear inside real machine-learning algorithms.

---

# 42. Installation

Install the core Python dependencies with:

```bash
pip install numpy matplotlib jupyter
```

---

# 43. Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
linear_algebra_foundations.ipynb
```

The notebook can also be executed in Google Colab.

---

# 44. Repository Structure

```text
.
├── linear_algebra_foundations.ipynb
└── README.md
```

---

# 45. Final Perspective

Linear algebra is not merely a prerequisite for machine learning.

It is one of the mathematical languages in which machine-learning systems are expressed.

A surprisingly large amount of modern ML can be understood through a relatively small set of ideas:

```text
vectors
   ↓
dot products
   ↓
norms
   ↓
similarity
   ↓
normalization
   ↓
matrix multiplication
   ↓
transformations
   ↓
learned representations
```

Once these relationships become intuitive, more advanced topics such as:

* linear regression
* neural networks
* embeddings
* attention
* recommendation systems
* vector search
* dimensionality reduction

become significantly easier to reason about.

> **The objective of this notebook is not to memorize linear-algebra formulas. It is to develop the geometric and computational intuition required to recognize linear algebra inside machine-learning systems.**

---

## License

This project is intended for educational and learning purposes.
