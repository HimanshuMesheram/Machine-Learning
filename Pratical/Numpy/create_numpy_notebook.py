import json

# Complete NumPy notebook content
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NumPy: Numerical Python - Complete Guide\n",
    "\n",
    "NumPy is the fundamental package for scientific computing in Python. It provides:\n",
    "- **ndarray**: A powerful N-dimensional array object\n",
    "- **Mathematical functions**: Fast operations on arrays\n",
    "- **Linear algebra**: Matrix operations and decompositions\n",
    "- **Random number generation**: Various probability distributions\n",
    "- **Broadcasting**: Operations on arrays of different shapes\n",
    "\n",
    "This notebook covers all essential NumPy concepts with practical examples."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Installation and Import\n",
    "\n",
    "```bash\n",
    "# Install NumPy (if not already installed)\n",
    "pip install numpy\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import NumPy\n",
    "import numpy as np\n",
    "\n",
    "# Check version\n",
    "print(f\"NumPy version: {np.__version__}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Creating Arrays\n",
    "\n",
    "### 2.1 From Python Lists\n",
    "\n",
    "Creating arrays from lists using `np.array()`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1D array from list\n",
    "arr1d = np.array([1, 2, 3, 4, 5])\n",
    "print(\"1D array:\", arr1d)\n",
    "print(\"Shape:\", arr1d.shape)\n",
    "print(\"Data type:\", arr1d.dtype)\n",
    "print()\n",
    "\n",
    "# 2D array from nested lists\n",
    "arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n",
    "print(\"2D array:\\n\", arr2d)\n",
    "print(\"Shape:\", arr2d.shape)\n",
    "print(\"Data type:\", arr2d.dtype)\n",
    "print()\n",
    "\n",
    "# 3D array\n",
    "arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])\n",
    "print(\"3D array:\\n\", arr3d)\n",
    "print(\"Shape:\", arr3d.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2 Special Array Creation Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Zeros array\n",
    "zeros = np.zeros((3, 4))\n",
    "print(\"Zeros (3x4):\\n\", zeros)\n",
    "print()\n",
    "\n",
    "# Ones array\n",
    "ones = np.ones((2, 3))\n",
    "print(\"Ones (2x3):\\n\", ones)\n",
    "print()\n",
    "\n",
    "# Identity matrix\n",
    "identity = np.eye(4)\n",
    "print(\"Identity (4x4):\\n\", identity)\n",
    "print()\n",
    "\n",
    "# Full array (filled with constant value)\n",
    "full = np.full((3, 3), 7)\n",
    "print(\"Full array (3x3) filled with 7:\\n\", full)\n",
    "print()\n",
    "\n",
    "# Array with a specific data type\n",
    "float_arr = np.array([1, 2, 3], dtype=np.float64)\n",
    "print(\"Float array:\", float_arr)\n",
    "print(\"Data type:\", float_arr.dtype)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.3 Array Ranges and Sequences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# np.arange(start, stop, step)\n",
    "range1 = np.arange(10)  # 0 to 9\n",
    "print(\"arange(10):\", range1)\n",
    "\n",
    "range2 = np.arange(2, 10, 2)  # 2 to 8 step 2\n",
    "print(\"arange(2, 10, 2):\", range2)\n",
    "print()\n",
    "\n",
    "# np.linspace(start, stop, num)\n",
    "linspace = np.linspace(0, 1, 5)  # 5 equally spaced points\n",
    "print(\"linspace(0, 1, 5):\", linspace)\n",
    "\n",
    "# np.logspace(start, stop, num)\n",
    "logspace = np.logspace(0, 2, 5)  # 10^0 to 10^2\n",
    "print(\"logspace(0, 2, 5):\", logspace)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.4 Random Arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set seed for reproducibility\n",
    "np.random.seed(42)\n",
    "\n",
    "# Random values from uniform distribution [0, 1)\n",
    "uniform = np.random.random((2, 3))\n",
    "print(\"Uniform random (2x3):\\n\", uniform)\n",
    "print()\n",
    "\n",
    "# Random integers\n",
    "randint = np.random.randint(1, 100, (3, 4))  # 1 to 99\n",
    "print(\"Random integers (3x4):\\n\", randint)\n",
    "print()\n",
    "\n",
    "# Normal distribution\n",
    "normal = np.random.normal(0, 1, (2, 3))  # mean=0, std=1\n",
    "print(\"Normal distribution (2x3):\\n\", normal)\n",
    "print()\n",
    "\n",
    "# Random permutation\n",
    "perm = np.random.permutation(10)\n",
    "print(\"Random permutation of 0-9:\", perm)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Array Attributes and Properties"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a sample array\n",
    "arr = np.array([[1, 2, 3, 4],\n",
    "                [5, 6, 7, 8],\n",
    "                [9, 10, 11, 12]])\n",
    "\n",
    "print(\"Array:\\n\", arr)\n",
    "print()\n",
    "\n",
    "# Shape - dimensions of the array\n",
    "print(\"Shape:\", arr.shape)  # (rows, columns)\n",
    "\n",
    "# Size - total number of elements\n",
    "print(\"Size:\", arr.size)  # 3 * 4 = 12\n",
    "\n",
    "# Data type\n",
    "print(\"Data type:\", arr.dtype)\n",
    "\n",
    "# Number of dimensions\n",
    "print(\"Number of dimensions:\", arr.ndim)\n",
    "\n",
    "# Itemsize - size of each element in bytes\n",
    "print(\"Itemsize:\", arr.itemsize, \"bytes\")\n",
    "\n",
    "# Total bytes used\n",
    "print(\"Total bytes:\", arr.nbytes, \"bytes\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Array Indexing and Slicing\n",
    "\n",
    "### 4.1 Basic Indexing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1D array indexing\n",
    "arr1d = np.array([10, 20, 30, 40, 50])\n",
    "print(\"1D array:\", arr1d)\n",
    "print(\"arr1d[0]:\", arr1d[0])\n",
    "print(\"arr1d[-1]:\", arr1d[-1])\n",
    "print(\"arr1d[2:4]:\", arr1d[2:4])  # Slicing\n",
    "print()\n",
    "\n",
    "# 2D array indexing\n",
    "arr2d = np.array([[1, 2, 3, 4],\n",
    "                  [5, 6, 7, 8],\n",
    "                  [9, 10, 11, 12]])\n",
    "print(\"2D array:\\n\", arr2d)\n",
    "print(\"arr2d[0, 2]:\", arr2d[0, 2])  # Row 0, Col 2\n",
    "print(\"arr2d[1, :]:\", arr2d[1, :])  # Row 1, all columns\n",
    "print(\"arr2d[:, 2]:\", arr2d[:, 2])  # All rows, Col 2\n",
    "print(\"arr2d[1:3, 1:3]:\\n\", arr2d[1:3, 1:3])  # Sub-matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Fancy Indexing (Integer Array Indexing)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])\n",
    "print(\"Original array:\", arr)\n",
    "\n",
    "# Indexing with a list of indices\n",
    "indices = [1, 3, 5]\n",
    "print(\"arr[[1, 3, 5]]:\", arr[indices])\n",
    "\n",
    "# 2D fancy indexing\n",
    "arr2d = np.array([[1, 2, 3],\n",
    "                  [4, 5, 6],\n",
    "                  [7, 8, 9]])\n",
    "print(\"\\n2D array:\\n\", arr2d)\n",
    "\n",
    "# Select specific elements\n",
    "row_indices = [0, 1, 2]\n",
    "col_indices = [2, 0, 1]\n",
    "print(\"arr2d[[0,1,2], [2,0,1]]:\", arr2d[row_indices, col_indices])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.3 Boolean Indexing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])\n",
    "print(\"Original array:\", arr)\n",
    "\n",
    "# Boolean mask\n",
    "mask = arr > 5\n",
    "print(\"Mask (arr > 5):\", mask)\n",
    "print(\"arr[mask]:\", arr[mask])\n",
    "print()\n",
    "\n",
    "# More complex conditions\n",
    "condition = (arr % 2 == 0) & (arr > 3)\n",
    "print(\"Even numbers > 3:\", arr[condition])\n",
    "print()\n",
    "\n",
    "# Using where function\n",
    "even_indices = np.where(arr % 2 == 0)\n",
    "print(\"Indices of even numbers:\", even_indices)\n",
    "print(\"Even numbers:\", arr[even_indices])\n",
    "\n",
    "# Replace values based on condition\n",
    "arr_copy = arr.copy()\n",
    "arr_copy[arr_copy > 5] = 0  # Replace all >5 with 0\n",
    "print(\"After replacing >5 with 0:\", arr_copy)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Array Operations\n",
    "\n",
    "### 5.1 Arithmetic Operations (Element-wise)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.array([1, 2, 3, 4])\n",
    "b = np.array([5, 6, 7, 8])\n",
    "\n",
    "print(\"a =\", a)\n",
    "print(\"b =\", b)\n",
    "print()\n",
    "\n",
    "# Addition\n",
    "print(\"a + b =\", a + b)\n",
    "print(\"np.add(a, b) =\", np.add(a, b))\n",
    "\n",
    "# Subtraction\n",
    "print(\"a - b =\", a - b)\n",
    "\n",
    "# Multiplication\n",
    "print(\"a * b =\", a * b)\n",
    "\n",
    "# Division\n",
    "print(\"a / b =\", a / b)\n",
    "\n",
    "# Power\n",
    "print(\"a ** 2 =\", a ** 2)\n",
    "print(\"np.power(a, 3) =\", np.power(a, 3))\n",
    "\n",
    "# Square root\n",
    "print(\"np.sqrt(a) =\", np.sqrt(a))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.2 Universal Functions (ufunc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.array([1, 2, 3, 4, 5])\n",
    "\n",
    "# Trigonometric functions\n",
    "print(\"sin(arr):\", np.sin(arr))\n",
    "print(\"cos(arr):\", np.cos(arr))\n",
    "print(\"tan(arr):\", np.tan(arr))\n",
    "print()\n",
    "\n",
    "# Exponential and logarithmic\n",
    "print(\"exp(arr):\", np.exp(arr))\n",
    "print(\"log(arr):\", np.log(arr))\n",
    "print(\"log10(arr):\", np.log10(arr))\n",
    "print(\"log2(arr):\", np.log2(arr))\n",
    "print()\n",
    "\n",
    "# Absolute value\n",
    "neg_arr = np.array([-1, -2, 3, -4, 5])\n",
    "print(\"abs(neg_arr):\", np.abs(neg_arr))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.3 Statistical Operations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.array([[1, 2, 3],\n",
    "                [4, 5, 6],\n",
    "                [7, 8, 9]])\n",
    "\n",
    "print(\"Array:\\n\", arr)\n",
    "print()\n",
    "\n",
    "# Summary statistics\n",
    "print(\"Mean:\", np.mean(arr))\n",
    "print(\"Median:\", np.median(arr))\n",
    "print(\"Standard deviation:\", np.std(arr))\n",
    "print(\"Variance:\", np.var(arr))\n",
    "print(\"Min:\", np.min(arr))\n",
    "print(\"Max:\", np.max(arr))\n",
    "print(\"Sum:\", np.sum(arr))\n",
    "print(\"Product:\", np.prod(arr))\n",
    "print()\n",
    "\n",
    "# Axis-wise operations\n",
    "print(\"Mean along columns (axis=0):\", np.mean(arr, axis=0))\n",
    "print(\"Mean along rows (axis=1):\", np.mean(arr, axis=1))\n",
    "print(\"Sum along columns (axis=0):\", np.sum(arr, axis=0))\n",
    "print(\"Sum along rows (axis=1):\", np.sum(arr, axis=1))\n",
    "print()\n",
    "\n",
    "# Cumulative operations\n",
    "arr1d = np.array([1, 2, 3, 4, 5])\n",
    "print(\"Cumulative sum:\", np.cumsum(arr1d))\n",
    "print(\"Cumulative product:\", np.cumprod(arr1d))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Array Manipulation\n",
    "\n",
    "### 6.1 Reshaping and Resizing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.arange(12)\n",
    "print(\"Original array:\", arr)\n",
    "print()\n",
    "\n",
    "# Reshape - changes shape without changing data\n",
    "reshaped = arr.reshape(3, 4)\n",
    "print(\"Reshaped (3x4):\\n\", reshaped)\n",
    "\n",
    "# Reshape with -1 (auto-calculate dimension)\n",
    "reshaped2 = arr.reshape(2, -1)\n",
    "print(\"Reshaped (2x6):\\n\", reshaped2)\n",
    "print()\n",
    "\n",
    "# Flatten - convert to 1D\n",
    "arr2d = np.array([[1, 2, 3], [4, 5, 6]])\n",
    "print(\"2D array:\\n\", arr2d)\n",
    "print(\"Flattened:\", arr2d.flatten())\n",
    "print(\"Ravel (flattened view):\", arr2d.ravel())\n",
    "print()\n",
    "\n",
    "# Transpose\n",
    "print(\"Transpose:\\n\", arr2d.T)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 Concatenation and Stacking"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.array([[1, 2], [3, 4]])\n",
    "b = np.array([[5, 6], [7, 8]])\n",
    "\n",
    "print(\"a:\\n\", a)\n",
    "print(\"b:\\n\", b)\n",
    "print()\n",
    "\n",
    "# Concatenation along rows (axis=0)\n",
    "v_concat = np.concatenate((a, b), axis=0)\n",
    "print(\"Vertical concatenation (axis=0):\\n\", v_concat)\n",
    "\n",
    "# Concatenation along columns (axis=1)\n",
    "h_concat = np.concatenate((a, b), axis=1)\n",
    "print(\"Horizontal concatenation (axis=1):\\n\", h_concat)\n",
    "print()\n",
    "\n",
    "# Stacking\n",
    "v_stack = np.vstack((a, b))  # Same as concatenate axis=0\n",
    "print(\"vstack:\\n\", v_stack)\n",
    "\n",
    "h_stack = np.hstack((a, b))  # Same as concatenate axis=1\n",
    "print(\"hstack:\\n\", h_stack)\n",
    "\n",
    "d_stack = np.dstack((a, b))  # Stack along depth\n",
    "print(\"dstack (depth):\\n\", d_stack)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.3 Splitting Arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.array([[1, 2, 3, 4],\n",
    "                [5, 6, 7, 8],\n",
    "                [9, 10, 11, 12]])\n",
    "\n",
    "print(\"Original array:\\n\", arr)\n",
    "print()\n",
    "\n",
    "# Split into equal parts\n",
    "split_arr = np.split(arr, 3, axis=0)  # Split into 3 rows\n",
    "print(\"Split into 3 parts (axis=0):\")\n",
    "for i, sub_arr in enumerate(split_arr):\n",
    "    print(f\"Part {i}:\\n{sub_arr}\")\n",
    "\n",
    "# Split at specific indices\n",
    "split_arr2 = np.split(arr, [1, 2], axis=1)  # Split at columns 1 and 2\n",
    "print(\"\\nSplit at indices [1, 2] (axis=1):\")\n",
    "for i, sub_arr in enumerate(split_arr2):\n",
    "    print(f\"Part {i}:\\n{sub_arr}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Broadcasting\n",
    "\n",
    "Broadcasting allows arithmetic operations on arrays of different shapes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scalar broadcasting\n",
    "arr = np.array([1, 2, 3, 4])\n",
    "print(\"arr + 10:\", arr + 10)\n",
    "print(\"arr * 2:\", arr * 2)\n",
    "print()\n",
    "\n",
    "# Array broadcasting\n",
    "a = np.array([[1, 2, 3], [4, 5, 6]])\n",
    "b = np.array([10, 20, 30])\n",
    "\n",
    "print(\"a:\\n\", a)\n",
    "print(\"b:\", b)\n",
    "print(\"a + b:\\n\", a + b)  # b is broadcast to each row\n",
    "print()\n",
    "\n",
    "# Vector vs matrix broadcasting\n",
    "column_vec = np.array([[1], [2], [3]])\n",
    "row_vec = np.array([10, 20, 30])\n",
    "\n",
    "print(\"Column vector:\\n\", column_vec)\n",
    "print(\"Row vector:\", row_vec)\n",
    "print(\"Column + Row:\\n\", column_vec + row_vec)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Linear Algebra Operations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.array([[1, 2], [3, 4]])\n",
    "b = np.array([[5, 6], [7, 8]])\n",
    "\n",
    "print(\"a:\\n\", a)\n",
    "print(\"b:\\n\", b)\n",
    "print()\n",
    "\n",
    "# Dot product / Matrix multiplication\n",
    "dot_product = np.dot(a, b)\n",
    "print(\"Dot product (a @ b):\\n\", dot_product)\n",
    "\n",
    "# Alternative matrix multiplication\n",
    "matmul = a @ b\n",
    "print(\"Matrix multiplication (a @ b):\\n\", matmul)\n",
    "print()\n",
    "\n",
    "# Determinant\n",
    "det = np.linalg.det(a)\n",
    "print(\"Determinant of a:\", det)\n",
    "\n",
    "# Inverse\n",
    "inv = np.linalg.inv(a)\n",
    "print(\"Inverse of a:\\n\", inv)\n",
    "print(\"a @ inv:\\n\", a @ inv)  # Should be identity matrix\n",
    "print()\n",
    "\n",
    "# Eigenvalues and eigenvectors\n",
    "eigenvalues, eigenvectors = np.linalg.eig(a)\n",
    "print(\"Eigenvalues:\", eigenvalues)\n",
    "print(\"Eigenvectors:\\n\", eigenvectors)\n",
    "print()\n",
    "\n",
    "# Solving linear equations\n",
    "# Solve: 2x + 3y = 8, 4x + y = 6\n",
    "A = np.array([[2, 3], [4, 1]])\n",
    "B = np.array([8, 6])\n",
    "solution = np.linalg.solve(A, B)\n",
    "print(f\"Solution for 2x + 3y = 8, 4x + y = 6:\")\n",
    "print(f\"x = {solution[0]}, y = {solution[1]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Sorting and Searching"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sorting\n",
    "arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])\n",
    "print(\"Original:\", arr)\n",
    "print(\"Sorted:\", np.sort(arr))\n",
    "print(\"Argsort (indices):\", np.argsort(arr))\n",
    "print()\n",
    "\n",
    "# 2D sorting\n",
    "arr2d = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])\n",
    "print(\"2D array:\\n\", arr2d)\n",
    "print(\"Sort each row:\\n\", np.sort(arr2d, axis=1))\n",
    "print(\"Sort each column:\\n\", np.sort(arr2d, axis=0))\n",
    "print()\n",
    "\n",
    "# Searching\n",
    "arr = np.array([1, 2, 3, 4, 3, 2, 1])\n",
    "print(\"Array:\", arr)\n",
    "print(\"Where arr == 3:\", np.where(arr == 3))\n",
    "print(\"Unique values:\", np.unique(arr))\n",
    "print(\"Count of each unique value:\", np.unique(arr, return_counts=True))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Copies and Views\n",
    "\n",
    "Understanding the difference between copies and views is crucial for memory efficiency."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = np.array([1, 2, 3, 4, 5])\n",
    "\n",
    "# View (shallow copy) - shares data\n",
    "view = arr.view()\n",
    "print(\"Original arr:\", arr)\n",
    "print(\"View:\", view)\n",
    "view[0] = 100\n",
    "print(\"After modifying view[0] to 100:\")\n",
    "print(\"Original arr:\", arr)  # Modified!\n",
    "print(\"View:\", view)\n",
    "print()\n",
    "\n",
    "# Copy (deep copy) - independent data\n",
    "arr = np.array([1, 2, 3, 4, 5])\n",
    "copy = arr.copy()\n",
    "print(\"Original arr:\", arr)\n",
    "print(\"Copy:\", copy)\n",
    "copy[0] = 100\n",
    "print(\"After modifying copy[0] to 100:\")\n",
    "print(\"Original arr:\", arr)  # Not modified!\n",
    "print(\"Copy:\", copy)\n",
    "print()\n",
    "\n",
    "# Base attribute\n",
    "arr = np.array([1, 2, 3, 4])\n",
    "view = arr.view()\n",
    "copy = arr.copy()\n",
    "print(\"arr.base:\", arr.base)  # None for original array\n",
    "print(\"view.base:\", view.base is arr)  # True\n",
    "print(\"copy.base:\", copy.base is arr)  # False"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Advanced Features\n",
    "\n",
    "### 11.1 Masked Arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a masked array\n",
    "data = np.array([1, 2, -999, 4, -999, 6])\n",
    "mask = data == -999\n",
    "masked_arr = np.ma.masked_array(data, mask=mask)\n",
    "\n",
    "print(\"Data:\", data)\n",
    "print(\"Mask:\", mask)\n",
    "print(\"Masked array:\", masked_arr)\n",
    "print(\"Mean without masked values:\", np.mean(masked_arr))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 11.2 Structured Arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define structured data type\n",
    "dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f8')]\n",
    "\n",
    "# Create structured array\n",
    "data = [('Alice', 25, 165.5), ('Bob', 30, 175.2), ('Charlie', 35, 180.0)]\n",
    "structured = np.array(data, dtype=dtype)\n",
    "\n",
    "print(\"Structured array:\")\n",
    "print(structured)\n",
    "print(\"\\nNames:\", structured['name'])\n",
    "print(\"Ages:\", structured['age'])\n",
    "print(\"Heights:\", structured['height'])\n",
    "\n",
    "# Filtering\n",
    "filtered = structured[structured['age'] > 28]\n",
    "print(\"\\nPeople older than 28:\")\n",
    "print(filtered)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 11.3 Performance Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "\n",
    "# Create a large array\n",
    "size = 1000000\n",
    "arr = np.arange(size)\n",
    "\n",
    "# NumPy operation\n",
    "start_time = time.time()\n",
    "result_numpy = arr * 2\n",
    "numpy_time = time.time() - start_time\n",
    "\n",
    "# Python list operation\n",
    "lst = list(range(size))\n",
    "start_time = time.time()\n",
    "result_python = [x * 2 for x in lst]\n",
    "python_time = time.time() - start_time\n",
    "\n",
    "print(f\"NumPy time: {numpy_time:.4f} seconds\")\n",
    "print(f\"Python list time: {python_time:.4f} seconds\")\n",
    "print(f\"NumPy is {python_time/numpy_time:.1f}x faster!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Practical Examples\n",
    "\n",
    "### 12.1 Image Processing (Grayscale)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simulate an RGB image (3 channels)\n",
    "image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)\n",
    "print(\"Image shape:\", image.shape)\n",
    "\n",
    "# Convert to grayscale using weighted average\n",
    "weights = np.array([0.299, 0.587, 0.114])\n",
    "grayscale = np.dot(image, weights).astype(np.uint8)\n",
    "print(\"Grayscale shape:\", grayscale.shape)\n",
    "\n",
    "# Apply threshold\n",
    "threshold = 128\n",
    "binary = grayscale > threshold\n",
    "print(\"Binary shape:\", binary.shape)\n",
    "print(f\"Percentage of white pixels: {np.mean(binary) * 100:.1f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 12.2 Data Analysis Example"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate sample data\n",
    "np.random.seed(42)\n",
    "data = np.random.normal(100, 15, 1000)\n",
    "\n",
    "# Basic statistics\n",
    "print(\"=== Data Statistics ===\")\n",
    "print(f\"Mean: {np.mean(data):.2f}\")\n",
    "print(f\"Median: {np.median(data):.2f}\")\n",
    "print(f\"Std Dev: {np.std(data):.2f}\")\n",
    "print(f\"Min: {np.min(data):.2f}\")\n",
    "print(f\"Max: {np.max(data):.2f}\")\n",
    "print(f\"Q1 (25th percentile): {np.percentile(data, 25):.2f}\")\n",
    "print(f\"Q3 (75th percentile): {np.percentile(data, 75):.2f}\")\n",
    "\n",
    "# Outlier detection using IQR method\n",
    "q1 = np.percentile(data, 25)\n",
    "q3 = np.percentile(data, 75)\n",
    "iqr = q3 - q1\n",
    "lower_bound = q1 - 1.5 * iqr\n",
    "upper_bound = q3 + 1.5 * iqr\n",
    "outliers = data[(data < lower_bound) | (data > upper_bound)]\n",
    "print(f\"\\nOutliers detected: {len(outliers)}\")\n",
    "print(f\"Outlier percentage: {len(outliers)/len(data)*100:.1f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "### Key NumPy Concepts Covered:\n",
    "\n",
    "1. **Creating Arrays**: From lists, special functions, random generation\n",
    "2. **Array Properties**: shape, size, dtype, ndim\n",
    "3. **Indexing**: Basic, fancy, boolean indexing\n",
    "4. **Operations**: Arithmetic, universal functions, statistics\n",
    "5. **Manipulation**: Reshaping, concatenation, splitting\n",
    "6. **Broadcasting**: Operations on arrays of different shapes\n",
    "7. **Linear Algebra**: Dot product, inverse, eigenvalues, solving equations\n",
    "8. **Sorting and Searching**: Sorting, argsort, where, unique\n",
    "9. **Memory Management**: Views vs copies\n",
    "10. **Advanced Features**: Masked arrays, structured arrays\n",
    "\n",
    "### Why NumPy?\n",
    "- **Performance**: Much faster than Python lists for numerical operations\n",
    "- **Convenience**: Vectorized operations eliminate loops\n",
    "- **Memory Efficiency**: Compact storage with homogeneous data types\n",
    "- **Integration**: Foundation for SciPy, Pandas, Matplotlib, scikit-learn\n",
    "- **Community**: Extensive documentation and active development"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# Save the notebook
with open('NumPy_Complete_Guide.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ NumPy_Complete_Guide.ipynb has been created successfully!")
print("You can now open this file in Jupyter Notebook or Jupyter Lab.")