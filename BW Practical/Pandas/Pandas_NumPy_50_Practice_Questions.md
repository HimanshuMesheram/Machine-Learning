# 🔢 50 Pandas + NumPy Integration Practice Questions
> **Instructions:** Each question requires both **Pandas** and **NumPy**. Focus on interoperability, vectorized operations, and building ML-ready datasets.  
> **Do NOT look at solutions** — hints are provided to guide your thinking.  
> Recommended pace: 5–10 questions per day

---

## 🔹 NP1–NP10: Array Conversions & Basic Integration

---

### NP1. Create a NumPy array and convert it to a Pandas Series.
> 💡 **Hint:** `np.array([...])` creates the array. Pass it directly to `pd.Series(array, index=[...], name="...")`.

---

### NP2. Convert a 2D NumPy array to a Pandas DataFrame.
> 💡 **Hint:** `pd.DataFrame(np_2d_array, columns=[...])` — shape `(n, m)` gives `n` rows and `m` columns.

---

### NP3. Generate a random dataset using NumPy and store it in a DataFrame.
> 💡 **Hint:** Use `np.random.seed()` for reproducibility. Try `np.random.randn(n, m)` for normal distribution or `np.random.randint(lo, hi, n)` for integers.

---

### NP4. Perform element-wise operations using NumPy and Pandas.
> 💡 **Hint:** You can apply NumPy ufuncs directly on a Pandas Series: `np.sqrt(df["col"])`. The result is a Series.

---

### NP5. Apply `np.mean` on DataFrame columns.
> 💡 **Hint:** `np.mean(df["col"])` works the same as `df["col"].mean()`. Apply to the `.values` array for pure NumPy speed: `np.mean(df["col"].values)`.

---

### NP6. Replace values using `np.where`.
> 💡 **Hint:** `np.where(condition, value_if_true, value_if_false)` — assign the result back to a column: `df["col"] = np.where(df["col"] > 50, "High", "Low")`.

---

### NP7. Apply logarithmic transformation using `np.log`.
> 💡 **Hint:** `df["col_log"] = np.log(df["col"])`. Note: `np.log(0)` returns `-inf`. Use `np.log1p(x)` which computes `log(1 + x)` to safely handle zeros.

---

### NP8. Apply square root transformation using `np.sqrt`.
> 💡 **Hint:** `df["col_sqrt"] = np.sqrt(df["col"])`. Values must be non-negative; use `np.clip(df["col"], 0, None)` first if negatives are possible.

---

### NP9. Use NumPy functions inside `apply()`.
> 💡 **Hint:** `df["col"].apply(lambda x: np.log(x))` — or pass the NumPy function directly: `df["col"].apply(np.log)`.

---

### NP10. Perform broadcasting in a DataFrame using NumPy.
> 💡 **Hint:** Extract `.values` (a 2D NumPy array) and multiply by a weights array: `df[cols].values @ weights_array`. The result can be assigned back as a new column.

---

## 🔹 NP11–NP20: Data Filtering, Transformation & Statistics

---

### NP11. Filter DataFrame rows using a NumPy boolean mask.
> 💡 **Hint:** `mask = np.array(df["col"]) > threshold` creates a NumPy boolean array. Use `df[mask]` to filter.

---

### NP12. Count non-zero values in a column using NumPy.
> 💡 **Hint:** `np.count_nonzero(df["col"].values)` or `np.count_nonzero(df["col"].values != 0)`.

---

### NP13. Find unique values using `np.unique`.
> 💡 **Hint:** `np.unique(df["col"])` returns sorted unique values. Use `return_counts=True` to also get the frequency of each unique value.

---

### NP14. Flatten DataFrame values to a 1D NumPy array.
> 💡 **Hint:** `df[["col1","col2"]].values.flatten()` — the `.flatten()` method converts the 2D array to 1D row-by-row.

---

### NP15. Reshape a NumPy array before creating a DataFrame.
> 💡 **Hint:** `np.arange(1, 25).reshape(6, 4)` creates a 6×4 array. Pass directly to `pd.DataFrame(arr, columns=[...])`.

---

### NP16. Perform matrix multiplication using `np.dot`.
> 💡 **Hint:** `np.dot(A, B)` or `A @ B`. Extract `.values` from DataFrames first, then wrap the result in a new `pd.DataFrame(result, columns=[...])`.

---

### NP17. Normalize data using NumPy (Min-Max scaling).
> 💡 **Hint:** `arr = df[cols].values` then `(arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))`. Use `axis=0` to operate column-wise.

---

### NP18. Standardize data using Z-score with NumPy.
> 💡 **Hint:** `z = (arr - arr.mean(axis=0)) / arr.std(axis=0)`. Wrap the result in a DataFrame using the original column names.

---

### NP19. Clip values using `np.clip`.
> 💡 **Hint:** `np.clip(df["col"].values, a_min=lower_bound, a_max=upper_bound)` — values outside the range are replaced by the boundary values.

---

### NP20. Handle missing values using NumPy.
> 💡 **Hint:** Detect NaN with `np.isnan(arr)`. Fill using `np.nanmean(arr)` to compute the mean ignoring NaN, then replace: `arr[np.isnan(arr)] = np.nanmean(arr)`.

---

## 🔹 NP21–NP30: Performance, Sorting & Statistical Analysis

---

### NP21. Compare NumPy vs Pandas performance on a large array.
> 💡 **Hint:** Use `time.time()` before and after each operation. Apply the same computation (`arr * 2 + 1`) to a raw NumPy array and a Pandas Series. NumPy is typically faster on pure numeric operations.

---

### NP22. Calculate cumulative sum using NumPy.
> 💡 **Hint:** `np.cumsum(df["col"].values)` — assign back: `df["cum_col"] = np.cumsum(df["col"])`.

---

### NP23. Use `np.argmax` and `np.argmin`.
> 💡 **Hint:** `np.argmax(df["col"].values)` returns the **integer position** of the maximum. Use `df.iloc[idx]` or `df.loc[idx]` to retrieve the full row.

---

### NP24. Sort values using NumPy `np.argsort`.
> 💡 **Hint:** `sorted_idx = np.argsort(df["col"].values)` gives indices that would sort the array. Use `df.iloc[sorted_idx]` to reorder the DataFrame.

---

### NP25. Compute correlation using `np.corrcoef`.
> 💡 **Hint:** `np.corrcoef([df["col1"].values, df["col2"].values])` returns a 2×2 correlation matrix. Wrap in a DataFrame for readability. Compare with `df.corr()`.

---

### NP26. Detect NaN values using NumPy.
> 💡 **Hint:** `np.isnan(arr)` returns a boolean array. Count NaN with `np.isnan(arr).sum()`. Compare with `pd.isnull(series)`.

---

### NP27. Create boolean masks using NumPy.
> 💡 **Hint:** `mask = (np.array(df["col1"]) > 80) & (np.array(df["col2"]) < 40)` — use `df[mask]` to filter.

---

### NP28. Apply logical operations (`np.logical_and`, `np.logical_or`, `np.logical_not`).
> 💡 **Hint:** `np.logical_and(cond1, cond2)` is equivalent to `cond1 & cond2`. These functions work on NumPy arrays; convert Series with `.values` first.

---

### NP29. Apply `np.select` for multi-condition assignment.
> 💡 **Hint:** `df["Grade"] = np.select([df["Score"] >= 90, df["Score"] >= 75], ["A","B"], default="C")` — conditions are checked in order; the first match wins.

---

### NP30. Multiply DataFrame columns element-wise using NumPy.
> 💡 **Hint:** `np.multiply(df["col1"].values, df["col2"].values)` is equivalent to `df["col1"] * df["col2"]` but uses NumPy internally.

---

## 🔹 NP31–NP40: Data Wrangling & Conversion

---

### NP31. Shuffle dataset rows using NumPy.
> 💡 **Hint:** `shuffled_idx = np.random.permutation(df.index)` creates a random permutation of row indices. Use `df.loc[shuffled_idx].reset_index(drop=True)`.

---

### NP32. Sample random rows using NumPy.
> 💡 **Hint:** `sample_idx = np.random.choice(df.index, size=N, replace=False)` — use `df.loc[sample_idx]`. Compare with `df.sample(N)`.

---

### NP33. Calculate percentiles using NumPy.
> 💡 **Hint:** `np.percentile(df["col"].values, [25, 50, 75])` returns Q1, median, Q3. Use `np.percentile(arr, p)` for any percentile p in [0, 100].

---

### NP34. Calculate variance and standard deviation.
> 💡 **Hint:** `np.var(arr)` and `np.std(arr)` use `ddof=0` (population). Pandas `.var()` and `.std()` use `ddof=1` (sample). Be aware of this difference when comparing results.

---

### NP35. Replace a DataFrame column with a NumPy array.
> 💡 **Hint:** `df["col"] = new_numpy_array` — the array length must match the number of rows. NumPy arrays are accepted directly.

---

### NP36. Convert DataFrame to NumPy using `to_numpy()`.
> 💡 **Hint:** `df[["col1","col2"]].to_numpy()` returns a 2D NumPy array. This is the preferred method over `.values` as it accepts `dtype` and `na_value` parameters.

---

### NP37. Compare `.values` vs `.to_numpy()`.
> 💡 **Hint:** Both return a NumPy array. Key difference: `.to_numpy(dtype=np.float32)` accepts a `dtype` argument for explicit casting. Use `np.array_equal(a, b)` to confirm they produce the same result.

---

### NP38. Slice a column using NumPy array indexing.
> 💡 **Hint:** Extract with `.values`, then apply NumPy slicing: `arr[::2]` (every other element), `arr[::-1]` (reversed), `arr[[0,3,5]]` (fancy indexing).

---

### NP39. Merge NumPy-generated data into a DataFrame.
> 💡 **Hint:** Create two DataFrames from NumPy arrays with a common key column (e.g., `ID`). Use `pd.merge(df1, df2, on="ID", how="outer")`.

---

### NP40. Apply rolling computation using NumPy.
> 💡 **Hint:** Use a list comprehension: `[np.mean(arr[max(0,i-w+1):i+1]) for i in range(len(arr))]` to manually compute a rolling mean with window `w`. Compare with `df["col"].rolling(w).mean()`.

---

## 🔹 NP41–NP50: ML-Ready Dataset Preparation

---

### NP41. Create a synthetic ML dataset using NumPy.
> 💡 **Hint:** Use `np.random.normal`, `np.random.randint`, `np.random.choice` to generate features. Use `np.where` or `np.select` to create a binary/multi-class label column.

---

### NP42. Detect and replace outliers using NumPy.
> 💡 **Hint:** Compute Q1, Q3 with `np.percentile`. Define bounds as `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR`. Use `np.clip(arr, lo, hi)` to replace outliers with boundary values.

---

### NP43. Scale dataset using NumPy functions.
> 💡 **Hint:** Apply Min-Max or Z-score scaling using NumPy on the `.values` array. Store results as new columns in the DataFrame.

---

### NP44. Engineer new features using NumPy.
> 💡 **Hint:** Create polynomial features (`np.square`, `np.power`), log features (`np.log1p`), interaction features (column1 × column2), and binned features (`np.digitize`).

---

### NP45. Use exponential functions (`np.exp`).
> 💡 **Hint:** `np.exp(df["col"])` computes `e^x` for each element. This is the inverse of `np.log`. Useful for reverting log-transformed values.

---

### NP46. Compute differences using `np.diff`.
> 💡 **Hint:** `np.diff(df["col"].values)` returns an array of consecutive differences (length = n−1). Prepend `np.nan` and assign back: `df["col_diff"] = np.insert(np.diff(arr), 0, np.nan)`.

---

### NP47. Apply conditional encoding using NumPy.
> 💡 **Hint:** Use `np.select` with a list of conditions and corresponding integer codes to encode categorical columns numerically. For example: `DS → 0`, `ML → 1`, `AI → 2`.

---

### NP48. Build an ML-ready dataset (feature matrix X and label vector y).
> 💡 **Hint:** Select numeric feature columns → call `.to_numpy(dtype=np.float32)` → store as `X`. Create a binary label: `y = (df["col"] >= threshold).astype(int).to_numpy()`. Check shapes with `X.shape` and `y.shape`.

---

### NP49. Perform a mini data analysis project.
> 💡 **Hint:** Follow these steps: (1) Generate data with NumPy, (2) Store in DataFrame, (3) Clean with Pandas, (4) Engineer features with NumPy, (5) GroupBy analysis, (6) Detect outliers, (7) Export summary.

---

### NP50. Build an end-to-end pipeline using NumPy + Pandas.
> 💡 **Hint:** Chain all steps together:  
> 1. `np.random.seed(X)` → generate raw data  
> 2. `np.clip` → clean invalid values  
> 3. `np.log`, `np.square`, `np.where` → feature engineering  
> 4. `df.groupby().agg(...)` → statistical analysis  
> 5. IQR + `np.clip` → outlier removal  
> 6. `df[features].to_numpy(dtype=np.float32)` → ML feature matrix  
> 7. Export with `df.to_csv(...)`

---

## 🎯 Quick Hints Reference

| Task | NumPy Function | Pandas Equivalent |
|------|---------------|-------------------|
| Convert to array | `arr = df["col"].values` | `arr = df["col"].to_numpy()` |
| Conditional assign | `np.where(cond, a, b)` | `df.loc[cond, "col"] = val` |
| Multi-condition | `np.select([c1,c2],[v1,v2])` | `map()` / `pd.cut()` |
| Clip values | `np.clip(arr, lo, hi)` | `df["col"].clip(lo, hi)` |
| Count NaN | `np.isnan(arr).sum()` | `df.isnull().sum()` |
| Correlation | `np.corrcoef([a, b])` | `df.corr()` |
| Percentile | `np.percentile(arr, q)` | `df["col"].quantile(q/100)` |
| Cumulative sum | `np.cumsum(arr)` | `df["col"].cumsum()` |
| Sort indices | `np.argsort(arr)` | `df.sort_values(...).index` |
| Unique values | `np.unique(arr)` | `df["col"].unique()` |
| Reshape | `arr.reshape(r, c)` | — (use before DataFrame creation) |
| Min-Max scale | `(x - x.min()) / (x.max()-x.min())` | Manual or sklearn |
| Z-score | `(x - x.mean()) / x.std()` | Manual or sklearn |

---

*📌 Always try solving without hints first.*  
*⭐ Happy Learning NumPy + Pandas — the backbone of every ML project!*
