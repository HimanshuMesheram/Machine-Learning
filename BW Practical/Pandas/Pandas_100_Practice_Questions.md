# 🐼 100 Pandas Practical Questions
> **Instructions:** Try to solve each question on your own. Use the hints only if you are stuck.  
> Progress: Basic → Intermediate → Advanced  
> Recommended pace: 10–15 questions per day

---

## 🔹 Section 1: Basics & Series (Q1–Q20)

---

### Q1. Create a Pandas Series from a list of integers.
> 💡 **Hint:** Use `pd.Series([...])`. The default index starts from 0.

---

### Q2. Create a Series from a dictionary with custom index.
> 💡 **Hint:** Pass a `dict` to `pd.Series()`. Keys become the index labels automatically.

---

### Q3. Access elements using index labels.
> 💡 **Hint:** Use `s["label"]` for a single value, `s[["a","b"]]` for multiple, and `s["a":"c"]` for a range. Label slices are **inclusive** on both ends.

---

### Q4. Perform arithmetic operations on a Series.
> 💡 **Hint:** Use `+`, `-`, `*`, `/`, `//`, `%`, `**` directly on the Series. Operations are **element-wise** and **index-aligned**.

---

### Q5. Find mean, min, max of a Series.
> 💡 **Hint:** Use `.mean()`, `.min()`, `.max()`, `.sum()`, `.median()`, `.std()` directly on the Series.

---

### Q6. Apply conditional filtering on a Series.
> 💡 **Hint:** Use a boolean mask: `s[s > 50]`. You can also chain conditions with `&` and `|`.

---

### Q7. Replace values in a Series.
> 💡 **Hint:** Use `.replace(old_val, new_val)` or pass a dictionary `{old: new}` to replace multiple values at once.

---

### Q8. Convert Series data type.
> 💡 **Hint:** Use `.astype(dtype)` — e.g., `.astype(int)`, `.astype(float)`, `.astype("category")`.

---

### Q9. Check for null values in Series.
> 💡 **Hint:** Use `.isnull()` (returns boolean Series) or `.isnull().sum()` for a count.

---

### Q10. Fill missing values in Series.
> 💡 **Hint:** Use `.fillna(value)` for constant, `.fillna(s.mean())` for mean, `.ffill()` for forward fill, `.bfill()` for backward fill.

---

### Q11. Create DataFrame from a list.
> 💡 **Hint:** Use `pd.DataFrame(list_of_lists, columns=[...])`. Each inner list becomes one row.

---

### Q12. Create DataFrame from a dictionary.
> 💡 **Hint:** Use `pd.DataFrame({col: values})`. Keys become column names; values must have equal length.

---

### Q13. Create DataFrame from list of dictionaries.
> 💡 **Hint:** Use `pd.DataFrame([{...}, {...}])`. Each dict is a row; missing keys become `NaN`.

---

### Q14. Display DataFrame structure (shape, columns, dtypes).
> 💡 **Hint:** Use `df.shape`, `df.columns`, `df.dtypes`, `df.info()`, and `df.describe()`.

---

### Q15. Select single column from DataFrame.
> 💡 **Hint:** Use `df["col_name"]` — this returns a **Series**. You can also use `df.col_name` (dot notation).

---

### Q16. Select multiple columns.
> 💡 **Hint:** Use `df[["col1", "col2"]]` with a list inside the brackets — this returns a **DataFrame**.

---

### Q17. Add new column with scalar value.
> 💡 **Hint:** `df["new_col"] = 10` — the scalar is **broadcast** to all rows automatically.

---

### Q18. Add column using Series.
> 💡 **Hint:** `df["new_col"] = pd.Series([...])`. The Series index must align with the DataFrame index.

---

### Q19. Rename columns using `rename()`.
> 💡 **Hint:** Use `df.rename(columns={"old": "new"})`. Add `inplace=True` to modify in-place.

---

### Q20. Rename multiple columns.
> 💡 **Hint:** Pass a dictionary with multiple key-value pairs to `rename(columns={...})`.

---

## 🔹 Section 2: Indexing, Selection & Slicing (Q21–Q40)

---

### Q21. Select rows using `loc`.
> 💡 **Hint:** `df.loc[label]` for a single row, `df.loc[label1:label2]` for a range. **Inclusive on both ends.**

---

### Q22. Select rows using `iloc`.
> 💡 **Hint:** `df.iloc[0]` for first row, `df.iloc[1:4]` for rows 1–3. **Exclusive on the end** (Python-style).

---

### Q23. Access specific row and column.
> 💡 **Hint:** `df.loc[row_label, "col_name"]` or `df.iloc[row_pos, col_pos]` for a single cell.

---

### Q24. Select subset using multiple rows & columns.
> 💡 **Hint:** `df.loc[[r1,r2], ["col1","col2"]]` or `df.iloc[[0,1], [0,2]]`.

---

### Q25. Apply conditional filtering using `loc`.
> 💡 **Hint:** `df.loc[df["col"] > value, ["col1","col2"]]` — pass a boolean Series as the row selector.

---

### Q26. Slice rows using index range.
> 💡 **Hint:** Use `df.iloc[start:end]` for position-based slicing or `df.loc[label1:label2]` for label-based slicing.

---

### Q27. Slice columns using labels.
> 💡 **Hint:** `df.loc[:, "col1":"col3"]` — selects all rows, columns from col1 to col3 (label-inclusive).

---

### Q28. Slice columns using position.
> 💡 **Hint:** `df.iloc[:, 1:4]` — selects all rows, columns at positions 1, 2, 3.

---

### Q29. Perform combined row & column slicing.
> 💡 **Hint:** `df.loc[row_condition, ["col1","col2"]]` or `df.iloc[0:5, [0, 2, 4]]`.

---

### Q30. Set a column as index.
> 💡 **Hint:** Use `df.set_index("col_name")`. Add `inplace=True` to modify without reassigning.

---

### Q31. Create hierarchical index (MultiIndex).
> 💡 **Hint:** `df.set_index(["col1","col2"])` — pass a **list** of column names to create a MultiIndex.

---

### Q32. Reset index.
> 💡 **Hint:** `df.reset_index()` — moves the current index back to a column. Use `inplace=True` to modify in-place.

---

### Q33. Reset multi-index.
> 💡 **Hint:** `df.reset_index()` on a MultiIndex DataFrame flattens both index levels back to columns.

---

### Q34. Convert index to column.
> 💡 **Hint:** `df.reset_index()` by default moves the index to a column. Use `drop=False` (default) to keep it.

---

### Q35. Use drop option in `reset_index`.
> 💡 **Hint:** `df.reset_index(drop=True)` discards the old index completely instead of converting it to a column.

---

### Q36. Perform label-based indexing.
> 💡 **Hint:** Use `df.loc[label]` — works with any index type (string, integer, datetime labels).

---

### Q37. Perform position-based indexing.
> 💡 **Hint:** Use `df.iloc[0]` — always integer-based, regardless of the index type.

---

### Q38. Access first N rows.
> 💡 **Hint:** `df.head(N)` — default N is 5.

---

### Q39. Access last N rows.
> 💡 **Hint:** `df.tail(N)` — default N is 5.

---

### Q40. Extract random sample from DataFrame.
> 💡 **Hint:** `df.sample(n=5)` for random rows, `df.sample(frac=0.2)` for 20% of rows. Set `random_state` for reproducibility.

---

## 🔹 Section 3: Data Cleaning (Q41–Q60)

---

### Q41. Detect missing values using `isnull()`.
> 💡 **Hint:** `df.isnull()` returns a boolean DataFrame. Use `.isnull().sum()` for column-wise counts or `.isnull().any()` to check if any exist.

---

### Q42. Drop missing values using `dropna()`.
> 💡 **Hint:** `df.dropna()` drops any row with a NaN. Use `axis=1` for columns, `subset=["col"]` for specific columns, and `thresh=N` to keep rows with at least N non-null values.

---

### Q43. Fill missing values with a constant.
> 💡 **Hint:** `df.fillna(0)` or `df["col"].fillna("Unknown")`. You can also pass a dict to fill different columns with different values.

---

### Q44. Fill missing values using mean/median.
> 💡 **Hint:** `df["col"].fillna(df["col"].mean())` or `df["col"].fillna(df["col"].median())`.

---

### Q45. Use forward fill and backward fill.
> 💡 **Hint:** `df.ffill()` propagates the last valid value forward. `df.bfill()` fills using the next valid value.

---

### Q46. Convert data type using `astype()`.
> 💡 **Hint:** `df["col"].astype(int)` or `df["col"].astype("category")`. Use `errors="coerce"` inside `pd.to_numeric()` to handle invalid values.

---

### Q47. Clean string column using `.str.lower()`.
> 💡 **Hint:** Access the `.str` accessor first: `df["col"].str.lower()`. Similarly `.str.upper()`, `.str.title()`.

---

### Q48. Trim whitespace in strings.
> 💡 **Hint:** `df["col"].str.strip()` removes leading and trailing spaces. Use `.str.lstrip()` / `.str.rstrip()` for one side.

---

### Q49. Replace substring in column.
> 💡 **Hint:** `df["col"].str.replace("old", "new")`. Add `regex=True` for pattern-based replacement.

---

### Q50. Extract part of string using `.str`.
> 💡 **Hint:** `df["col"].str[:3]` for first 3 chars. Use `.str.split("@").str[1]` to extract after a delimiter.

---

### Q51. Detect duplicate rows.
> 💡 **Hint:** `df.duplicated()` returns a boolean Series. Use `df[df.duplicated()]` to see the actual duplicate rows.

---

### Q52. Remove duplicates using `drop_duplicates()`.
> 💡 **Hint:** `df.drop_duplicates()` keeps the first occurrence by default. Use `subset=["col"]` to check specific columns, and `keep="last"` or `keep=False`.

---

### Q53. Rename row labels.
> 💡 **Hint:** `df.rename(index={old_label: new_label})`.

---

### Q54. Handle inconsistent data.
> 💡 **Hint:** Use `.str.strip().str.lower()` to normalize strings. Use `.replace({...})` to map inconsistent values to a standard form.

---

### Q55. Replace values in DataFrame.
> 💡 **Hint:** `df.replace("old", "new")` works DataFrame-wide. Pass a dict `{old: new}` or nested dict `{"col": {old: new}}` for column-specific replacement.

---

### Q56. Detect outliers in a column.
> 💡 **Hint:** Use the **IQR method**: compute Q1, Q3, IQR = Q3–Q1. Outliers are values below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR`. Alternatively use Z-score: `|z| > 3`.

---

### Q57. Handle outliers using clipping.
> 💡 **Hint:** `df["col"].clip(lower=lo, upper=hi)` replaces values outside the bounds with the boundary values.

---

### Q58. Standardize column names.
> 💡 **Hint:** `df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")` — clean all column names in one line.

---

### Q59. Handle mixed data types in a column.
> 💡 **Hint:** Use `pd.to_numeric(df["col"], errors="coerce")` to convert to numbers, turning invalid strings into `NaN`.

---

### Q60. Remove special characters from text data.
> 💡 **Hint:** `df["col"].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)` — use regex to keep only alphanumeric characters.

---

## 🔹 Section 4: Data Manipulation (Q61–Q80)

---

### Q61. Filter rows using AND condition.
> 💡 **Hint:** `df[(df["col1"] > val) & (df["col2"] == "x")]` — wrap each condition in parentheses and use `&`.

---

### Q62. Filter rows using OR condition.
> 💡 **Hint:** `df[(df["col1"] > val) | (df["col2"] == "x")]` — use `|` between conditions.

---

### Q63. Filter rows using NOT condition.
> 💡 **Hint:** `df[~(df["col"] == "val")]` — use `~` to negate a boolean condition.

---

### Q64. Sort DataFrame by column values.
> 💡 **Hint:** `df.sort_values("col")`. Use `ascending=False` for descending, `by=["col1","col2"]` for multi-column sort.

---

### Q65. Sort DataFrame by index.
> 💡 **Hint:** `df.sort_index()`. Use `axis=1` to sort columns alphabetically.

---

### Q66. Add calculated column.
> 💡 **Hint:** `df["new"] = df["col1"] * df["col2"]` or use `.assign(new=lambda x: ...)` for chainable operations.

---

### Q67. Remove column using `drop()`.
> 💡 **Hint:** `df.drop(columns=["col1","col2"])`. Always use the `columns=` parameter to be explicit.

---

### Q68. Remove rows using `drop()`.
> 💡 **Hint:** `df.drop(index=[0, 3])` removes rows with those index labels.

---

### Q69. Use `map()` on a column.
> 💡 **Hint:** `df["col"].map({"a": 1, "b": 2})` substitutes values using a dict. Pass a function for element-wise transformation.

---

### Q70. Apply function using `apply()`.
> 💡 **Hint:** `df["col"].apply(func)` for a Series. `df.apply(func, axis=0)` column-wise, `df.apply(func, axis=1)` row-wise.

---

### Q71. Apply element-wise using `applymap()` / `map()`.
> 💡 **Hint:** In older pandas use `df.applymap(func)`. In pandas ≥ 2.1 use `df.map(func)` for element-wise operations on the entire DataFrame.

---

### Q72. Chain multiple operations.
> 💡 **Hint:** Use method chaining with `(df.copy().loc[...].sort_values(...)[cols].assign(...).reset_index(drop=True))` — wrap in parentheses for multi-line readability.

---

### Q73. Normalize numeric columns (Min-Max).
> 💡 **Hint:** `(df["col"] - df["col"].min()) / (df["col"].max() - df["col"].min())` — result is in range [0, 1].

---

### Q74. Scale numerical data (Z-score standardization).
> 💡 **Hint:** `(df["col"] - df["col"].mean()) / df["col"].std()` — result has mean=0 and std=1.

---

### Q75. Rank values in a column.
> 💡 **Hint:** `df["col"].rank()` — use `ascending=False` for descending rank, `method="dense"` to avoid gaps.

---

### Q76. Create bins using `cut()`.
> 💡 **Hint:** `pd.cut(df["col"], bins=4)` for equal-width bins or `pd.cut(df["col"], bins=[0,25,50,75,100], labels=["Low","Mid","High","Top"])` for custom intervals.

---

### Q77. Create bins using `qcut()`.
> 💡 **Hint:** `pd.qcut(df["col"], q=4)` for **equal-frequency** (quantile-based) bins. Use `labels=["Q1","Q2","Q3","Q4"]` for named bins.

---

### Q78. Count unique values.
> 💡 **Hint:** `df["col"].nunique()` for the count, `df["col"].value_counts()` for frequency per unique value.

---

### Q79. Replace multiple values at once.
> 💡 **Hint:** `df.replace({"col": {"old1": "new1", "old2": "new2"}})` or use a list: `df.replace(["old1","old2"], "new")`.

---

### Q80. Create flag column based on condition.
> 💡 **Hint:** `df["flag"] = (df["col"] > threshold).astype(int)` — converts True/False to 1/0. Or use `np.where(condition, 1, 0)`.

---

## 🔹 Section 5: Iteration, GroupBy & Pivot (Q81–Q100)

---

### Q81. Iterate over rows using `iterrows()`.
> 💡 **Hint:** `for idx, row in df.iterrows():` — `idx` is the index label, `row` is a Series with column values.

---

### Q82. Iterate using `itertuples()`.
> 💡 **Hint:** `for row in df.itertuples():` — `row.col_name` accesses values. **Faster** than `iterrows()` due to named tuples.

---

### Q83. Iterate over columns.
> 💡 **Hint:** `for col in df.columns:` then access `df[col]`. Or use `df.items()` which yields `(col_name, Series)` pairs.

---

### Q84. Avoid iteration using vectorization.
> 💡 **Hint:** Replace `for` loops with direct column operations: `df["new"] = df["a"] * 2 + 1` instead of a loop over rows.

---

### Q85. Group data using `groupby()`.
> 💡 **Hint:** `df.groupby("col")["value_col"].mean()` — specify the grouping key and the aggregation in one line.

---

### Q86. Calculate sum per group.
> 💡 **Hint:** `df.groupby("col")["value"].sum()` — replace `sum` with any aggregation function.

---

### Q87. Calculate multiple aggregations.
> 💡 **Hint:** `df.groupby("col").agg({"col1": "mean", "col2": "sum"})` or `df.groupby("col").agg(name=("col","func"))` for named aggregation.

---

### Q88. Group by multiple columns.
> 💡 **Hint:** `df.groupby(["col1","col2"])["value"].mean()` — pass a list of column names.

---

### Q89. Filter grouped data.
> 💡 **Hint:** `df.groupby("col").filter(lambda g: g["value"].mean() > threshold)` — keeps entire groups satisfying the condition.

---

### Q90. Transform grouped data using `transform()`.
> 💡 **Hint:** `df.groupby("col")["value"].transform("mean")` — returns a Series **same size as the original DataFrame** filled with group statistics.

---

### Q91. Apply custom function in groupby.
> 💡 **Hint:** `df.groupby("col")["value"].agg(lambda s: s.max() - s.min())` — pass any function that takes a Series and returns a scalar.

---

### Q92. Create basic pivot table.
> 💡 **Hint:** `pd.pivot_table(df, values="val", index="row_col", columns="col_col", aggfunc="mean")`.

---

### Q93. Use multiple aggregation functions in pivot.
> 💡 **Hint:** Pass a list to `aggfunc`: `aggfunc=["mean","max"]` — the result will have a MultiLevel column header.

---

### Q94. Handle missing values in pivot tables.
> 💡 **Hint:** Use `fill_value=0` in `pivot_table()` to replace NaN with a specific value.

---

### Q95. Add margins (totals) in pivot tables.
> 💡 **Hint:** `margins=True, margins_name="TOTAL"` in `pivot_table()` adds a summary row and column.

---

### Q96. Create multi-level pivot table.
> 💡 **Hint:** Pass lists to `index` and `columns` parameters: `index=["col1","col2"], columns=["col3"]`.

---

### Q97. Convert DataFrame to pivot format.
> 💡 **Hint:** `df.pivot(index="row_key", columns="col_key", values="value")` — requires **unique** combinations of row/column keys (no aggregation).

---

### Q98. Compare pivot vs groupby.
> 💡 **Hint:** `groupby()` is more flexible and returns a flat structure. `pivot_table()` gives a 2D grid and handles duplicates with aggregation. `pivot()` does **no aggregation** and fails on duplicates.

---

### Q99. Build summary report using pivot table.
> 💡 **Hint:** Combine `pivot_table()` with `margins=True` and `fill_value=0`. Chain `.round(2)` for presentation. Export with `.to_csv()` or `.to_excel()`.

---

### Q100. Create an end-to-end analysis workflow.
> 💡 **Hint:** Follow this pipeline:  
> 1. Load data → 2. Inspect (shape, dtypes, nulls) → 3. Clean (fillna, drop_duplicates) → 4. Feature engineer → 5. GroupBy/pivot analysis → 6. Sort and rank → 7. Export results.

---

*📌 Remember: Try solving without hints first. Look at hints only when genuinely stuck.*  
*⭐ Happy Learning Pandas!*
