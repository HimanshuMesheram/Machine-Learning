import json

# Complete Random Forest notebook content
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Random Forest: Complete Guide\n",
    "\n",
    "Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model.\n",
    "\n",
    "## Topics Covered:\n",
    "1. **What is Random Forest?** - Core concepts and intuition\n",
    "2. **How Random Forest Works** - Step-by-step explanation\n",
    "3. **Key Parameters** - Important hyperparameters\n",
    "4. **Feature Importance** - Understanding feature contributions\n",
    "5. **Implementation on Diabetes Dataset** - Complete example\n",
    "6. **Model Evaluation** - Performance metrics\n",
    "7. **Hyperparameter Tuning** - Grid search and cross-validation\n",
    "8. **Comparison with Decision Tree** - Why ensemble works better\n",
    "9. **Advantages and Disadvantages**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,\n",
    "                             roc_auc_score, roc_curve, precision_score, recall_score,\n",
    "                             f1_score)\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Set random seed for reproducibility\n",
    "np.random.seed(42)\n",
    "\n",
    "# Plotting style\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. What is Random Forest?\n",
    "\n",
    "### Definition\n",
    "Random Forest is an ensemble learning method that:\n",
    "- **Builds multiple decision trees** on different subsets of data\n",
    "- **Averages predictions** from all trees (for regression) or uses voting (for classification)\n",
    "- **Reduces overfitting** by introducing randomness\n",
    "\n",
    "### Key Concepts:\n",
    "\n",
    "1. **Bootstrap Aggregation (Bagging)**:\n",
    "   - Creates multiple subsets of the training data by sampling with replacement\n",
    "   - Each tree is trained on a different bootstrap sample\n",
    "\n",
    "2. **Random Feature Selection**:\n",
    "   - Each node in a tree considers only a random subset of features\n",
    "   - Reduces correlation between trees\n",
    "\n",
    "3. **Ensemble Voting**:\n",
    "   - For classification: majority voting\n",
    "   - For regression: average of predictions\n",
    "\n",
    "### Intuition\n",
    "\"The wisdom of the crowd\" - multiple trees working together make better decisions than a single tree.\n",
    "\n",
    "![Random Forest](https://miro.medium.com/v2/resize:fit:1400/1*0YFBWJZnt0_vuy2z6hwCQQ.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Load and Explore Diabetes Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_csv('diabetes_dataset.csv')\n",
    "\n",
    "print(\"=== Diabetes Dataset Overview ===\")\n",
    "print(f\"Dataset shape: {df.shape}\")\n",
    "print(f\"Features: {df.columns.tolist()}\")\n",
    "print()\n",
    "\n",
    "# Display first few rows\n",
    "print(\"First 5 rows of the dataset:\")\n",
    "display(df.head())\n",
    "print()\n",
    "\n",
    "# Dataset information\n",
    "print(\"Dataset Info:\")\n",
    "print(df.info())\n",
    "print()\n",
    "\n",
    "# Statistical summary\n",
    "print(\"Statistical Summary:\")\n",
    "display(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values\n",
    "print(\"Missing Values:\")\n",
    "missing = df.isnull().sum()\n",
    "print(missing[missing > 0] if any(missing > 0) else \"No missing values found!\")\n",
    "print()\n",
    "\n",
    "# Check for zero values in features (possible missing data indicator)\n",
    "print(\"Features with Zero Values (possible missing values):\")\n",
    "features_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n",
    "for feature in features_with_zero:\n",
    "    zero_count = (df[feature] == 0).sum()\n",
    "    if zero_count > 0:\n",
    "        print(f\"{feature}: {zero_count} zero values ({zero_count/len(df)*100:.1f}%)\")\n",
    "print()\n",
    "\n",
    "# Target variable distribution\n",
    "print(\"Target Variable Distribution:\")\n",
    "outcome_counts = df['Outcome'].value_counts()\n",
    "print(f\"Class 0 (No Diabetes): {outcome_counts[0]} samples ({outcome_counts[0]/len(df)*100:.1f}%)\")\n",
    "print(f\"Class 1 (Diabetes): {outcome_counts[1]} samples ({outcome_counts[1]/len(df)*100:.1f}%)\")\n",
    "\n",
    "# Visualize target distribution\n",
    "fig, ax = plt.subplots(1, 2, figsize=(12, 5))\n",
    "colors = ['lightblue', 'lightcoral']\n",
    "ax[0].pie(outcome_counts.values, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', colors=colors, startangle=90)\n",
    "ax[0].set_title('Diabetes Outcome Distribution')\n",
    "ax[1].bar(['No Diabetes', 'Diabetes'], outcome_counts.values, color=colors)\n",
    "ax[1].set_title('Count of Diabetes Cases')\n",
    "ax[1].set_ylabel('Count')\n",
    "for i, v in enumerate(outcome_counts.values):\n",
    "    ax[1].text(i, v + 5, str(v), ha='center')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Handle zero values in features (replace with median)\n",
    "df_clean = df.copy()\n",
    "\n",
    "print(\"=== Handling Zero Values ===\")\n",
    "for feature in features_with_zero:\n",
    "    zero_mask = df_clean[feature] == 0\n",
    "    zero_count = zero_mask.sum()\n",
    "    if zero_count > 0:\n",
    "        median_value = df_clean[df_clean[feature] != 0][feature].median()\n",
    "        df_clean.loc[zero_mask, feature] = median_value\n",
    "        print(f\"{feature}: Replaced {zero_count} zeros with median={median_value:.1f}\")\n",
    "print()\n",
    "\n",
    "# Separate features and target\n",
    "X = df_clean.drop('Outcome', axis=1)\n",
    "y = df_clean['Outcome']\n",
    "feature_names = X.columns.tolist()\n",
    "\n",
    "print(f\"Features: {feature_names}\")\n",
    "print(f\"Target: Outcome\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data into train and test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "print(f\"Training set size: {X_train.shape[0]} samples\")\n",
    "print(f\"Test set size: {X_test.shape[0]} samples\")\n",
    "print()\n",
    "print(\"Training set class distribution:\")\n",
    "print(y_train.value_counts())\n",
    "print()\n",
    "print(\"Test set class distribution:\")\n",
    "print(y_test.value_counts())\n",
    "\n",
    "# Feature scaling (though Random Forest doesn't require scaling)\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Random Forest: Core Concepts\n",
    "\n",
    "### How Random Forest Works:\n",
    "\n",
    "1. **Step 1**: Randomly select 'n' samples from the training data (bootstrap sampling)\n",
    "2. **Step 2**: For each bootstrap sample, build a decision tree\n",
    "3. **Step 3**: At each node, randomly select 'm' features from all features\n",
    "4. **Step 4**: Choose the best split among the selected features\n",
    "5. **Step 5**: Repeat steps 2-4 to create multiple trees\n",
    "6. **Step 6**: For prediction, use majority voting (classification) or averaging (regression)\n",
    "\n",
    "### Key Parameters:\n",
    "\n",
    "- **n_estimators**: Number of trees in the forest\n",
    "- **max_depth**: Maximum depth of each tree\n",
    "- **min_samples_split**: Minimum samples required to split a node\n",
    "- **min_samples_leaf**: Minimum samples required at a leaf node\n",
    "- **max_features**: Number of features to consider for best split\n",
    "- **bootstrap**: Whether to use bootstrap sampling\n",
    "- **oob_score**: Whether to use out-of-bag samples for validation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Training Random Forest Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize Random Forest Classifier\n",
    "rf_model = RandomForestClassifier(\n",
    "    n_estimators=100,        # Number of trees\n",
    "    max_depth=10,            # Maximum depth of each tree\n",
    "    min_samples_split=5,     # Minimum samples to split a node\n",
    "    min_samples_leaf=2,      # Minimum samples at leaf node\n",
    "    max_features='sqrt',     # Number of features for best split\n",
    "    bootstrap=True,          # Use bootstrap sampling\n",
    "    oob_score=True,          # Use out-of-bag samples for validation\n",
    "    random_state=42,         # For reproducibility\n",
    "    n_jobs=-1                # Use all CPU cores\n",
    ")\n",
    "\n",
    "# Train the model\n",
    "rf_model.fit(X_train, y_train)\n",
    "\n",
    "print(\"=== Random Forest Model Training Complete ===\")\n",
    "print(f\"Number of trees: {rf_model.n_estimators}\")\n",
    "print(f\"Number of features: {rf_model.n_features_in_}\")\n",
    "print(f\"Out-of-bag score: {rf_model.oob_score_:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Model Evaluation\n",
    "\n",
    "### 7.1 Predictions and Basic Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions\n",
    "y_pred = rf_model.predict(X_test)\n",
    "y_pred_proba = rf_model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "# Calculate metrics\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "precision = precision_score(y_test, y_pred)\n",
    "recall = recall_score(y_test, y_pred)\n",
    "f1 = f1_score(y_test, y_pred)\n",
    "roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
    "\n",
    "print(\"=== Random Forest Model Performance ===\")\n",
    "print(f\"Accuracy:  {accuracy:.4f}\")\n",
    "print(f\"Precision: {precision:.4f}\")\n",
    "print(f\"Recall:    {recall:.4f}\")\n",
    "print(f\"F1 Score:  {f1:.4f}\")\n",
    "print(f\"ROC-AUC:   {roc_auc:.4f}\")\n",
    "print()\n",
    "\n",
    "print(\"Classification Report:\")\n",
    "print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.2 Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Confusion Matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
    "            xticklabels=['No Diabetes', 'Diabetes'],\n",
    "            yticklabels=['No Diabetes', 'Diabetes'])\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.title('Confusion Matrix - Random Forest')\n",
    "plt.show()\n",
    "\n",
    "print(\"Confusion Matrix Interpretation:\")\n",
    "print(f\"True Negatives:  {cm[0,0]} (Correctly predicted No Diabetes)\")\n",
    "print(f\"False Positives: {cm[0,1]} (Incorrectly predicted Diabetes)\")\n",
    "print(f\"False Negatives: {cm[1,0]} (Missed Diabetes cases)\")\n",
    "print(f\"True Positives:  {cm[1,1]} (Correctly predicted Diabetes)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.3 ROC Curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ROC Curve\n",
    "fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "ax.plot(fpr, tpr, color='darkorange', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')\n",
    "ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2, label='Random Classifier')\n",
    "ax.set_xlabel('False Positive Rate')\n",
    "ax.set_ylabel('True Positive Rate')\n",
    "ax.set_title('ROC Curve - Random Forest')\n",
    "ax.legend(loc='lower right')\n",
    "ax.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get feature importance\n",
    "feature_importance = pd.DataFrame({\n",
    "    'Feature': feature_names,\n",
    "    'Importance': rf_model.feature_importances_\n",
    "}).sort_values('Importance', ascending=False)\n",
    "\n",
    "print(\"=== Feature Importance ===\")\n",
    "print(feature_importance.to_string(index=False))\n",
    "print()\n",
    "\n",
    "# Visualize feature importance\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))\n",
    "ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)\n",
    "ax.set_xlabel('Importance')\n",
    "ax.set_ylabel('Features')\n",
    "ax.set_title('Feature Importance in Random Forest')\n",
    "ax.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Top features explanation\n",
    "print(\"Top 5 Most Important Features:\")\n",
    "for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:5], \n",
    "                                              feature_importance['Importance'][:5]), 1):\n",
    "    print(f\"{i}. {feature}: {importance:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Comparison with Single Decision Tree\n",
    "\n",
    "Comparing Random Forest with a single Decision Tree to demonstrate the power of ensembling."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train a single Decision Tree\n",
    "dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)\n",
    "dt_model.fit(X_train, y_train)\n",
    "dt_pred = dt_model.predict(X_test)\n",
    "dt_accuracy = accuracy_score(y_test, dt_pred)\n",
    "\n",
    "# Random Forest metrics\n",
    "rf_accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "# Cross-validation scores\n",
    "dt_cv = cross_val_score(dt_model, X_train, y_train, cv=5).mean()\n",
    "rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5).mean()\n",
    "\n",
    "print(\"=== Model Comparison: Decision Tree vs Random Forest ===\")\n",
    "comparison_df = pd.DataFrame({\n",
    "    'Model': ['Decision Tree', 'Random Forest'],\n",
    "    'Test Accuracy': [dt_accuracy, rf_accuracy],\n",
    "    'CV Score': [dt_cv, rf_cv],\n",
    "    'Improvement': ['-', f\"{(rf_accuracy - dt_accuracy)*100:.1f}%\"]\n",
    "})\n",
    "print(comparison_df.to_string(index=False))\n",
    "print()\n",
    "\n",
    "# Visual comparison\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "models = ['Decision Tree', 'Random Forest']\n",
    "test_scores = [dt_accuracy, rf_accuracy]\n",
    "cv_scores = [dt_cv, rf_cv]\n",
    "\n",
    "x = np.arange(len(models))\n",
    "width = 0.35\n",
    "\n",
    "ax.bar(x - width/2, test_scores, width, label='Test Accuracy', color=['lightblue', 'lightcoral'])\n",
    "ax.bar(x + width/2, cv_scores, width, label='CV Score', color=['darkblue', 'darkred'])\n",
    "ax.set_xlabel('Model')\n",
    "ax.set_ylabel('Score')\n",
    "ax.set_title('Decision Tree vs Random Forest')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(models)\n",
    "ax.legend()\n",
    "ax.grid(True, alpha=0.3)\n",
    "\n",
    "for i, (test, cv) in enumerate(zip(test_scores, cv_scores)):\n",
    "    ax.text(i - width/2, test + 0.01, f'{test:.3f}', ha='center')\n",
    "    ax.text(i + width/2, cv + 0.01, f'{cv:.3f}', ha='center')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Hyperparameter Tuning with Grid Search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Parameter grid for tuning\n",
    "param_grid = {\n",
    "    'n_estimators': [50, 100, 200],\n",
    "    'max_depth': [5, 10, 15, None],\n",
    "    'min_samples_split': [2, 5, 10],\n",
    "    'min_samples_leaf': [1, 2, 4],\n",
    "    'max_features': ['sqrt', 'log2', None]\n",
    "}\n",
    "\n",
    "print(\"=== Hyperparameter Tuning ===\")\n",
    "print(f\"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations\")\n",
    "print(\"Performing Grid Search (this may take a few minutes)...\")\n",
    "\n",
    "# Grid search\n",
    "rf_grid = RandomForestClassifier(random_state=42, oob_score=True)\n",
    "grid_search = GridSearchCV(rf_grid, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "print(\"\\n=== Best Parameters ===\")\n",
    "for param, value in grid_search.best_params_.items():\n",
    "    print(f\"{param}: {value}\")\n",
    "\n",
    "print(f\"\\nBest Cross-validation Score: {grid_search.best_score_:.4f}\")\n",
    "print(f\"Test Set Score: {grid_search.score(X_test, y_test):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train final model with best parameters\n",
    "best_rf = grid_search.best_estimator_\n",
    "y_pred_best = best_rf.predict(X_test)\n",
    "\n",
    "# Compare with default model\n",
    "default_acc = accuracy_score(y_test, y_pred)\n",
    "tuned_acc = accuracy_score(y_test, y_pred_best)\n",
    "\n",
    "print(\"=== Default vs Tuned Model ===\")\n",
    "print(f\"Default Model Accuracy: {default_acc:.4f}\")\n",
    "print(f\"Tuned Model Accuracy: {tuned_acc:.4f}\")\n",
    "print(f\"Improvement: {(tuned_acc - default_acc)*100:.2f}%\")\n",
    "print()\n",
    "\n",
    "print(\"Tuned Model Performance:\")\n",
    "print(classification_report(y_test, y_pred_best, target_names=['No Diabetes', 'Diabetes']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Cross-Validation Scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Perform 5-fold cross-validation\n",
    "cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)\n",
    "\n",
    "print(\"=== Cross-Validation Results ===\")\n",
    "print(f\"CV Scores: {cv_scores}\")\n",
    "print(f\"Mean CV Score: {cv_scores.mean():.4f}\")\n",
    "print(f\"Std CV Score: {cv_scores.std():.4f}\")\n",
    "print(f\"95% Confidence Interval: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]\")\n",
    "\n",
    "# Visualize CV scores\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "ax.bar(range(1, 6), cv_scores, color='lightgreen', alpha=0.7)\n",
    "ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean = {cv_scores.mean():.3f}')\n",
    "ax.fill_between([0.5, 5.5], cv_scores.mean() - cv_scores.std(), \n",
    "                cv_scores.mean() + cv_scores.std(), alpha=0.2, color='gray', \n",
    "                label=f'±1 Std = ±{cv_scores.std():.3f}')\n",
    "ax.set_xlabel('Fold')\n",
    "ax.set_ylabel('Accuracy')\n",
    "ax.set_title('5-Fold Cross-Validation Scores')\n",
    "ax.set_xticks(range(1, 6))\n",
    "ax.legend()\n",
    "ax.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Out-of-Bag (OOB) Error Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare OOB error vs test error\n",
    "oob_error = 1 - best_rf.oob_score_\n",
    "test_error = 1 - accuracy_score(y_test, y_pred_best)\n",
    "\n",
    "print(\"=== Out-of-Bag Error Analysis ===\")\n",
    "print(f\"OOB Score: {best_rf.oob_score_:.4f}\")\n",
    "print(f\"OOB Error: {oob_error:.4f}\")\n",
    "print(f\"Test Set Error: {test_error:.4f}\")\n",
    "print(f\"Difference: {abs(oob_error - test_error):.4f}\")\n",
    "print()\n",
    "\n",
    "if abs(oob_error - test_error) < 0.05:\n",
    "    print(\"✅ OOB error is close to test error - model is well-calibrated\")\n",
    "else:\n",
    "    print(\"⚠️ OOB error differs from test error - check model performance\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 13. Visualization of Random Forest Structure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import plot_tree\n",
    "\n",
    "# Visualize one tree from the forest\n",
    "fig, ax = plt.subplots(figsize=(20, 10))\n",
    "plot_tree(best_rf.estimators_[0], \n",
    "          feature_names=feature_names,\n",
    "          class_names=['No Diabetes', 'Diabetes'],\n",
    "          filled=True, \n",
    "          rounded=True,\n",
    "          max_depth=3,\n",
    "          fontsize=8)\n",
    "plt.title('Example Tree from Random Forest (First Tree, Depth=3)')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 14. Feature Interactions Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze relationships between top features and target\n",
    "top_features = feature_importance['Feature'][:3].tolist()\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "\n",
    "for idx, feature in enumerate(top_features):\n",
    "    ax = axes[idx]\n",
    "    \n",
    "    # Boxplot by outcome\n",
    "    data_to_plot = [df_clean[df_clean['Outcome'] == 0][feature], \n",
    "                    df_clean[df_clean['Outcome'] == 1][feature]]\n",
    "    bp = ax.boxplot(data_to_plot, labels=['No Diabetes', 'Diabetes'], patch_artist=True)\n",
    "    bp['boxes'][0].set_facecolor('lightblue')\n",
    "    bp['boxes'][1].set_facecolor('lightcoral')\n",
    "    ax.set_ylabel(feature)\n",
    "    ax.set_title(f'{feature} by Diabetes Outcome')\n",
    "    ax.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 15. Advantages and Disadvantages of Random Forest\n",
    "\n",
    "### Advantages:\n",
    "\n",
    "1. **High Accuracy**: Generally outperforms single decision trees\n",
    "2. **Robust to Overfitting**: Ensemble averaging reduces overfitting\n",
    "3. **Handles High Dimensional Data**: Works well with many features\n",
    "4. **Feature Importance**: Built-in feature ranking\n",
    "5. **Handles Missing Data**: Can handle missing values well\n",
    "6. **Parallelizable**: Each tree can be built independently\n",
    "7. **Non-linear Relationships**: Captures complex patterns\n",
    "8. **Imbalanced Data**: Can handle class imbalance with class_weight\n",
    "\n",
    "### Disadvantages:\n",
    "\n",
    "1. **Less Interpretable**: Not as interpretable as single decision tree\n",
    "2. **Memory Intensive**: Requires more memory for large forests\n",
    "3. **Computationally Expensive**: Slower than simpler models\n",
    "4. **Not Ideal for Online Learning**: Hard to update with new data\n",
    "5. **Black Box**: Harder to explain predictions\n",
    "\n",
    "### Best Practices:\n",
    "\n",
    "1. **n_estimators**: Use more trees for better performance (100-500)\n",
    "2. **max_features**: Use 'sqrt' for classification, 'log2' for regression\n",
    "3. **max_depth**: Tune to prevent overfitting\n",
    "4. **min_samples_split**: Start with 2, increase to reduce overfitting\n",
    "5. **OOB Score**: Use for validation without cross-validation\n",
    "6. **Feature Scaling**: Not required (tree-based models are scale-invariant)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 16. Summary and Key Takeaways"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"=\" * 60)\n",
    "print(\"RANDOM FOREST - SUMMARY\")\n",
    "print(\"=\" * 60)\n",
    "print()\n",
    "print(f\"Dataset: Diabetes Dataset ({len(df)} samples, {len(feature_names)} features)\")\n",
    "print(f\"Model: Random Forest Classifier with {best_rf.n_estimators} trees\")\n",
    "print()\n",
    "print(\"PERFORMANCE METRICS:\")\n",
    "print(f\"  • Test Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}\")\n",
    "print(f\"  • Precision:      {precision_score(y_test, y_pred_best):.4f}\")\n",
    "print(f\"  • Recall:         {recall_score(y_test, y_pred_best):.4f}\")\n",
    "print(f\"  • F1 Score:       {f1_score(y_test, y_pred_best):.4f}\")\n",
    "print(f\"  • ROC-AUC:        {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]):.4f}\")\n",
    "print()\n",
    "print(\"TOP 5 IMPORTANT FEATURES:\")\n",
    "for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:5], \n",
    "                                              feature_importance['Importance'][:5]), 1):\n",
    "    print(f\"  {i}. {feature}: {importance:.4f}\")\n",
    "print()\n",
    "print(\"=\" * 60)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Final Notes\n",
    "\n",
    "### When to Use Random Forest:\n",
    "\n",
    "✅ **Use Random Forest when:**\n",
    "- You need high accuracy\n",
    "- Data has many features\n",
    "- There are complex non-linear relationships\n",
    "- You need feature importance scores\n",
    "- Interpretability is not critical\n",
    "\n",
    "❌ **Avoid Random Forest when:**\n",
    "- Interpretability is crucial (use Decision Tree instead)\n",
    "- Data is very small (< 100 samples)\n",
    "- You need real-time predictions\n",
    "- Memory is limited\n",
    "- You need simple, linear relationships\n",
    "\n",
    "### Next Steps for Improvement:\n",
    "\n",
    "1. **Feature Engineering**: Create new features from existing ones\n",
    "2. **Class Imbalance**: Use class_weight or SMOTE\n",
    "3. **Ensemble Methods**: Try Gradient Boosting or XGBoost\n",
    "4. **Feature Selection**: Remove irrelevant features\n",
    "5. **More Tuning**: Use Bayesian optimization for hyperparameters"
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
with open('Random_Forest_Diabetes_Complete_Guide.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Random_Forest_Diabetes_Complete_Guide.ipynb has been created successfully!")
print("You can now open this file in Jupyter Notebook or Jupyter Lab.")