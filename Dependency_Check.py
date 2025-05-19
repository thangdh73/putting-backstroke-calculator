import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests

# Streamlit App
st.title("Dependency Check Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())

    # Parameter selection
    numerical_columns = list(data.select_dtypes(include=[np.number]).columns)
    target_column = st.selectbox("Select Dependent (Target) Variable:", ["None"] + numerical_columns)
    feature_columns = st.multiselect("Select Features (Independent Variables):", numerical_columns)

    if feature_columns:
        X = data[feature_columns]
        if target_column != "None":
            y = data[target_column]

        dependency_summary = []

        # ================================
        # 1. Correlation Analysis (Pearson)
        # ================================
        st.write("## 1. Correlation Analysis (Pearson)")
        correlation_matrix = X.corr()
        st.write(correlation_matrix)
        st.bar_chart(correlation_matrix.abs())
        # Add summary based on correlation
        for col in feature_columns:
            high_corr = correlation_matrix[col][correlation_matrix[col] > 0.8].index.tolist()
            if len(high_corr) > 1:
                dependency_summary.append(f"'{col}' is highly correlated with {', '.join([x for x in high_corr if x != col])}.")

        # ================================
        # 2. Variance Inflation Factor (VIF)
        # ================================
        st.write("## 2. Variance Inflation Factor (VIF)")
        X['Intercept'] = 1  # Add intercept for VIF calculation
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns[:-1]
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1] - 1)]
        st.write(vif_data)
        X = X.drop(columns=['Intercept'])  # Clean up after VIF
        # Add summary based on VIF
        for i, row in vif_data.iterrows():
            if row['VIF'] > 10:
                dependency_summary.append(f"'{row['Feature']}' has a high VIF ({row['VIF']}), indicating multicollinearity.")

        # ================================
        # 3. Chi-Square Test (For Categorical Data)
        # ================================
        st.write("## 3. Chi-Square Test (Categorical Variables)")
        if target_column != "None":
            contingency_table = pd.crosstab(pd.qcut(data[feature_columns[0]], q=4), data[target_column])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            st.write(f"Chi-Square Statistic: {chi2}, p-value: {p}")
            if p < 0.05:
                dependency_summary.append(f"'{feature_columns[0]}' is significantly associated with '{target_column}' (p-value: {p}).")

        # ================================
        # 4. Mutual Information
        # ================================
        st.write("## 4. Mutual Information")
        if target_column != "None":
            mi = mutual_info_regression(X, y)
            mi_results = pd.DataFrame({"Feature": X.columns, "Mutual Information": mi})
            st.write(mi_results)
            # Add summary based on Mutual Information
            for i, row in mi_results.iterrows():
                if row['Mutual Information'] > 0.1:
                    dependency_summary.append(f"'{row['Feature']}' shares significant information with '{target_column}' (MI: {row['Mutual Information']}).")

        # ================================
        # 5. PCA (Principal Component Analysis)
        # ================================
        st.write("## 5. Principal Component Analysis (PCA)")
        pca = PCA()
        pca.fit(X)
        explained_variance = pd.DataFrame({
            "Component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "Explained Variance Ratio": pca.explained_variance_ratio_,
        })
        st.write(explained_variance)

        # ================================
        # 6. Regression Analysis (R² and Coefficients)
        # ================================
        st.write("## 6. Regression Analysis")
        if target_column != "None":
            reg = LinearRegression()
            reg.fit(X, y)
            r_squared = reg.score(X, y)
            coeffs = pd.DataFrame({"Feature": X.columns, "Coefficient": reg.coef_})
            st.write(f"R²: {r_squared}")
            st.write(coeffs)

        # ================================
        # 7. Granger Causality Test (Time Series)
        # ================================
        st.write("## 7. Granger Causality Test (Time Series)")
        if len(feature_columns) >= 2:
            try:
                granger_results = grangercausalitytests(data[feature_columns[:2]], maxlag=2, verbose=False)
                for lag, test in granger_results.items():
                    st.write(f"Lag {lag}: p-value = {test[0]['ssr_ftest'][1]}")
            except:
                st.write("Granger Causality requires valid time-series data.")

        # ================================
        # 8. Cross-Correlation (Optional for Time-Series)
        # ================================
        st.write("## 8. Cross-Correlation (Time-Series)")
        if len(feature_columns) >= 2:
            cross_corr = np.corrcoef(data[feature_columns[0]], data[feature_columns[1]])
            st.write(f"Cross-Correlation between {feature_columns[0]} and {feature_columns[1]}: {cross_corr[0, 1]}")

# ================================
# Summary of Dependency Analysis
# ================================
st.write("## Summary of Dependency Analysis")
if dependency_summary:
    st.write("### Findings:")
    for item in dependency_summary:
        st.write(f"- {item}")

    # Provide a conclusion about dependency for Monte Carlo simulation
    st.write("### Conclusion for Monte Carlo Simulation:")
    dependent_features = [row['Feature'] for i, row in vif_data.iterrows() if row['VIF'] > 10]
    if dependent_features:
        st.write(f"The following variables are highly dependent: {', '.join(dependent_features)}.")
        st.write("**Recommendation**: You need to set dependency among these variables during the Monte Carlo simulation.")
        st.write("For example, use statistical relationships (e.g., correlation matrix or regression equations) to ensure realistic dependencies between these variables.")
    else:
        st.write("The selected variables show no significant dependency. You can run the Monte Carlo simulation assuming they are independent.")
else:
    st.write("No significant dependencies or multicollinearity detected.")
    st.write("The selected variables can be treated as independent during the Monte Carlo simulation.")
