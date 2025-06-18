import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests

# --------------------------------------------------
# Streamlit app header
# --------------------------------------------------
st.title("Dependency Check Analysis")

# Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Initialise summary list so it ALWAYS exists
dependency_summary: list[str] = []

if uploaded_file:
    # --------------------------------------------------
    # 1.  Load data and let user pick variables
    # --------------------------------------------------
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())

    numerical_columns = list(data.select_dtypes(include=[np.number]).columns)
    target_column = st.selectbox(
        "Select Dependent (Target) Variable:", ["None"] + numerical_columns
    )
    feature_columns = st.multiselect(
        "Select Features (Independent Variables):", numerical_columns
    )

    # --------------------------------------------------
    # 2.  Run analyses only if the user picked features
    # --------------------------------------------------
    if feature_columns:
        X = data[feature_columns].copy()          # .copy() to avoid pandas warn
        if target_column != "None":
            y = data[target_column]

        # ========== 1. Correlation (Pearson) ==========
        st.subheader("1. Correlation Analysis (Pearson)")
        corr = X.corr()
        st.dataframe(corr)
        st.bar_chart(corr.abs())

        for col in feature_columns:
            high_corr = corr[col][(corr[col] > 0.8) & (corr[col] < 1)].index
            if len(high_corr):
                dependency_summary.append(
                    f"'{col}' is highly correlated with {', '.join(high_corr)}."
                )

        # ========== 2. Variance Inflation Factor ======
        st.subheader("2. Variance Inflation Factor (VIF)")
        X_with_const = X.assign(Intercept=1)
        vif_vals = [
            variance_inflation_factor(X_with_const.values, i)
            for i in range(X_with_const.shape[1] - 1)
        ]
        vif_df = pd.DataFrame({"Feature": X.columns, "VIF": vif_vals})
        st.dataframe(vif_df)

        for _, row in vif_df.iterrows():
            if row.VIF > 10:
                dependency_summary.append(
                    f"'{row.Feature}' has high multicollinearity (VIF {row.VIF:.1f})."
                )

        # ========== 3. Chi-square (categorical demo) ==
        st.subheader("3. Chi-Square Test (using first feature binned)")
        if target_column != "None":
            try:
                table = pd.crosstab(
                    pd.qcut(data[feature_columns[0]], q=4),
                    data[target_column]
                )
                chi2, p, _, _ = chi2_contingency(table)
                st.write(f"Chi² = {chi2:.2f}, p-value = {p:.4f}")
                if p < 0.05:
                    dependency_summary.append(
                        f"'{feature_columns[0]}' is significantly associated with "
                        f"'{target_column}' (p = {p:.4f})."
                    )
            except Exception:
                st.info("Chi-square skipped (requires discrete target).")

        # ========== 4. Mutual information =============
        st.subheader("4. Mutual Information")
        if target_column != "None":
            mi = mutual_info_regression(X, y)
            mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Info": mi})
            st.dataframe(mi_df)
            for _, row in mi_df.iterrows():
                if row["Mutual Info"] > 0.1:
                    dependency_summary.append(
                        f"'{row.Feature}' shares notable information "
                        f"with '{target_column}' (MI {row['Mutual Info']:.3f})."
                    )

        # ========== 5. PCA ============================
        st.subheader("5. Principal Component Analysis (PCA)")
        pca = PCA()
        pca.fit(X)
        pca_df = pd.DataFrame(
            {"Component": range(1, len(pca.explained_variance_ratio_) + 1),
             "Explained Var %": pca.explained_variance_ratio_})
        st.dataframe(pca_df)

        # ========== 6. Linear regression ==============
        st.subheader("6. Regression Fit (R²)")
        if target_column != "None":
            reg = LinearRegression().fit(X, y)
            st.write(f"R² = {reg.score(X, y):.3f}")
            coef_df = pd.DataFrame({"Feature": X.columns, "Coeff": reg.coef_})
            st.dataframe(coef_df)

        # ========== 7. Granger causality ==============
        st.subheader("7. Granger Causality (demo: first two features)")
        if len(feature_columns) >= 2:
            try:
                g_res = grangercausalitytests(
                    data[feature_columns[:2]], maxlag=2, verbose=False
                )
                for lag, res in g_res.items():
                    p_val = res[0]['ssr_ftest'][1]
                    st.write(f"Lag {lag}: p = {p_val:.4f}")
            except Exception:
                st.info("Granger test needs proper time-series data.")

        # ========== 8. Cross-correlation (simple) =====
        st.subheader("8. Cross-Correlation")
        if len(feature_columns) >= 2:
            cmat = np.corrcoef(data[feature_columns[0]], data[feature_columns[1]])
            st.write(f"{feature_columns[0]} ↔ {feature_columns[1]}: "
                     f"{cmat[0,1]:.3f}")

        # --------------------------------------------------
        # Summary block
        # --------------------------------------------------
        st.subheader("Summary of Dependency Analysis")

        if dependency_summary:
            st.markdown("**Findings:**")
            for item in dependency_summary:
                st.write(f"• {item}")

            vif_problem = [row.Feature for _, row in vif_df.iterrows()
                           if row.VIF > 10]

            if vif_problem:
                st.markdown("**Conclusion for Monte-Carlo:**")
                st.write(
                    f"Variables with high dependency: {', '.join(vif_problem)}."
                )
                st.write(
                    "Set suitable correlations or regression equations when "
                    "simulating these variables."
                )
            else:
                st.write(
                    "No major multicollinearity; variables may be treated as "
                    "independent in a Monte-Carlo simulation."
                )
        else:
            st.write("No strong dependencies detected.")

else:
    st.info("Upload a CSV file to begin.")
