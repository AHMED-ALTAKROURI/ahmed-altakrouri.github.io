import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def feature_importance_plot(importances, feature_labels, ax=None):
    importa = pd.DataFrame({"Importance": importances,
                            "Feature": feature_labels})

    importa.sort_values("Importance", inplace=True, ascending=False)
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    sns.barplot(x="Importance", y="Feature", ax=axis, data=importa)
    axis.set_title('Feature Importance Measures')
    plt.close()

    return axis if ax else fig


def coefficient_plot(coefficients, features, ax=None):
    c = pd.DataFrame({"coefficients": coefficients, "Feature": features}).sort_values("coefficients", ascending=False)
    fig, axis = (None, ax) if ax else plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    sns.barplot(x="coefficients", y="Feature", ax=axis, data=c)
    axis.set_title('Linear Regression Coefficients')
    plt.close()

    return axis if ax else fig


def get_regressor_coefficients(model, columns):
    coefficients = {}
    for coefficient, feature in zip(model.coef_, columns):
        coefficients[feature] = coefficient

    return coefficients


def calculate_vif(x):
    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif


def prepare_out_of_sample_data(golden_data):
    # categorize your variables here:
    golden_data.set_index("index", inplace=True)
    golden_data.drop(["Standard of living",
                      "Coefficient bonus malus",
                      "CRM score",
                      "Yearly maintenance cost"], inplace=True, axis=1)
    categories = ['Brand', 'Vehicle type', 'Socioeconomic category']
    golden_data = pd.get_dummies(golden_data, columns=categories, drop_first=True)
    golden_data.dropna(inplace=True)
    golden_data["Brand_other"] = 0
    return golden_data


def get_dataframe_from_summary(est):
    results_summary = est.summary()
    results_as_html = results_summary.tables[1].as_html()
    return pd.read_html(results_as_html, header=0, index_col=0)[0]


def save_model(model, path):
    joblib.dump(model, path)
    print("model saved....")

# %%
