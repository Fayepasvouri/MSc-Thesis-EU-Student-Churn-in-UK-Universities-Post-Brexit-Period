"""
Faye

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objects as go

# read the CSV file that has been saved as utf=8
df = pd.read_csv("C:/Users/Faye/Desktop/Master/data_thesis.csv")
print(df)

print(df.shape)

print(df.dtypes)
print(df.nunique())
print(df.isnull().any())

print(df.select_dtypes("object").head())

# Turn categorical data into numerical
print(df.Q6.eq("Yes").mul(1))
print(df.Q7.eq("Yes").mul(1))
print(df.Q8.eq("Yes").mul(1))
print(df.Q9.eq("Yes").mul(1))
print(df.Q10.eq("Yes").mul(1))
print(df.Q11.eq("Yes").mul(1))
print(df.Q12.eq("Yes").mul(1))
print(df.Q13.eq("Yes").mul(1))
print(df.Q14.eq("Yes").mul(1))
print(df.Q15.eq("Yes").mul(1))
print(df.Q16.eq("Yes").mul(1))
print(df.Q2.eq("male").mul(1))

# Customer attrition in data
print(df["Q16"].value_counts())
print(df.select_dtypes("number").head())


plt.style.use(["bmh", "ggplot"])

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

df["Q16"].value_counts().plot.pie(
    explode=[0, 0.08],
    ax=ax[0],
    autopct="%1.2f%%",
    shadow=True,
    fontsize=14,
    startangle=30,
    colors=["#00eb9c", "#a6814c"],
)
ax[0].set_title("Total Churn Percentage")

sns.countplot("Q16", data=df, ax=ax[1], palette=["#6abbcf", "#eb6851"])
ax[1].set_title("Total Number of Churn Customers")
ax[1].set_ylabel(" ")

plt.show()

Id_col = ["Q1"]
target_col = ["Q16"]
cat_cols = df.nunique()[df.nunique() < 6].keys().tolist()
cat_cols = [x for x in cat_cols if x not in target_col]
num_cols = [x for x in df.columns if x not in cat_cols + target_col + Id_col]

# Separating churn and non churn customers
churn = df[df["Q16"] == "Yes"]
not_churn = df[df["Q16"] == "No"]

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import plotly.graph_objs as go
import plotly.offline as py

py.init_notebook_mode(connected=True)


def plot_pie(column):

    trace1 = go.Pie(
        values=churn[column].value_counts().values.tolist(),
        labels=churn[column].value_counts().keys().tolist(),
        hoverinfo="label+percent+name",
        domain=dict(x=[0, 0.48]),
        name="Churn Customers",
        marker=dict(line=dict(width=2, color="rgb(33, 75, 198)")),
        hole=0.6,
    )
    trace2 = go.Pie(
        values=not_churn[column].value_counts().values.tolist(),
        labels=not_churn[column].value_counts().keys().tolist(),
        hoverinfo="label+percent+name",
        marker=dict(line=dict(width=2, color="rgb(33, 75, 99)")),
        domain=dict(x=[0.52, 1]),
        hole=0.6,
        name="Non Churn Customers",
    )

    layout = go.Layout(
        dict(
            title=column + " distribution in customer attrition ",
            plot_bgcolor="rgb(243,243,243)",
            paper_bgcolor="rgb(243,243,243)",
            annotations=[
                dict(
                    text="churn customers",
                    font=dict(size=13),
                    showarrow=False,
                    x=0.15,
                    y=0.5,
                ),
                dict(
                    text="Non Churn Customers",
                    font=dict(size=13),
                    showarrow=False,
                    x=0.88,
                    y=0.5,
                ),
            ],
        )
    )
    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


# function  for histogram for customer attrition types
def histogram(column):
    trace1 = go.Histogram(
        x=churn[column],
        histnorm="percent",
        name="Churn Customers",
        marker=dict(line=dict(width=0.5, color="black")),
        opacity=0.9,
    )

    trace2 = go.Histogram(
        x=not_churn[column],
        histnorm="percent",
        name="Non Churn customers",
        marker=dict(line=dict(width=0.5, color="black")),
        opacity=0.9,
    )

    data = [trace1, trace2]
    layout = go.Layout(
        dict(
            title=column + " distribution in customer attrition ",
            plot_bgcolor="rgb(243,243,243)",
            paper_bgcolor="rgb(243,243,243)",
            xaxis=dict(
                gridcolor="rgb(255, 255, 255)",
                title=column,
                zerolinewidth=1,
                ticklen=5,
                gridwidth=2,
            ),
            yaxis=dict(
                gridcolor="rgb(255, 255, 255)",
                title="percent",
                zerolinewidth=1,
                ticklen=5,
                gridwidth=2,
            ),
        )
    )
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)


# function  for scatter plot matrix  for numerical columns in data
def scatter_matrix(df):

    df = df.sort_values(by="Churn", ascending=True)
    classes = df["Churn"].unique().tolist()
    classes

    class_code = {classes[k]: k for k in range(2)}
    class_code

    color_vals = [class_code[cl] for cl in df["Churn"]]
    color_vals

    pl_colorscale = "Viridis"

    pl_colorscale

    text = [df.loc[k, "Churn"] for k in range(len(df))]
    text

    trace = go.Splom(
        dimensions=[
            dict(label="tenure", values=df["tenure"]),
            dict(label="MonthlyCharges", values=df["MonthlyCharges"]),
            dict(label="TotalCharges", values=df["TotalCharges"]),
        ],
        text=text,
        marker=dict(
            color=color_vals,
            colorscale=pl_colorscale,
            size=3,
            showscale=False,
            line=dict(width=0.1, color="rgb(230,230,230)"),
        ),
    )
    axis = dict(showline=True, zeroline=False, gridcolor="#fff", ticklen=4)

    layout = go.Layout(
        dict(
            title="Scatter plot matrix for Numerical columns for customer attrition",
            autosize=False,
            height=800,
            width=800,
            dragmode="select",
            hovermode="closest",
            plot_bgcolor="rgba(240,240,240, 0.95)",
            xaxis1=dict(axis),
            yaxis1=dict(axis),
            xaxis2=dict(axis),
            yaxis2=dict(axis),
            xaxis3=dict(axis),
            yaxis3=dict(axis),
        )
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


# all categorical columns plot pie
for i in cat_cols:
    plot_pie(i)

# for all categorical columns plot pie
for i in cat_cols:
    plot_pie(i)

# for all categorical columns plot histogram
for i in num_cols:
    histogram(i)

# scatter plot matrix


# Relationship analysis between income vs Churn

sns.boxplot(x="Q16", y="Q4", data=df)
sns.set_palette("cubehelix", 5)
plt.show()

# Churn based on the increase in tuition fees
sns.set(style="darkgrid")
sns.husl_palette(5, h=0.4)
fig, ax = plt.subplots(figsize=(20, 10))
ax = sns.countplot(x="Q8", hue="Q16", data=df)

# Customer Churn vs Non-recognition of the degree
sns.set(style="darkgrid")
sns.husl_palette(5, h=0.4)
fig, ax = plt.subplots(figsize=(20, 10))
ax = sns.countplot(x="Q10", hue="Q16", data=df)

# Customer churn vs no work permit
sns.set(style="darkgrid")
sns.husl_palette(5, h=0.4)
fig, ax = plt.subplots(figsize=(20, 10))
ax = sns.countplot(x="Q11", hue="Q16", data=df)

# Customer churn vs no EU student loans
sns.set(style="darkgrid")
sns.husl_palette(5, h=0.4)
fig, ax = plt.subplots(figsize=(20, 10))
ax = sns.countplot(x="Q9", hue="Q16", data=df)

# Correlation matrix and preprocessing for modelling

df["Q2"].replace(["male", "female"], [0, 1], inplace=True)
df["Q3"].replace(["18-25", "26-34", "35-44", "45-54"], [0, 1, 2, 3], inplace=True)
df["Q5"].replace(
    ["Single", "Married", "Separated", "None of the above"], [0, 1, 2, 3], inplace=True
)
df["Q6"].replace(["No", "Yes"], [0, 1], inplace=True)
df["Q7"].replace(["No", "Yes"], [0, 1], inplace=True)
df["Q8"].replace(["No", "Yes", "Do not know yet"], [0, 1, 2], inplace=True)
df["Q9"].replace(["No", "Yes", "Do not know yet"], [0, 1, 2], inplace=True)
df["Q10"].replace(["No", "Yes", "Do not know yet"], [0, 1, 3], inplace=True)
df["Q11"].replace(["No", "Yes", "Do not know yet"], [0, 1, 2], inplace=True)
df["Q12"].replace(
    [
        "Non-recognition of the degree",
        "No work permit",
        "International Tuition Fees",
        "No student loans offered",
    ],
    [0, 1, 2, 3],
    inplace=True,
)
df["Q13"].replace(["No", "Yes", "Do not know yet"], [0, 1, 2], inplace=True)
df["Q14"].replace(["No", "Yes", "Do not know yet"], [0, 1, 2], inplace=True)
df["Q15"].replace(["No", "Yes", "Do not know yet"], [0, 1, 2], inplace=True)
df["Q16"].replace(["No", "Yes"], [0, 1], inplace=True)


df.pop("Participant ID")
df.info()

sns.set(font_scale=1)
plot = sns.heatmap(df.corr(), cmap="cubehelix", linewidth=2, square=True)
print(plot)
print(df.corr())
