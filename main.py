"""Main file."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas import DataFrame, Series

from utils.correlations import tetrachoric

SETTINGS = {"THEME": None}

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

test_df = pd.read_csv(DATA_DIR / "test.csv", index_col="PassengerId")
train_df = pd.read_csv(DATA_DIR / "train.csv", index_col="PassengerId")

"""
# Titanic Dataset


Analysing the titanic dataset.

Source: https://www.kaggle.com/c/titanic

## Raw Train Data
"""

st.dataframe(train_df)

"""
## Data Cleaning

"""
column_counts = train_df.count()
column_counts_fig = px.bar(
    column_counts,
    labels={"value": "count", "index": "column"},
    title="Column Counts of Titanic Data",
)
column_counts_fig.update_layout(showlegend=False)
st.plotly_chart(column_counts_fig, use_container_width=True)

display_nulls = train_df.notna()
display_nulls_fig = px.imshow(train_df.notna(), title="Nulls in Titanic Data")
st.plotly_chart(display_nulls_fig, use_container_width=True)

"""
1. Drop column:
    - `Cabin` due to low count
    - `Name` because not relevant, can identify by `PassengerId`.
2. Removed any nulls on a row.

Cleaned Data:
"""
cleaned_train_df = train_df.drop(["Cabin", "Name"], axis=1).dropna(how="any", axis=0)
st.dataframe(cleaned_train_df)

"""
3. Types
"""
st.dataframe(cleaned_train_df.dtypes)

f"""
Passengers: `{len(cleaned_train_df)}`
"""

"""
"""

"""
### Exploratory Data Analysis

#### Data distributions
"""


def determine_proportion(df: DataFrame, title: str):
    """Provide proportion on survived."""
    df = df["survived"] / (df["survived"] + df["deceased"])
    fig = px.bar(df, title=title)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def split_data(df: DataFrame, ds: Series) -> DataFrame:
    """Split into surivived and deceased."""
    return pd.DataFrame(
        {
            "survived": ds[df["Survived"] == 1].value_counts(),
            "deceased": ds[df["Survived"] == 0].value_counts(),
        }
    )


def _survived(df: DataFrame) -> None:
    data = (
        df["Survived"].apply(lambda x: {1: "Survived", 0: "Deceased"}[x]).value_counts()
    )
    fig = px.bar(data, title="Survived Totals")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _pclass(df: DataFrame) -> None:
    ds = df["Pclass"].apply(lambda x: {1: "1st", 2: "2nd", 3: "3rd"}[x])
    data = split_data(df=df, ds=ds)
    fig = px.bar(data, title="Pclass Totals")
    st.plotly_chart(fig, use_container_width=True)

    determine_proportion(df=data, title="Pclass Survivability")


def _sex(df: pd.DataFrame) -> None:
    ds = df["Sex"]
    data = split_data(df=df, ds=ds)
    fig = px.bar(data, title="Sex Total")
    st.plotly_chart(fig, use_container_width=True)

    determine_proportion(df=data, title="Sex Survivability")


def _age(df: DataFrame):
    ds = df["Age"]
    data = pd.DataFrame(
        {
            "survived": ds[df["Survived"] == 1],
            "deceased": ds[df["Survived"] == 0],
        }
    )
    fig = px.histogram(data, title="Age", marginal="box")
    st.plotly_chart(fig, use_container_width=True)


def _sibsp(df: DataFrame):
    ds = df["SibSp"]
    data = split_data(df=df, ds=ds)
    fig = px.bar(data, title="SipSp Totals")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    determine_proportion(df=data, title="SibSp Survivability")


def _parch(df: DataFrame):
    ds = df["Parch"]
    data = split_data(df=df, ds=ds)
    fig = px.bar(data, title="Parch")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    determine_proportion(df=data, title="Parch Survivability")


def _fare(df: DataFrame):
    ds = df["Fare"]
    data = pd.DataFrame(
        {
            "survived": ds[df["Survived"] == 1],
            "deceased": ds[df["Survived"] == 0],
        }
    )
    fig = px.histogram(data, title="Fare", marginal="box")
    st.plotly_chart(fig, use_container_width=True)


def _embarked(df: DataFrame):
    ds = df["Embarked"]
    data = split_data(df=df, ds=ds)
    fig = px.bar(data, title="Embarked")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    determine_proportion(df=data, title="Embarked Survivability")


_survived(cleaned_train_df)
_pclass(cleaned_train_df)
_sex(cleaned_train_df)
_age(cleaned_train_df)
_sibsp(cleaned_train_df)
_parch(cleaned_train_df)
_fare(cleaned_train_df)
_embarked(cleaned_train_df)

"""
#### Correlations
"""


def _sex_correlation(df: DataFrame):
    ds = cleaned_train_df["Sex"]
    data = split_data(df, ds)
    correlation = abs(tetrachoric(data.values))
    st.dataframe(data)
    return correlation


def _pclass_correlation():
    pass
