"""Main file."""

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas import DataFrame, Series
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SETTINGS = {"THEME": None}

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

test_df = pd.read_csv(DATA_DIR / "test.csv", index_col="PassengerId")
train_df = pd.read_csv(DATA_DIR / "train.csv", index_col="PassengerId")

"""
# Titanic Dataset

Analysing the titanic dataset.

Source: https://www.kaggle.com/c/titanic
"""

"""## Raw Train Data"""
st.dataframe(train_df.head(4))
with st.expander("Expand"):
    st.dataframe(train_df)
    st.dataframe(train_df.describe())

"""
## Data Cleaning

"""


def _column_counts(df: DataFrame):
    data = df.count()
    fig = px.bar(
        data,
        labels={"value": "count", "index": "column"},
        title="Column Counts of Titanic Data",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _display_nulls(df: DataFrame):
    data = df.notna()
    fig = px.imshow(data, title="Nulls in Titanic Data")
    st.plotly_chart(fig, use_container_width=True)


column_counts, display_nulls = st.tabs(["Column Counts", "Nulls in Data"])
with column_counts:
    _column_counts(train_df)
with display_nulls:
    _display_nulls(train_df)


"""
1. Drop column:
    - `Cabin` due to low count.
    - `Name` because not relevant, can identify by `PassengerId`.
    - `Ticket` is likely not relevant.
2. Removed any nulls on a row.
3. Reset index and drop the passengerId.
4. Apply same changes to the test data.
"""


def cleaning(df: DataFrame) -> DataFrame:
    """Clean up data."""
    columns_to_drop = ["Cabin", "Name", "Ticket"]
    df.drop(columns_to_drop, axis=1, inplace=True)
    df.dropna(how="any", axis=0, inplace=True)
    return df


cleaned_train_df = cleaning(train_df)
st.dataframe(cleaned_train_df.head())

with st.expander("Expand"):
    st.text("All data")
    st.dataframe(cleaned_train_df)
    st.text("Types")
    st.dataframe(cleaned_train_df.dtypes)

cleaned_test_df = cleaning(test_df)

f"""
Passengers in train data: `{len(cleaned_train_df)}`
"""

"""
### Exploratory Data Analysis
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


survived, pclass, sex, age, sibsp, parch, fare, embarked = st.tabs(
    ["Surived", "Pclass", "Sex", "Age", "Sibsp", "Parch", "Fare", "Embarked"]
)
with survived:
    _survived(cleaned_train_df)
with pclass:
    _pclass(cleaned_train_df)
with sex:
    _sex(cleaned_train_df)
with age:
    _age(cleaned_train_df)
with sibsp:
    _sibsp(cleaned_train_df)
with parch:
    _parch(cleaned_train_df)
with fare:
    _fare(cleaned_train_df)
with embarked:
    _embarked(cleaned_train_df)

"""
## Encoding and Further cleaning
"""

"""
1. Separate input from results.
2. Turn `Sex` column into `Is_Female` and drop `Sex`.
3. Make dummy variables from `Embarked` and drop `Embarked`.
4. Normalise `Age`.
5. Normalise `Fare`.
6. Apply this to the test data.
"""

x_tab, y_tab = st.tabs(["Input", "Results"])


Transformer = Any | None


def encoding(
    df: DataFrame, transformer: Transformer = None
) -> tuple[DataFrame, Transformer]:
    """Encode data."""
    df["Female"] = df.apply(lambda x: 1 if x["Sex"] == "female" else 0, axis=1)
    embarked_encoding = pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=int)
    df = df.join(embarked_encoding)
    scaler = StandardScaler()
    if transformer is None:
        transformer = scaler.fit(df[["Age", "Fare"]].to_numpy())
    scaled = pd.DataFrame(
        transformer.transform(df[["Age", "Fare"]].to_numpy()),
        columns=["Age", "Fare"],
        index=df.index,
    )
    df.drop(["Sex", "Embarked", "Age", "Fare"], axis=1, inplace=True)
    df = df.join(scaled)
    return df, transformer


y_train = cleaned_train_df["Survived"]
x_train_before_encoding = cleaned_train_df.drop("Survived", axis=1)
x_train, transformer = encoding(x_train_before_encoding)
x_test, _ = encoding(cleaned_test_df, transformer)

with x_tab:
    st.dataframe(x_train)

with y_tab:
    st.dataframe(y_train)


epochs = range(1, 10, 5)
f"""
## Create model and run predictions

1. Create model.
2. Grid Search to fine-tune on `epochs = {list(epochs)}`.
"""


def create_model():
    """Create model."""
    model = keras.Sequential(
        [
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


model = KerasClassifier(build_fn=create_model)
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
param_grid = dict(epochs=epochs)
clf = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    n_jobs=-1,
    cv=cv,
)

if st.button("Train model"):
    model.fit(X=x_train, y=y_train)
    clf.fit(x_train, y_train)
    st.dataframe(clf.cv_results_)

"""
## Prediction
"""
# y_preds = model.predict(x_train)
# st.dataframe(
#     pd.DataFrame(y_preds, columns=["Survived"], index=x_train.index)
# )
