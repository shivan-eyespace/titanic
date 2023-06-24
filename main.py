"""Main file."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SETTINGS = {"THEME": None}

BASE_DIR = Path(__file__).parent
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
"""
cleaned_train_df = train_df.drop(["Cabin", "Name", "Ticket"], axis=1).dropna(
    how="any", axis=0
)
st.dataframe(cleaned_train_df.head())

with st.expander("Expand"):
    st.text("All data")
    st.dataframe(cleaned_train_df)
    st.text("Types")
    st.dataframe(cleaned_train_df.dtypes)

f"""
Passengers: `{len(cleaned_train_df)}`
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
1. Seperate input from results.
2. Turn `Sex` column into `Is_Female` and drop `Sex`.
3. Make dummy variables from `Embarked` and drop `Embarked`.
4. Normalise `Age`
5. Normalise `Fare`
"""

x_tab, y_tab = st.tabs(["Input", "Results"])


with x_tab:
    x_train = cleaned_train_df.drop("Survived", axis=1)
    x_train["Female"] = x_train.apply(
        lambda x: 1 if x["Sex"] == "female" else 0, axis=1
    )
    x_train.drop("Sex", axis=1, inplace=True)
    embarked_encoding = pd.get_dummies(
        x_train["Embarked"], prefix="Embarked", dtype=int
    )
    x_train = x_train.join(embarked_encoding)
    x_train.drop("Embarked", axis=1, inplace=True)
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(x_train[["Age", "Fare"]].to_numpy()),
        columns=["Age", "Fare"],
    )
    x_train.drop(columns=["Age", "Fare"], axis=1, inplace=True)
    x_train = x_train.join(scaled)
    st.dataframe(x_train)

with y_tab:
    y_train = cleaned_train_df["Survived"]
    st.dataframe(y_train)


# skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# for i, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
# X = x_train.iloc[train_index]
# Y = y_train.iloc[train_index]

# X_val = x_train.iloc[val_index]
# Y_val = y_train.iloc[val_index]

epochs = 2

X, X_val, Y, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# print(f"{X.shape}")
# print(f"{X.shape[1]}")

model = keras.Sequential(
    [
        layers.Dense(16, input_dim=X.shape[1], activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

if st.button("Train model"):
    history = model.fit(
        X, Y, epochs=epochs, batch_size=64, validation_data=(X_val, Y_val)
    )
    history_dict = history.history
    st.text(history_dict)
    fig = px.scatter(x=history_dict["loss"], y=history_dict["val_loss"])
    st.plotly_chart(fig, use_container_width=True)

    # FIXME : keep getting nan
