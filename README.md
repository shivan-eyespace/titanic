# Titanic Dataset Analysis

## Acknowledgements

Dataset is [sourced from Kaggle](https://www.kaggle.com/c/titanic).

## Requirements

- Python
- Docker
- Docker Compose

## How to Run for Development

### Docker

Using Docker Compose:

```sh
docker-compose up --build
```

Just using Docker (WIP):
```sh
docker build
  -t titanic_app .
  -f .docker/dev/Dockerfile
docker run titanic_app
  -p 8501:8501
  -v .:/src
```

### Python

I suggest creating a virtual environment and then installing packages through this.

```sh
# create virtual environment
python -m venv .venv

# activate environment
source bin/.venv/activate
# for fish users (like me)
source bin/.venv/activate.fish

# install packages
pip install -r requirements.txt
pip install -r dev-requirements.txt

# run streamlit
streamlit run main.py
```
