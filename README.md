---
tags: [vertical, tabular, advanced]
dataset: [Titanic]
framework: [torch, pandas, scikit-learn]
---

# Vertical Federated Learning with Flower

This project implements Vertical Federated Learning using  Flower on tabular data.

We conduct a classification task for fraud detection based on data from a bank and a telecommunication company, each with 31 features. 

On the VFL side, the server holds the labels (the `fraud` column) and will be the only one capable of performing evaluation.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/vertical-fl . \
        && rm -rf _tmp \
        && cd vertical-fl
```

This will create a new directory called `vertical-fl` with the following structure:
following files:

```shell
vertical-fl
├── vertical_fl
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   ├── strategy.py     # Defines your Strategy
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
├── data
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `mlxexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower project with the Deployment Engine and TLS certificates, or with Docker.
