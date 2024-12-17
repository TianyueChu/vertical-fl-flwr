from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
import pandas as pd
from pathlib import Path

from vertical_fl.strategy import Strategy


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Get dataset
    processed_df = pd.read_csv(Path(__file__).parents[1] / f"data/server.csv")

    # Define the strategy
    strategy = Strategy(processed_df["fraud"].values)

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Start Flower server
app = ServerApp(server_fn=server_fn)
