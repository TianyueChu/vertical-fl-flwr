from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import torch

from vertical_fl.task import ClientModel, load_data

Ndata = ['bank','telco']

class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, lr):
        self.v_split_id = v_split_id
        self.data = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(input_size=self.data.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        embedding = self.model(self.data)
        return [embedding.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        self.model.zero_grad()
        embedding = self.model(self.data)
        embedding.backward(torch.from_numpy(parameters[int(self.v_split_id)]))
        self.optimizer.step()
        return 0.0, 1, {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    partition = load_data(data_name=Ndata[partition_id])
    lr = context.run_config["learning-rate"]
    return FlowerClient(partition_id, partition, lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)
