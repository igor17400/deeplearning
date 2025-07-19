import os
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import F1Score


# --- Torch Classes ---
class MLPModel(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            dropout_rate
    ):
        super().__init__()
        layers = []
        for l_idx in range(num_layers - 1):
            layers += [
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            in_channels = hidden_channels
        layers += [nn.Linear(hidden_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)


class GNNModel(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            gnn_layer_by_name,
            layer_name="GCN",
            dropout_rate=0.1,
            **kwargs
    ):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            in_channels = hidden_channels
        layers += [
            gnn_layer(in_channels=hidden_channels, out_channels=out_channels, **kwargs),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


# --- Lightning Modules ---
class NodeLevelClassifier(L.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.f1_score = F1Score(task="multiclass", num_classes=model_kwargs.get("out_channels", 7))

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown mode"

        loss = self.loss_fn(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        f1 = self.f1_score(x[mask].argmax(dim=-1), data.y[mask])
        return loss, acc, f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_f1", f1)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc, f1 = self.forward(batch, mode="val")
        self.log("val_acc", acc)
        self.log("val_f1", f1)

    def test_step(self, batch, batch_idx):
        _, acc, f1 = self.forward(batch, mode="test")
        self.log("test_acc", acc)
        self.log("test_f1", f1)


def train_node_classifier(model_name, dataset, **model_kwargs):
    L.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    checkpoint_path = "./saved_models"

    root_dir = os.path.join(os.getcwd(), checkpoint_path, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = L.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(
            save_weights_only=True,
            mode="max",
            monitor="val_acc"
        )],
        accelerator="auto",
        devices=1,
        max_epochs=200,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    model = NodeLevelClassifier(
        model_name=model_name,
        in_channels=dataset.num_node_features,
        out_channels=dataset.num_classes,
        **model_kwargs
    )
    trainer.fit(model, node_data_loader, node_data_loader)

    # Test best model on the test set
    test_result = trainer.test(model, node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc, train_f1 = model.forward(batch, mode="train")
    _, val_acc, val_f1 = model.forward(batch, mode="val")
    result = {"train_acc": train_acc, "train_f1": train_f1,
              "val_acc": val_acc, "val_f1": val_f1,
              "test_acc": test_result[0]['test_acc'], "test_f1": test_result[0]['test_f1']}
    return model, result


def print_results(result_dict):
    if "train_acc" in result_dict:
        print(f"Train accuracy: {(100.0 * result_dict['train_acc']):4.2f}%")
        print(f"Train F1:       {(100.0 * result_dict['train_f1']):4.2f}%")
    if "val_acc" in result_dict:
        print(f"Val accuracy:   {(100.0 * result_dict['val_acc']):4.2f}%")
        print(f"Val F1:         {(100.0 * result_dict['val_f1']):4.2f}%")
    print(f"Test accuracy:  {(100.0 * result_dict['test_acc']):4.2f}%")
    print(f"Test F1:        {(100.0 * result_dict['test_f1']):4.2f}%")


# --- Model Instantiate ---
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
}

# --- Dataset ---
DATASET_PATH = "./datasets/"
cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")
print("-- cora dataset --")
print(cora_dataset[0])

# --- MLP Model ---
node_mlp_model, node_mlp_result = train_node_classifier(
    model_name="MLP",
    dataset=cora_dataset,
    hidden_channels=16,
    num_layers=2,
    dropout_rate=0.1
)
print(node_mlp_result)

# --- GCN Model ---
node_gcn_model, node_gcn_result = train_node_classifier(
    model_name="GCN",
    layer_name="GCN",
    gnn_layer_by_name=gnn_layer_by_name,
    dataset=cora_dataset,
    hidden_channels=16,
    num_layers=2,
    dropout_rate=0.1
)
print(node_gcn_result)
