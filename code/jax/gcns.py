# -- Libraries --

from flax import nnx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp  # Useful for handling sparse adjacency matrices
from typing import Sequence
import optax
import matplotlib.pyplot as plt


# -- Data --

def _load_content_file(filepath: str):
    """Loads features and labels from the .content file."""
    print(f"Loading content file: {filepath}")
    idx_features_labels = np.genfromtxt(
        filepath,
        dtype=np.dtype(str)
    )
    features = sp.csr_matrix(
        idx_features_labels[:, 1:-1],
        dtype=np.float32
    )
    labels = idx_features_labels[:, -1]
    paper_ids = np.array(
        idx_features_labels[:, 0],
        dtype=np.int32
    )
    return features, labels, paper_ids


def _load_cites_file(filepath: str, idx_map: dict):
    """Loads citation links and builds the adjacency matrix."""
    print(f"Loading cites file: {filepath}")
    edges_unordered = np.genfromtxt(
        filepath,
        dtype=np.int32
    )

    # Convert paper IDs in edges to our new integer indices
    edges = np.array(
        list(map(
            idx_map.get,
            edges_unordered.flatten())
        ), dtype=np.int32,
    ).reshape(edges_unordered.shape)

    # Create COO (Coordinate) format sparse adjacency matrix
    adj = sp.coo_matrix(
        (
            np.ones(edges.shape[0]),
            (edges[:, 0], edges[:, 1])
        ),
        # Use len(idx_map) for the number of nodes
        shape=(len(idx_map), len(idx_map)),
        dtype=np.float32
    )

    # Build symmetric adjacency matrix
    adj = (
            adj +
            adj.T.multiply(adj.T > adj) -
            adj.multiply(adj.T > adj)
    )
    return adj


def _normalize_features(features: np.ndarray):
    """Normalizes node features."""
    features = features.todense()
    row_sums = features.sum(1).reshape(-1, 1)
    # Avoid division by zero for nodes with no features
    features = features / np.where(row_sums > 0, row_sums, 1.0)
    return features


def _one_hot_encode_labels(labels: np.ndarray):
    """Converts categorical labels to one-hot encoded format."""
    classes = sorted(list(set(labels)))
    class_to_idx = {
        c: i for i, c in enumerate(classes)
    }

    # Corrected labels_one_hot initialization shape
    labels_one_hot = np.zeros(
        (len(labels), len(classes)),
        dtype=np.float32
    )
    for i, label in enumerate(labels):
        labels_one_hot[i, class_to_idx[label]] = 1
    return labels_one_hot


def _create_masks(num_nodes: int):
    """Defines standard train, validation, and test masks."""
    idx_train = jnp.arange(140)
    idx_val = jnp.arange(140, 140 + 500)
    idx_test = jnp.arange(140 + 500, 140 + 500 + 1000)

    train_mask = jnp.zeros(
        num_nodes, dtype=bool
    ).at[idx_train].set(True)
    val_mask = jnp.zeros(
        num_nodes, dtype=bool
    ).at[idx_val].set(True)
    test_mask = jnp.zeros(
        num_nodes, dtype=bool
    ).at[idx_test].set(True)

    return train_mask, val_mask, test_mask


def load_cora(path: str = "./dataset/cora/"):
    """
    Loads and preprocesses the Cora dataset
    from raw .content and .cites files.

    Args:
        path (str): Directory path where
            cora.content and cora.cites are located.
            Defaults to "./dataset/cora/".

    Returns:
        tuple:
        (adj, features, labels, train_mask, val_mask, test_mask)
        where adj, features, and labels are JAX numpy arrays.
    """
    # 1. Load features and labels, and create paper ID to index mapping
    features_sparse, labels_raw, paper_ids = _load_content_file(path + "cora.content")
    idx_map = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}

    # 2. Build adjacency matrix
    adj_sparse = _load_cites_file(path + "cora.cites", idx_map)

    # 3. Preprocess features
    features_normalized = _normalize_features(features_sparse)

    # 4. One-hot encode labels
    labels_one_hot = _one_hot_encode_labels(labels_raw)

    # 5. Create masks for data splits
    num_nodes = labels_raw.shape[0]  # Use the original number of labels for node count
    train_mask, val_mask, test_mask = _create_masks(num_nodes)

    # 6. Convert to JAX numpy arrays
    adj_jax = jnp.asarray(adj_sparse.todense(), dtype=jnp.float32)
    features_jax = jnp.asarray(features_normalized, dtype=jnp.float32)
    labels_jax = jnp.asarray(labels_one_hot, dtype=jnp.float32)

    return adj_jax, features_jax, labels_jax, train_mask, val_mask, test_mask


def preprocess_adjacency_matrix(adj):
    """
    Adds self-loops and symmetrically normalizes the adjacency matrix.

    Args:
        adj (jnp.ndarray): The raw adjacency matrix from your dataset.

    Returns:
        jnp.ndarray: The transformed adjacency matrix ready for the GCN.
    """
    # Input validation
    if adj is None:
        raise ValueError("Adjacency matrix cannot be None")
    if adj.ndim != 2:
        raise ValueError(f"Expected 2D adjacency matrix, got {adj.ndim}D")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj.shape}")

    # Step 1: Add self-loops (A' = A + I)
    # This ensures each node considers its own features during aggregation.
    adj_self_loops = adj + jnp.eye(adj.shape[0])

    # Step 2: Calculate the inverse square root of the degree matrix (D'^-0.5)
    # This creates the normalization factor.
    degrees = jnp.sum(adj_self_loops, axis=1)
    # Add small epsilon to prevent division by zero for isolated nodes
    D_hat = jnp.diag((degrees + 1e-8) ** -0.5)

    # Step 3: Symmetrically normalize the matrix (D'^-0.5 * A' * D'^-0.5)
    # This averages neighbor features, preventing issues from node degrees.
    A_hat = D_hat @ adj_self_loops @ D_hat

    return A_hat


# -- GCN --
class GCNLayer(nnx.Module):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            dropout_rate: float = 0.0,
            apply_activation: bool = True,
            *, rngs: nnx.Rngs
    ):
        self.linear = nnx.Linear(
            input_features,
            output_features,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
        )
        self.dropout = nnx.Dropout(
            rate=dropout_rate,
            rngs=rngs
        )
        self.apply_activation = apply_activation

    def __call__(
            self,
            A_hat: jnp.ndarray,
            H: jnp.ndarray,
            *, rngs: nnx.Rngs,
    ):
        """
        Applies a single Graph Convolutional
        Network layer using NNX.

        Args:
            A_hat (jnp.ndarray): The symmetrically
                normalized adjacency matrix.
            H (jnp.ndarray): The input node embeddings/
                features from the previous layer.
            rngs (nnx.Rngs): Random number generator
                collection for dropout.

        Returns:
            jnp.ndarray: The output node embeddings/
                features for the current layer.
        """
        # 1. Dropout for regularization
        # nnx knows when to apply dropout - train
        #   and when don't - eval
        H_dropped = self.dropout(H, rngs=rngs)

        # 2. Linear Transformation (H^(l) * W^(l))
        H_linear = self.linear(H_dropped)

        # 3. Graph Convolution (A_hat * H_linear)
        H_aggregated = A_hat @ H_linear

        # 4. Activation Function (if enabled)
        if self.apply_activation:
            H_output = nnx.relu(H_aggregated)
        else:
            H_output = H_aggregated

        return H_output


class GCN(nnx.Module):
    """A graph Convolutional Network Model"""

    def __init__(
            self,
            input_features: int,
            hidden_features: Sequence[int],
            output_features: int,
            dropout_rate: float,
            *, rngs: nnx.Rngs
    ):
        self.gcn_layers = []
        current_features = input_features

        # Create the hidden GCN layers
        for hidden_dim in hidden_features:
            self.gcn_layers.append(
                GCNLayer(
                    input_features=current_features,
                    output_features=hidden_dim,
                    dropout_rate=dropout_rate,
                    rngs=rngs
                )
            )
            # The input of the next layer is the
            #   output to this one
            current_features = hidden_dim

        # The final layer maps node embeddings to output classes
        # No activation for final layer (raw logits for classification)
        self.output_layer = GCNLayer(
            input_features=current_features,
            output_features=output_features,
            dropout_rate=0.0,  # No dropout in final layer
            apply_activation=False,
            rngs=rngs
        )

    def __call__(
            self,
            A_hat: jnp.ndarray,
            H: jnp.ndarray,
            *, rngs: nnx.Rngs,
    ):
        """Performs the forward pass of the GCN."""
        # Pass data through all the GCN layers
        for layer in self.gcn_layers:
            H = layer(A_hat=A_hat, H=H, rngs=rngs)

        # Apply the final layer to get logits
        logits = self.output_layer(A_hat=A_hat, H=H, rngs=rngs)
        return logits


# --- Loss & Accuracy Functions ---
def categorical_loss(logits, labels, mask):
    """Calculates masked cross-entropy loss."""
    # Convert one-hot labels to integer labels for this loss function
    int_labels = jnp.argmax(labels, axis=1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, int_labels)
    # Use jnp.where to apply mask without boolean indexing
    masked_loss = jnp.where(mask, loss, 0.0)
    # Return the mean loss only for the nodes in the mask
    return jnp.sum(masked_loss) / jnp.sum(mask)


def loss_fn(model: GCN, batch, rngs: nnx.Rngs):
    logits = model(
        batch["adj"],
        batch["features"],
        rngs=rngs
    )
    loss = categorical_loss(
        logits,
        batch["label"],
        batch["mask"]
    )
    return loss, logits


@nnx.jit
def train_step(
        model: GCN,
        optimizer,
        metrics: nnx.MultiMetric,
        batch,
        rngs: nnx.Rngs
):
    """Performs one training step: computes grads, updates model."""
    grad_fn = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )
    (loss, logits), grads = grad_fn(model, batch, rngs)

    metrics.update(
        loss=loss,
        logits=logits,
        labels=jnp.argmax(batch["label"], axis=1),
    )
    optimizer.update(grads)  # In-place updates


@nnx.jit
def eval_step(
        model: GCN,
        metrics: nnx.MultiMetric,
        batch,
        rngs: nnx.Rngs
):
    # In evaluation mode, dropout is automatically disabled by NNX
    # when no training is happening (deterministic behavior)
    loss, logits = loss_fn(model, batch, rngs)
    
    metrics.update(
        loss=loss,
        logits=logits,
        labels=jnp.argmax(batch["label"], axis=1),
    )


# Create batch format for GCN training
def create_batch(adj, features, labels, mask):
    """Creates a batch in the format expected by the training loop."""
    return {
        "adj": adj,
        "label": labels,
        "mask": mask,
        "features": features
    }


# For the training loop, you can also create a simple dataset iterator
def create_dataset_iterator(batch, num_repeats=1):
    """Creates an iterator that yields the same batch repeatedly."""
    for _ in range(num_repeats):
        yield batch


# --- Model & Hyperparameters ---
LEARNING_RATE = 0.01
EPOCHS = 200
HIDDEN_FEATURES = [16, 16]
DROPOUT_RATE = 0.2

# --- Load Data ---
adj_raw, features, labels, train_mask, val_mask, test_mask = load_cora()
# Create the correctly processed adjacency matrix
adj_normalized = preprocess_adjacency_matrix(adj_raw)

# Create train, validation, and test batches
train_batch = create_batch(adj_normalized, features, labels, train_mask)
val_batch = create_batch(adj_normalized, features, labels, val_mask)
test_batch = create_batch(adj_normalized, features, labels, test_mask)

# Print dataset information
print(f"\nDataset splits:")
print(f"Training nodes: {jnp.sum(train_mask)}")
print(f"Validation nodes: {jnp.sum(val_mask)}")
print(f"Test nodes: {jnp.sum(test_mask)}")
print(f"Total nodes: {len(features)}")

# Create dataset iterators
train_ds = create_dataset_iterator(train_batch, EPOCHS)
val_ds = create_dataset_iterator(val_batch, 1)  # For evaluation
test_ds = create_dataset_iterator(test_batch, 1)  # For final evaluation

print("Batch format created successfully!")
print(f"Train batch keys: {list(train_batch.keys())}")
print(f"Features shape: {train_batch['features'].shape}")
print(f"Labels shape: {train_batch['label'].shape}")
print(f"Adjacency shape: {train_batch['adj'].shape}")
print(f"Train mask sum: {jnp.sum(train_batch['mask'])}")
print(f"Validation mask sum: {jnp.sum(val_batch['mask'])}")
print(f"Test mask sum: {jnp.sum(test_batch['mask'])}")

# --- Metrics definition ---
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
)

# --- Initialize Model and Optimizer ---
model = GCN(
    input_features=features.shape[1],
    output_features=labels.shape[1],
    hidden_features=HIDDEN_FEATURES,
    dropout_rate=DROPOUT_RATE,
    rngs=nnx.Rngs(0),
)
optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE))

# --- Training Loop ---
metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'val_loss': [],
    'val_accuracy': [],
}

# Best model tracking
best_val_acc = 0.0
best_model_state = None

eval_every = 20
train_steps = EPOCHS

# Initialize RNG for training
rng = nnx.Rngs(42)

for step in range(train_steps):
    # Run the optimization for one step and make a stateful update to the following:
    # - The train state's model parameters
    # - The optimizer state
    # - The training loss and accuracy batch metrics
    train_step(model, optimizer, metrics, train_batch, rng)

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
        # Log the training metrics.
        for metric, value in metrics.compute().items():  # Compute the metrics.
            metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
        metrics.reset()  # Reset the metrics for the test set.

        # Compute the metrics on the validation set after each training epoch.
        eval_step(model, metrics, val_batch, rng)

        # Log the validation metrics.
        for metric, value in metrics.compute().items():
            metrics_history[f'val_{metric}'].append(value)
        metrics.reset()  # Reset the metrics for the next training epoch.

        # Track best model based on validation accuracy
        current_val_acc = metrics_history['val_accuracy'][-1]
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_model_state = nnx.state(model)  # Save model state
            print(f"New best validation accuracy: {best_val_acc:.4f}")

        print(f"Step {step}: Train Loss: {metrics_history['train_loss'][-1]:.4f}, "
              f"Train Acc: {metrics_history['train_accuracy'][-1]:.4f}, "
              f"Val Loss: {metrics_history['val_loss'][-1]:.4f}, "
              f"Val Acc: {metrics_history['val_accuracy'][-1]:.4f}")

# --- Final Test Evaluation ---
print("\n=== Training Complete ===")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# Load best model for final testing
if best_model_state is not None:
    nnx.update(model, best_model_state)
    print("Loaded best model for final testing")

# Final test evaluation
print("\n=== Final Test Evaluation ===")
test_metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
)

eval_step(model, test_metrics, test_batch, rng)
final_test_results = test_metrics.compute()

print(f"Final Test Loss: {final_test_results['loss']:.4f}")
print(f"Final Test Accuracy: {final_test_results['accuracy']:.4f}")

# Plot final training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train', 'val'):
    ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
    ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()

# Summary
print("\n=== Summary ===")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Final Test Accuracy: {final_test_results['accuracy']:.4f}")
