# -- Libraries

import numpy as np
import warnings
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Input
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.random import set_seed

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess

# Suppress SciPy sparse efficiency warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.sparse')

# -- Seed
set_seed(0)

# -- Dataset
dataset = Citation("cora", normalize_x=True, transforms=[LayerPreprocess(GATConv)])

# -- Prepare Sample Weights
def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)

# training, validation, testing
weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)

# -- Parameters
channels = 8  # Number of channels in each head of the first GAT layer
n_attn_heads = 8  # Number of attention heads in first GAT layer
dropout = 0.6  # Dropout rate for the features and adjacency matrix
l2_reg = 2.5e-4  # L2 regularization rate
learning_rate = 5e-3  # Learning rate
epochs = 100 # Number of training epochs
patience = 5  # Patience for early stopping

# -- Graph Dimension
N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes

# -- Model Definition
# --- Input
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True) # (N,) for a square matrix of size N x N

# --- Dropoout + Attention (Part 1)
do_1 = Dropout(dropout)(x_in)

# Disable masking for GATConv to avoid None mask issues
# class GATConvNoMask(GATConv):
#     def call(self, inputs, **kwargs):
#         # Remove mask from kwargs to prevent None mask issues
#         kwargs.pop('mask', None)
#         return super().call(inputs, **kwargs)

gc_1 = GATConv(
    channels,
    attn_heads=n_attn_heads,
    concat_heads=True,
    dropout_rate=dropout,
    activation="elu",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([do_1, a_in])

# --- Dropoout + Attention (Part 2)
do_2 = Dropout(dropout)(gc_1)
gc_2 = GATConv(
    n_out,
    attn_heads=1,
    concat_heads=False,
    dropout_rate=dropout,
    activation="softmax",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([do_2, a_in])

# -- Build model
model = Model(inputs=[x_in, a_in], outputs=gc_2)

# -- Model Compilation
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(reduction="sum"),
    weighted_metrics=["acc"],
)
model.summary()

# -- Train model
loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)

# -- Evaluate Model
print("Evaluating model.")

loader_te = SingleLoader(dataset, sample_weights=weights_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)

print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))