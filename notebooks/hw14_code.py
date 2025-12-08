import os
import random

import numpy as np
import pandas as pd

# Ask XLA for deterministic kernels (no device pinning so it works on CPU/GPU/TPU)
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops")

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
jax_master_rng = jax.random.PRNGKey(SEED)

import plotly.graph_objs as go
import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
df = pd.read_csv(url, parse_dates=True, index_col="timestamp")

# Define the anomalies
anomalies = {
    "NYC Marathon": "2014-11-02 01:00",
    "Thanksgiving": "2014-11-27 16:30",
    "Christmas": "2014-12-25 16:30",
    "New Year's Day": "2015-01-01 01:00",
    "Snow Storm": "2015-01-27 13:30"
}

# Create the plot
fig = go.Figure()

# Add the main trace
fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name='Taxi Passengers'))

# Add anomaly points with unique markers
markers = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down']
for (label, date), marker in zip(anomalies.items(), markers):
    fig.add_trace(go.Scatter(
        x=[date],
        y=[df.loc[date]['value']],
        mode='markers+text',
        name=label,
        text=label,
        textposition="top center",
        marker=dict(symbol=marker, size=10, color='red')
    ))

# Update layout
fig.update_layout(
    title='New York City Taxi Passengers with Anomalies Highlighted',
    xaxis_title='Date',
    yaxis_title='Number of Passengers',
    legend_title='Legend',
    template='plotly_white'
)

# Show the figure
# Config for the plot
plotly_config = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
        'filename': 'fmin',
        'height': None,
        'width': None,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
    },
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    'modeBarButtonsToAdd': [
                            'drawopenpath',
                            'eraseshape'
                            ]
}

# Show the figure with the specified config
fig.show(config=plotly_config)

fig.write_html(
    "./anomaly_detection.html",
    config=plotly_config,
    include_plotlyjs="cdn",
    full_html=False,
)

import plotly.graph_objs as go
from datetime import timedelta

# Load the dataset
url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
df = pd.read_csv(url, parse_dates=True, index_col="timestamp")

# Define the anomalies
anomalies = {
    "NYC Marathon": "2014-11-02 01:00",
    "Thanksgiving": "2014-11-27 16:30",
    "Christmas": "2014-12-25 16:30",
    "New Year's Day": "2015-01-01 01:00",
    "Snow Storm": "2015-01-27 13:30"
}

# Convert anomaly dates to datetime
anomaly_dates = [pd.to_datetime(date) for date in anomalies.values()]

# Function to split the data into continuous segments
def split_data(data, anomaly_dates, window=timedelta(days=1)):
    segments = []
    current_segment = []
    for date, row in data.iterrows():
        # if we are in the window before the segment, we go to the next segment
        if any(abs(date - anomaly_date) <= window for anomaly_date in anomaly_dates):
            if current_segment:
                segments.append(pd.DataFrame(current_segment))
                current_segment = []
        # else we fill the current segment
        else:
            current_segment.append(row)
    if current_segment:
        segments.append(pd.DataFrame(current_segment))
    return segments

# Split the normal data
normal_segments = split_data(df, anomaly_dates)

# Split the anomaly data
anomaly_window = timedelta(days=1)
anomaly_segments = []
for anomaly_date in anomaly_dates:
    segment = df[(df.index >= anomaly_date - anomaly_window) & (df.index <= anomaly_date + anomaly_window)]
    anomaly_segments.append(segment)

# Create the plot
fig = go.Figure()

# Add the normal data segments with legendgroup
for i, segment in enumerate(normal_segments):
    fig.add_trace(go.Scatter(
        x=segment.index, y=segment['value'], mode='lines',
        name='Normal Data' if i == 0 else None,
        legendgroup='Normal Data',
        line=dict(color='blue'),
        showlegend=i == 0
    ))

# Add the anomaly data segments with legendgroup
for i, segment in enumerate(anomaly_segments):
    fig.add_trace(go.Scatter(
        x=segment.index, y=segment['value'], mode='lines',
        name='Anomaly Data' if i == 0 else None,
        legendgroup='Anomaly Data',
        line=dict(color='red'),
        showlegend=i == 0
    ))

# Update layout
fig.update_layout(
    title='New York City Taxi Passengers with Anomalies Highlighted',
    xaxis_title='Date',
    yaxis_title='Number of Passengers',
    legend_title='Legend',
    template='plotly_white'
)

# Config for the plot
config = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
        'filename': 'fmin',
        'height': None,
        'width': None,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
    },
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    'modeBarButtonsToAdd': [
                            'drawopenpath',
                            'eraseshape'
                            ]
}

# Show the figure with the specified config
fig.show(config=config)


df_normalized = df
df_normalized['value'] = (df['value'] - df['value'].mean()) / df['value'].std()
# Split the normal data
normal_segments = split_data(df_normalized, anomaly_dates)

# Normalize each segment separately
def normalize_segment(segment):
    mean_value = segment['value'].mean()
    std_value = segment['value'].std()
    return (segment['value'] - mean_value) / std_value

normalized_segments = [normalize_segment(segment) for segment in normal_segments]

# Define the TIME_STEPS
TIME_STEPS = 100

# Function to create overlapping chunks from each segment
def create_chunks(segment, time_steps):
    chunks = []
    segment_values = segment.values
    for i in range(len(segment_values) - time_steps + 1):
        chunks.append(segment_values[i: i + time_steps])
    return chunks

# Create chunks from each normalized segment
x_train = []
x_full = []
for segment in normalized_segments:
    chunks = create_chunks(segment, TIME_STEPS)
    x_train.extend(chunks)

x_full = create_chunks(df_normalized, TIME_STEPS)

x_train = np.array(x_train)
x_full = np.array(x_full)[:,:,0]

# Print the shape of x_train
print(f'x_train shape: {x_train.shape}')
print(f'x_full shape: {x_full.shape}')

x_train = x_train.reshape((-1, TIME_STEPS, 1))
x_full = x_full.reshape((-1, TIME_STEPS, 1))

print(f'x_train shape: {x_train.shape}')
print(f'x_full shape: {x_full.shape}')


# Define the autoencoder model
class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(7,))(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(7,))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=16, kernel_size=(7,))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(7,))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(7,))(x)
        return x

# Initialize model
rng_init, rng_state, rng_sample = jax.random.split(jax_master_rng, 3)  # independent streams for init/train/visualization
input_shape = (128, TIME_STEPS, 1)
x = jnp.ones(input_shape)
model = Autoencoder()
params = model.init(rng_init, x)
output = model.apply(params, x_train)
print(f"Model output shape {output.shape}")
print(model.tabulate(jax.random.key(SEED), x,
                   compute_flops=True, compute_vjp_flops=True))

def create_train_state(rng, learning_rate, model, x):
    params = model.init(rng, x)
    tx = optax.nadam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def mse_loss(params, batch, model):
    def loss_fn(params):
        reconstruction = model.apply(params, batch)
        return jnp.mean((batch - reconstruction) ** 2)
    return jax.value_and_grad(loss_fn)(params)

@jax.jit
def train_step(state, batch):
    loss, grads = mse_loss(state.params, batch, Autoencoder())
    state = state.apply_gradients(grads=grads)
    return state, loss

# Set training parameters
n_epochs = 100
batch_size = 128
learning_rate = 1e-3

# Initialize model and state
x = jnp.ones((batch_size, TIME_STEPS, 1))
state = create_train_state(rng_state, learning_rate, Autoencoder(), x)

# Training loop
losses = []

for epoch in tqdm(range(n_epochs)):
    epoch_losses = []
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        if x_batch.shape[0] != batch_size:  # Skip the last batch if it is smaller than batch_size
            continue
        state, loss = train_step(state, x_batch)
        epoch_losses.append(loss)
    losses.append(np.mean(epoch_losses))

# Plot the training loss curve
plt.figure(figsize=(9, 4))
plt.semilogy(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(linestyle=":")
plt.legend()
plt.show()



# Function to calculate MAE
def calculate_mae(params, data, model):
    reconstructed = model.apply(params, data)
    mae = jnp.mean(jnp.abs(data - reconstructed), axis=(1, 2))
    return mae

# Calculate MAE for x_train
mae_train = calculate_mae(state.params, x_train, Autoencoder())
treshold = jnp.max(mae_train)

# Calculate MAE for x_full
mae_full = calculate_mae(state.params, x_full, Autoencoder())

# Plot histogram of MAE for x_full
plt.figure(figsize=(12, 4))
plt.hist(mae_full, bins=50, color='skyblue', edgecolor='black')
plt.axvline(treshold, color='red', linestyle='dashed', linewidth=2, label=f'Max MAE for train data = {treshold:.3f}')
plt.xlabel('Mean Absolute Error')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Absolute Error (MAE)')
plt.legend()
plt.grid(linestyle=":")
plt.show()

# Checking how the first sequence is learnt
index = int(jax.random.randint(rng_sample, (), 0, len(x_full)))

plt.figure(figsize=(12, 4))
plt.plot(x_full[index], label="Data")
prediction =  model.apply(state.params, x_full[index])
plt.plot(prediction, label="Reconstruction")
plt.ylabel("Value")
plt.xlabel("")
plt.legend()
plt.grid(linestyle=":")
plt.show()
