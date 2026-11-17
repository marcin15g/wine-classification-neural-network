# Wine classification with a neural network from scratch

A feed-forward neural network written from scratch in NumPy (no ML framework)
that classifies wines into three cultivars from their chemical analysis.
Built in 2020 to explore backpropagation and learning-rate strategies.

## Dataset

The classic [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine)
(`wine.data`, described in `wine.names`): 178 wines, 13 features such as
alcohol, flavanoids, color intensity and proline, and 3 classes.

## What's implemented

In [`neural_network.py`](neural_network.py):

- One hidden layer with sigmoid activations and a one-hot output layer
- Forward propagation, cross-entropy cost and backpropagation written by hand
- Weight and bias updates with gradient descent
- **Adaptive learning rate**: the rate grows while the cost keeps falling and
  shrinks when the cost rises by more than a set ratio

In [`main.py`](main.py), experiments that:

- Train and evaluate the network over many random train/test splits
- Sweep hidden-layer sizes and epoch counts and plot accuracy as a 3D surface
- Compare constant and adaptive learning rates and tune the adaptive
  increment/decrement factors

## Results

With the default settings (14 hidden neurons, 850 epochs, learning rate 0.01),
the network trains on only **20%** of the data and is tested on the remaining
80%. Over 30 random splits:

| | Accuracy |
|---|---|
| Mean on held-out test data | **95.3%** |
| Lowest on held-out test data | 90.2% |
| Training data | 97–100% |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy "pandas<2.3" scikit-learn matplotlib seaborn
python main.py
```

`main.py` runs the 30-split evaluation by default; the other experiments are
at the bottom of the file and can be switched on by uncommenting them. The
plotting experiments save figures to `plots/`, so create that folder first.
