# Neural Redshift Regression with PyTorch

This project extends the PLAsTiCC supernova analysis by using a neural network to predict an astronomical object's spectroscopic redshift from statistical features extracted from multiband light curves.

The complete workflow is available in [`Phase2_485.ipynb`](Phase2_485.ipynb).

## Dataset and features

The notebook loads the `MultimodalUniverse/plasticc` dataset from Hugging Face in streaming mode.

It creates a regression dataset containing 7,848 samples and 20 input features. For each object, the light curve is separated into the `u`, `g`, `r`, `i`, and `z` photometric bands. Four statistical descriptors are extracted from each band:

- Mean flux
- Flux standard deviation
- Maximum flux
- Linear time-series slope

The target is `hostgal_specz`, the spectroscopic redshift used as a distance-related continuous value.

## Data preparation

The data is divided into three reproducible partitions using random seed `42`:

- Training set: 5,493 samples (70%)
- Validation set: 1,177 samples (15%)
- Test set: 1,178 samples (15%)

Feature standardization is fitted only on the training set and then applied to the validation and test sets. This prevents information from the evaluation partitions from leaking into the training process.

## Neural network model

The main model is a fully connected PyTorch multilayer perceptron:

```text
20 inputs → 96 → 64 → 32 → 1 output
```

The hidden layers use ReLU activation functions, and the output layer produces one continuous redshift prediction. The model contains 10,337 trainable parameters and is trained with mean squared error loss.

## Experiments

### Baseline comparison

The notebook compares the neural model with the linear regression baseline from the previous analysis:

| Model | Test MSE | Test R² |
| --- | ---: | ---: |
| Linear Regression baseline | 14.5674 | 0.0081 |
| Baseline MLP | 0.1336 | 0.0334 |

The recorded results show a substantially lower MSE for the MLP, although the low R² indicates that predicting the target remains difficult and that the model does not explain most of the target variance.

### Learning-rate sensitivity

The training loop evaluates learning rates of `0.01`, `0.001`, and `0.0001`. The notebook studies the trade-off between convergence speed and stability. The intermediate learning rate provides a smoother optimization path, while the larger rate converges quickly but shows more fluctuation near the end of training.

### Optimizer comparison

Vanilla stochastic gradient descent and Adam are compared under the same regression setup:

| Optimizer | Convergence trigger | Validation MSE | Test MSE |
| --- | ---: | ---: | ---: |
| Vanilla SGD | Epoch 36 | 0.1096 | 0.1333 |
| Adam | Epoch 38 | 0.0875 | 0.0968 |

Adam achieves the lower validation and test errors. The notebook also notes that the earlier SGD convergence trigger does not necessarily mean better learning; it can indicate that optimization stopped improving prematurely.

### Early stopping

Manual early stopping is evaluated with a patience of 30 epochs and a minimum improvement threshold of `1e-4`. Training stops at epoch 214 after the validation loss fails to improve sufficiently for the configured patience window.

Recorded results:

- Final validation MSE: `0.1119`
- Held-out test MSE: `0.1362`

## Residual and error analysis

The notebook compares the residual distributions of the linear baseline and the best Adam-trained MLP. The MLP produces a tighter concentration of errors around zero, while the linear model has a wider spread and more extreme deviations.

It also identifies the five worst MLP predictions. These difficult samples are associated with very distant objects: the true target values are high, while the model consistently predicts much smaller redshift values. The analysis finds that many of these difficult samples are also the same observations that caused large errors for the linear baseline.

## Technologies

- Python
- Jupyter Notebook
- PyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Hugging Face Datasets

## Running the notebook

Install the required packages:

```bash
pip install datasets==3.6.0 pandas numpy matplotlib seaborn scikit-learn torch
pip install git+https://github.com/MultimodalUniverse/MultimodalUniverse.git
```

Open `Phase2_485.ipynb` in Jupyter or Google Colab and run the cells from top to bottom. An internet connection is required to download the dataset from Hugging Face. A Hugging Face token is optional for this public dataset but may provide higher request limits.

## Project status

This is an academic machine-learning experiment. The notebook documents model development and evaluation but is not packaged as a reusable training pipeline or production inference service.

