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

The notebook's evaluation cell compares the MLP with a Linear Regression baseline fit on the same Phase 2 training split. Both are evaluated on the same held-out test partition using MSE and R². The previously recorded Phase 1 scores used encoded classes and are invalid for redshift; rerun Phase 1 after its correction if you need a cross-phase comparison.

### Learning-rate sensitivity

The training loop evaluates learning rates of `0.01`, `0.001`, and `0.0001` using training loss. Each run uses the same initialization and minibatch order. Inspect the current curves to compare progress and stability; this experiment does not use validation loss to select a rate.

### Optimizer comparison

Vanilla stochastic gradient descent and Adam are compared under the same regression setup. The notebook reports the validation and test MSE calculated for the current run. The two optimizers now start from the same initialization and minibatch order to make the comparison more controlled. The convergence trigger is a plateau heuristic, not proof of reaching a global or local minimum.

### Early stopping

Manual early stopping uses a patience of 30 epochs and a minimum validation-loss improvement threshold of `1e-4`. It restores the best weights selected by validation loss before calculating the final test score. The actual stopping epoch and current validation/test MSE are printed by the notebook when it runs.

## Residual and error analysis

The notebook compares residuals from the Phase 2 linear baseline and Adam-trained MLP, both evaluated on the same test split. It also lists the five largest absolute MLP residuals and compares those samples with the linear baseline. Rerun the notebook to generate current residual plots and sample-level findings.

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

