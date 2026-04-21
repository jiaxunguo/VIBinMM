# Variational Inference for the Bingham Mixture Model (VIBinMM)

Official implementation of the paper:  
**“Variational Inference for the Bingham Mixture Model and Applications”**

---

## ⚙️ Installation

We recommend using Python 3.8+.

Install dependencies:

```bash
pip install numpy scipy scikit-learn matplotlib
```

(Optional) Create a virtual environment:

```bash
python -m venv vibinmm_env
source vibinmm_env/bin/activate
```

## 🧪 Reproducing Experiments

### Option 1: Run the Notebook (Recommended)

The easiest way to reproduce the results is via the Jupyter notebook:

```bash
jupyter notebook notebook.ipynb
```
The notebook includes:

- model implementation
- synthetic data generation
- training pipeline
- repeated experiments (mean ± std)
- result interpretation

### Option 2: Run Python Script
```bash
python train_synthetic_binMM.py
```
Example (from notebook)
```bash
results = run_simulation(
    config_id=1,
    S=10,
    max_iter=1500,
    thred=4e-2,
    init_params='random_from_data'
)
```
This will:

- run 10 independent experiments
- report ACC / ARI / NMI
- output mean ± standard deviation
