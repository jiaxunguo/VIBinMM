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

## 📂 Datasets

### Synthetic Data

The synthetic dataset is provided in:

```python
with open('data/syntheticdata.pkl', 'rb') as f:
    syntheticdata = pickle.load(f)

X = syntheticdata['X']
labels = syntheticdata['y']
```

### Real-World Data

The real-world experiments use two datasets that require permission or manual download from the official sources:

* **Earthquake Data**
  Source: https://www.geonet.org.nz/

* **Gene Expression Data**
  Source: https://www.ncbi.nlm.nih.gov/

Due to access restrictions, these datasets are **not included** in this repository.
Please download the datasets manually and place them under the `data/` directory.

Recommended structure:

```bash
data/
├── syntheticdata.pkl
├── earthquake.pkl
└── gene_expression.pkl
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
