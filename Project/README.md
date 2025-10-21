# The $Q$-score: a magnitude-weighted goodness-of-fit score for earthquake forecasting

This project evaluates models based on [https://cseptesting.org/grid-based-forecasts/]. It contains:
- Python files with utility functions.
- Jupyter notebooks with model evaluation and cumulative plots.
- External datasets (to be downloaded separately)

---

## Project Structure

Project/
    
    data/ # Place to store downloaded data (not included)

    notebooks/ # Jupyter notebooks for evaluation
        Cumulative Plots.ipynb
        Q and L-Score Evaluation.ipynb

    src/ # Python functions
        functions_for_hdf5_parallel.py
        QandL.py

    README.md
    requirements.txt #package requirements



---

## Getting Started

1. **Clone the Repository**

```bash
git clone https://github.com/a-arjon/Q-score-a-magnitude-weighted-goodness-of-fit-score-for-earthquake-forecasting.git
cd Project
```

2. **Create and Activate a Virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3.  **Install Dependecies**

``` bash
pip install -r requirenments.txt
```

If pyCSEP cannot be installed via pip, you can use: pip install git+https://github.com/SCECcode/pycsep.git


4.  **Download the Data**

Data is not included due to size. You can download it from:
https://cseptesting.org/grid-based-forecasts/

Place it in the data/ folder.

## Running the Notebook

Open the notebook with Jupyter

```bash
jupyter notebook notebooks/Q and L-Score Evaluation.ipynb
jupyter notebook notebooks/Cumulative Plots.ipynb
```
