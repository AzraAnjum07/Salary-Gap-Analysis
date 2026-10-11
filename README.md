# Gender-Based Earnings Gap Analysis

This project looks at differences in earnings between male and female workers in India's Periodic Labour Force Survey (PLFS) 2023–24 data. It combines straightforward comparisons with a Generalized Additive Model (GAM) to show how the observed gap changes after accounting for age, education, employment status, industry, and occupation.

The goal is to explain patterns in the data, not to predict any one person's salary or claim that gender or another characteristic causes an earnings difference.

## Dashboard

The Streamlit dashboard has three sections:

- **EDA** compares actual earnings and worker characteristics.
- **GAM Analysis** shows adjusted patterns, model checks, and plain-language graph explanations.
- **Conclusions** summarizes the main descriptive and adjusted comparisons.

You can also upload a CSV and choose its earnings, gender, numeric, and category columns for an exploratory analysis. The included sample is entirely synthetic and is only for trying this feature.

## Run it locally

Use Python 3.10 or newer. From the project folder:

```bash
python -m venv .venv
```

Activate the environment, then install the packages and start the dashboard:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux / WSL
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

To stop the dashboard, press `Ctrl+C` in the terminal.

## Use the original PLFS data

The raw PLFS file and prepared data are not included in this repository. The raw data is ignored by Git. If you have authorized access to the PLFS 2023–24 file, place `perv1.csv` in `data/raw/`, then run the cells in `notebooks/01_data_preparation.ipynb`. That notebook creates the prepared files used by the original-data dashboard.

To generate the GAM evaluation and graph data, run:

```bash
python src/gam_model.py
```

The full model fit may take several minutes and needs enough memory. Once preparation and model generation finish, run the dashboard as shown above. Without those local data and model files, use **Upload Custom Dataset** to explore a CSV.

## Try the upload with sample data

Upload [`data/synthetic/synthetic_plfs_test.csv`](data/synthetic/synthetic_plfs_test.csv). It contains 6,000 artificial rows with earnings, gender, age, formal education years, employment status, industry, and occupation. It does not represent PLFS respondents or any real workforce, so its results are not evidence about actual earnings differences.

## Validation approach

The original-data GAM script reserves 20% of the prepared sample for a final test, runs 5-fold cross-validation on the remaining training partition, then evaluates the final model on the untouched test partition. Its purpose is adjusted association analysis. A high or low model score does not establish a cause-and-effect relationship.

## Repository contents

```text
.
├── dashboard.py
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_eda.ipynb
├── src/
│   ├── gam_model.py
│   └── upload_dashboard.py
├── data/
│   └── synthetic/
└── requirements.txt
```

Raw and processed PLFS files, generated model outputs, and virtual environments are kept out of Git. If deploying the app publicly, remember that uploaded files are handled by the hosted app; don't upload confidential company or personal data unless the hosting and access setup is appropriate.
