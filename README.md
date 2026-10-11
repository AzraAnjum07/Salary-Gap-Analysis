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

## PLFS data and reproducibility

The dashboard's PLFS view uses the included cleaned analysis file (`data/processed/clean_earners.csv`), readable code labels (`data/processed/mappings.json`), and the matching GAM outputs in `outputs/`. This lets the hosted app show the same PLFS analysis without rerunning the model at startup.

The original raw microdata file is not included. It is available through the [official MoSPI PLFS 2023–24 catalog](https://microdata.gov.in/NADA/index.php/catalog/213), which requires an account to access the data. If you have authorized access to that file, place `perv1.csv` in `data/raw/` and run the cells in `notebooks/01_data_preparation.ipynb` to recreate the prepared files.

To generate the GAM evaluation and graph data, run:

```bash
python src/gam_model.py
```

The full model fit may take several minutes and needs enough memory. It rewrites the GAM results used by the dashboard. The existing prepared PLFS file and matching model results are already included, so you can run the dashboard directly after installing the requirements.

## Deploy

The app entry point is `dashboard.py`. Deploy that file from the `main` branch on Streamlit Community Cloud. The cleaned PLFS analysis file and matching GAM outputs are included in the repository, so both the original PLFS view and custom CSV upload are available after deployment.

The app supports Streamlit's light and dark themes. Use the app menu's **Settings → Theme** to switch.

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
│   ├── processed/       # cleaned PLFS analysis data and label mappings
│   └── synthetic/       # artificial CSV for testing uploads
├── outputs/             # matching GAM metrics and effect data
└── requirements.txt
```

The raw PLFS source file and virtual environments are kept out of Git. If deploying the app publicly, remember that uploaded files are handled by the hosted app; don't upload confidential company or personal data unless the hosting and access setup is appropriate.
