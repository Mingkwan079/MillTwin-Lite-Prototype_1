# MillTwin-Lite

Physics-informed CNC model for Sa/Sz prediction, inverse parameter search, and surface validation.

## Included files

- `app.py` — Streamlit UI/application
- `logo.png` — Millcore logo used by the UI
- `milltwin_pidl_sasz.onnx` — deployed Sa/Sz ONNX model
- `info.json` — model metadata/design domain
- `model_metrics.csv` — model performance data used in Dossier
- `physics_diagnostics.csv` — physics diagnostics used in Dossier
- `ablation_results.csv` — optional evidence table
- `model_card.md` — model documentation
- `requirements.txt` — cloud runtime dependencies only
- `.streamlit/config.toml` — Streamlit theme/server defaults
- `.gitignore` — prevents training outputs/secrets from being pushed accidentally

## Run locally

Use Python 3.11 if possible.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Publish so other people can use it

### 1. Create a GitHub repository

Create a new repository, for example:

`MillTwin-Lite`

Upload **the contents of this folder** to the repository root. `app.py` should be at the top level, not inside another nested folder.

Expected GitHub layout:

```text
MillTwin-Lite/
├── .streamlit/
│   └── config.toml
├── .gitignore
├── app.py
├── logo.png
├── milltwin_pidl_sasz.onnx
├── info.json
├── model_metrics.csv
├── physics_diagnostics.csv
├── ablation_results.csv
├── model_card.md
├── requirements.txt
└── README.md
```

### 2. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **Create app / Deploy app**.
3. Select the GitHub repository containing these files.
4. Branch: `main`.
5. Main file path / entrypoint: `app.py`.
6. In advanced settings, use **Python 3.11** if the option is available.
7. Deploy.

After the build finishes you will receive a public link similar to:

`https://your-app-name.streamlit.app`

Send that link to anyone you want to demo the application to. They only need a browser.

## Important demo limitations

- The deployed ONNX model predicts `Sa` and `Sz` only within its documented design scope.
- `Ra QC` checks a measured Ra value against a selected limit; it does **not** infer Ra from Sa/Sz.
- Prediction History currently uses Streamlit session state. It is a temporary per-session log, not a persistent user database.
- G-code generation is intentionally excluded from the current product scope.
- Do not present model output as a substitute for machine qualification or experimental validation.

## Public vs private deployment

If the GitHub repository/app is public, visitors can use the app without installing anything. Keep in mind that files stored in a public GitHub repository—including the ONNX model—are visible/downloadable from that repository.

If the ONNX model should remain private, use a private repository and restrict app access, or move inference to a private backend in a later production architecture.

## Updating the app later

Edit files locally, commit/push changes to the same GitHub repository, and the hosted Streamlit app can rebuild/redeploy from that repository. Keep model filenames stable unless you also update `app.py`.
