# MillTwin-Lite — Streamlit deployment package (Round 3 / Trial 86)

This folder is the lightweight GitHub/Streamlit deployment extracted from the locked WP3 Round-3 package.
It contains the production ONNX surrogate and only the runtime files needed by the public demo.

## Production model

- Model: PIDL Trial 86 final retrain
- Inputs: `n_rpm, fz_mm_per_tooth, ap_mm, ae_mm, eps_r_um, eps_a_um`
- Outputs: `Sa_um, Sz_um`
- Architecture: `96 → 32 → 128`
- Development set: 425 samples
- Frozen final test: 75 samples
- Final-test MAPE: Sa `9.5505%`, Sz `4.7386%`
- Tool: Sandvik `1P341-0600-XA 1630`, D=6 mm, Z=4, helix=45°, flute length=13 mm

The final-test reference values are FSM/simulation targets, not direct experimental validation.

## Included files

- `app.py` — Streamlit UI
- `logo.png` — Millcore logo
- `milltwin_pidl_sasz.onnx` — production Trial-86 ONNX model
- `info.json` — model metadata, input domain and final metrics
- `model_metrics.csv` — model performance table
- `physics_diagnostics.csv` — physics diagnostics
- `ablation_results.csv` — MLP/PIDL ablation evidence
- `model_card.md` — model documentation
- `requirements.txt` — runtime-only dependencies
- `.streamlit/config.toml` — Streamlit configuration
- `.gitignore` — excludes development/training artifacts and secrets

## UI changes in this deploy build

- Inverse Search top-candidate text uses dark navy text on the light result card for readability.
- Forward Sa/Sz bars use white fill with a status outline:
  - green border = PASS
  - red border = REVIEW
- The plot frame follows the same PASS/REVIEW status color.

These UI changes do not alter the ONNX model, scaler logic, prediction equations, input domain or trained weights.

## Run locally

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## GitHub layout

Upload the **contents of this folder** to the repository root:

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
├── DEPLOY_CHECKLIST.md
└── README.md
```

## Deploy on Streamlit Community Cloud

1. Push these files to GitHub.
2. Create a Streamlit Community Cloud app from the repository.
3. Branch: `main`.
4. Entry point: `app.py`.
5. Use Python 3.11 when available.
6. Deploy and run one Forward prediction and one Inverse Search.

## Scope / limitations

- The deployed model predicts Sa and Sz only.
- Current model scope is the documented D6 down-milling design domain in `info.json`.
- `Ra QC` is a separate measured/manual quality-control path; the ONNX model does not infer Ra.
- Do not present the frozen-test error as experimental machining error.
- Public GitHub repositories expose the ONNX file for download. Use a private repository/backend if the model must remain private.
