# MillTwin-Lite Round-3 deployment checklist

- [ ] Upload the contents of this folder to the GitHub repository root
- [ ] Confirm `app.py` is at repository root
- [ ] Confirm `milltwin_pidl_sasz.onnx` is at repository root
- [ ] Confirm `info.json` describes Trial 86 / 96-32-128 / D6-Z4-45deg
- [ ] Confirm `requirements.txt` is present
- [ ] Do not upload `.env`, API keys or `.streamlit/secrets.toml`
- [ ] Connect the repository to Streamlit Community Cloud
- [ ] Branch = `main`
- [ ] Entrypoint = `app.py`
- [ ] Prefer Python 3.11
- [ ] Confirm model status is ONLINE
- [ ] Forward test: PASS case shows white Sa/Sz bars with green outlines
- [ ] Forward test: REVIEW case shows white Sa/Sz bars with red outlines
- [ ] Confirm inverse TOP CANDIDATE values are readable in dark navy text
- [ ] Run Inverse Search and export CSV
- [ ] Test input-domain warning
- [ ] Test Ra QC if used in demo
- [ ] Verify app on desktop and mobile
