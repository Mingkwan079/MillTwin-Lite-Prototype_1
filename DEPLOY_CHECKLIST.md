# MillTwin-Lite deployment checklist

- [ ] Create GitHub repository `MillTwin-Lite`
- [ ] Upload all files from this deploy package to the repository root
- [ ] Confirm `app.py` and `milltwin_pidl_sasz.onnx` are visible at repository root
- [ ] Confirm `requirements.txt` is present
- [ ] Do **not** upload `.streamlit/secrets.toml`, passwords, API keys, or `.env`
- [ ] Open Streamlit Community Cloud and connect GitHub
- [ ] Select repository + `main` branch
- [ ] Set entrypoint to `app.py`
- [ ] Select Python 3.11 when available
- [ ] Deploy
- [ ] Open the generated `.streamlit.app` URL
- [ ] Confirm sidebar says `MODEL STATUS • ONLINE`
- [ ] Run one Forward prediction
- [ ] Run one Inverse search
- [ ] Test Ra QC
- [ ] Test CSV upload in Validation if needed
- [ ] Test the app on a phone and another computer
- [ ] Share the public link
