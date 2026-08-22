# MillTwin-Lite WP3 D6 EndMill Model Card

## Model
Physics-guided deep-learning surrogate for the frozen D=6 mm helical end-mill FSM.

Architecture: `6 -> 48 -> 96 -> 2` with Tanh hidden activations.
The production model uses six continuous engineering inputs. Milling mode is fixed to down milling and is not an inference feature.

## Inputs
- `n_rpm`
- `fz_mm_per_tooth`
- `ap_mm`
- `ae_mm`
- `eps_r_um`
- `eps_a_um`

## Targets
- `Sa_um`
- `Sz_um`

## Frozen tool/FSM physics
- D = 6 mm; R = 3 mm; Z = 4.
- Helix angle beta = 36 deg; flute length = 16 mm.
- Bottom dish angle = 1 deg; milling mode = down.
- Helical phase lag: `psi(z)=2*z*tan(beta)/D`.
- Down-milling engagement: `[pi-phi_c, pi]`, `phi_c=acos(1-2ae/D)`.
- Signed progressive radial/axial runout: `[0, eps, 2eps, 3eps]`.

## Physics-informed design
- `L = L_data + lambda_n L_n + lambda_fz L_fz`.
- `L_n`: matched spindle-speed counterfactual invariance for the ideal geometric FSM; not a real-machining vibration/chatter law.
- `L_fz`: pairwise monotonic penalty, OFF until an exact matched FSM sweep validates it.
- `Ra_th=fz^2/(8R)` is not used as a D6 flat helical end-mill physics law.
- Hard outputs: `Sa=Softplus(u); Sz=2*Sa+Softplus(v)`.

## Split and preprocessing
70/15/15, stratified by `regime` (rigid/moderate), fixed seed 42. `regime` is not a production input. StandardScaler is fit on TRAIN only.

## Test metrics
- Sa_um: R2=0.999105, MAE=0.093685 um, RMSE=0.155730 um, MAPE=5.458%
- Sz_um: R2=0.988686, MAE=0.866263 um, RMSE=1.743002 um, MAPE=10.638%

## Deployment
`11_export_onnx.py` exports a 6-input ONNX model. Input scaling is embedded; app.py sends raw engineering values directly.

## Limitations
- Training targets are FSM simulation outputs; this is a surrogate of the geometric FSM, not direct proof of real-machining accuracy.
- Current physics excludes vibration, chatter, material constitutive behavior and tool wear.
- Current training data contain down milling only; slot/up milling require new data and retraining.
- External FEM/experimental data remain an independent validation gate.
- Predictions outside the recorded D6 DoE domain are extrapolations.

## Machine-readable dossier
See `info.json`.

## Physics diagnostics
See `physics_diagnostics.csv`.