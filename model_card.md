# MillTwin-Lite WP3 D6 EndMill Model Card

## Model identity
Protocol: `WP3-D6-v2-locked-test`.
Production architecture: `6 -> 96 -> 32 -> 128 -> 2`, Tanh hidden layers, hard roughness output constraints.
Locked Optuna source: Optuna Trial 86 from user PowerShell log, 2026-08-23.
Locked hyperparameters: lr=0.00116003380633, weight_decay=5.41257120315e-05, lambda_n=0.498200694036, dropout=0, batch=128.

## Inputs / outputs
Inputs: `n_rpm, fz_mm_per_tooth, ap_mm, ae_mm, eps_r_um, eps_a_um`.
Outputs: `Sa_um, Sz_um`.
Milling mode is fixed to down milling and is not an inference feature. `regime` is split/reporting metadata only.

## Fixed tool and FSM scope
Sandvik `1P341-0600-XA 1630`: D=6 mm, Z=4, helix=45 deg, usable length=13 mm, chamfer=0.1 mm x 45 deg.
Bottom-dish angle 2 deg is an effective FSM calibration parameter, not a Sandvik catalog value.

## Data protocol
500 transferred legacy FSM-response rows are reparameterized to the current D6 operating domain.
Frozen split: 350 train / 75 validation / 75 final test. Development = train + validation = 425 rows.
5-fold CV and all HPO/physics selection operate only inside development. Scalers are fit inside each fold; final scalers are fit on all 425 development rows.
Historical caveat: metrics on the same 75-row test had been displayed by an earlier project version. The v2 code does not train/tune on those rows, but a completely blind publication-grade test should use a new external holdout.

## Physics loss
`L_total = L_data + lambda_n L_n + lambda_fz L_fz`.
`L_n` enforces spindle-speed invariance for the ideal geometry-only FSM at fixed fz/ap/ae/runout.
`L_fz` is OFF because no exact matched FSM counterfactual sweep has been supplied; nearest-neighbour LHS evidence is diagnostic only.
Hard outputs enforce `Sa > 0` and `Sz >= 2 Sa`.

## 5-fold CV cross-check of locked Trial 86
- Sa_um: MAPE=11.133% +/- 3.118%, R2=0.9780 +/- 0.0077.
- Sz_um: MAPE=5.903% +/- 1.903%, R2=0.9951 +/- 0.0007.

## Final-test metrics
- Sa_um: R2=0.965158, MAE=0.087587 um, RMSE=0.126227 um, MAPE=9.550%, WMAPE=9.330%.
- Sz_um: R2=0.995078, MAE=0.171158 um, RMSE=0.215377 um, MAPE=4.739%, WMAPE=3.512%.

## Physics ablation
- A_data_only_same_arch / Sa_um: mean CV MAPE=13.313%.
- A_data_only_same_arch / Sz_um: mean CV MAPE=7.562%.
- B_data_plus_Ln / Sa_um: mean CV MAPE=11.133%.
- B_data_plus_Ln / Sz_um: mean CV MAPE=5.903%.

## Deployment
ONNX max abs parity delta=9.53674e-07 um; p95 CPU latency=0.0294 ms.

## Inverse-problem limitation
The app and packaged solver can search the PIDL surrogate. Exact re-simulation of arbitrary inverse candidates against the original Sa/Sz FSM is not available in this transfer package because the callable legacy Sa/Sz solver is absent. Nearest observed FSM rows are references only, not H4 verification.

## External validation
Experimental/FEM data remain an independent validation gate. Do not present surrogate-to-FSM accuracy as real-machining accuracy.