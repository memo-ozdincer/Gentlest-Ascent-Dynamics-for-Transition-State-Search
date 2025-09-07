# gad-ts-search

Fast transition state search with **Gentlest Ascent Dynamics (GAD)** and machine-learned force fields.

This fork builds on the [HORM baseline](https://github.com/deepprinciple/HORM) and adds tooling to:
- Evaluate GAD velocity fields on molecular geometries
- Probe forces, Hessians, and eigenmodes from pretrained checkpoints
- Run Sella optimization with the HORM/EquiformerV2 model as an ASE calculator
- Validate fixed-point behavior via RMSD checks
- Perform dense Hessian frequency analysis (autograd + `torch.linalg.eigh`)

---

## Background

- **Transition states (TS):** high-energy saddle points that connect reactants (R) and products (P).
- **GAD (Gentlest Ascent Dynamics):** a vector field update rule  
  \[
  \dot{x} = -\nabla U(x) + 2 \langle \nabla U(x), v_1(x)\rangle v_1(x)
  \]  
  where \(v_1\) is the eigenvector of the Hessian with the smallest eigenvalue.
- **HORM:** Hessian-Optimized Reactive ML potential, trained to predict energies, forces, and Hessians.
- **RGD1 / Transition1x datasets:** reactive datasets with reactants, products, and TS geometries.

We use pretrained checkpoints (`eqv2.ckpt`, etc.) to predict energies and forces, and currently build Hessians by autograd. Later, we will swap in direct Hessian prediction from the model.

---

## Repo layout

- `scripts/`
  - `probe_force.py` — report force norm for an input `.xyz`
  - `run_gad_horm.py` — HVP-based GAD integration
  - `run_gad_horm_dense.py` — dense Hessian GAD (autograd + eigh)
  - `eval_rmsd.py` — RMSD check between reference and GAD trajectory
  - `freq_analysis.py` — dense Hessian frequency analysis (number of imaginary modes)
  - `ase_calc_horm.py` — ASE calculator wrapper for HORM/EquiformerV2
  - `_opt_ts_sella_once.py` — run a single Sella TS refinement with ML calculator
- `ckpt/` — pretrained checkpoints (`eqv2.ckpt`, etc.)
- `data/rgd1/` — input `.xyz` files and datasets

---
## Usage

### Environment (Colab / local)
```bash
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric ase sella h5py pyyaml
```
Probe force on an XYZ
```bash
python scripts/probe_force.py --cfg equiformer_v2.yml \
  --ckpt ckpt/eqv2.ckpt \
  --xyz data/rgd1/TSguess.xyz --device cuda
```
Run GAD integration
```bash
python scripts/run_gad_horm.py --cfg equiformer_v2.yml --ckpt ckpt/eqv2.ckpt \
  --xyz data/rgd1/TSguess.xyz --steps 5 --dt 0.02 --traj traj_gad.xyz --device cuda
```
Frequency analysis (CPU, dense Hessian)
```bash
python scripts/freq_analysis.py --cfg equiformer_v2.yml \
  --ckpt ckpt/eqv2.ckpt --xyz data/rgd1/TSguess.xyz \
  --device cpu --max_atoms 25 --report 12
```
Sella refinement
```bash
python scripts/_opt_ts_sella_once.py --cfg equiformer_v2.yml --ckpt ckpt/eqv2.ckpt \
  --xyz data/rgd1/TSguess.xyz --out data/rgd1/TSguess_mlts.xyz \
  --device cpu --fmax 1e-3 --steps 300
```

⸻

Current status
	•	Force probing and Hessian analysis works (autograd).
	•	GAD integration runs and converges to small RMSD.
	•	Frequency analysis shows many imaginary modes for raw TS guesses — refinement with Sella is being tested.
	•	Next step: use direct Hessian prediction head (HORM) instead of autograd.

⸻

Citation

If you use this work, please cite:
	•	E & Zhou (2010). Gentlest Ascent Dynamics. Chaos.
	•	Hermes et al. (2022). Sella: An open-source molecular saddle point optimizer. JCTC.
	•	Burger, Rønne, Thiede (2025). Fast Transition State Search by Learning GAD. (preprint)
