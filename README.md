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
