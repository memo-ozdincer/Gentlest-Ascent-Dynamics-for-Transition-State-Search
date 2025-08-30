# scripts/run_gad_horm.py
import argparse, yaml, torch
from pathlib import Path
from ase.io import read, write
from torch_geometric.data import Data as TGData, Batch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

# utils
def load_model(cfg_path, ckpt_path, device):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = EquiformerV2_OC20(**cfg["model"])
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model

def atoms_to_pyg(atoms):
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    z   = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    n   = int(z.numel())
    data = TGData(pos=pos, z=z, natoms=n)
    data.batch = torch.zeros(n, dtype=torch.long)  # single-graph batch ids
    return data

def get_energy(model, batch):
    """Return scalar energy (sum over graphs). Handles (tuple/dict/single) outputs."""
    out = model(batch)
    if isinstance(out, dict):
        if "energy" in out and out["energy"] is not None:
            E = out["energy"]
        elif "energies" in out and out["energies"] is not None:
            E = out["energies"]
        else:
            raise RuntimeError("Model output dict missing 'energy'/'energies'. Keys: " + str(list(out.keys())))
        return E.sum()
    elif isinstance(out, (list, tuple)):
        # HORM EquiformerV2_OC20 returns (energy, forces)
        return out[0].sum()
    else:
        # single tensor
        return out.sum()

# ---- Hessian-vector products & lowest eigenvector via power iteration --
def hvp_energy(E_fn, x, v):
    """Compute H(x) @ v using Pearlmutter’s trick; v is (3N,) or (N,3)."""
    x = x.detach().clone().requires_grad_(True)

    # scalar energy
    E = E_fn(x)

    # first gradient (N,3)
    (g,) = torch.autograd.grad(E, x, create_graph=True)
    g_flat = g.reshape(-1)              # (3N,)

    # grad_outputs must match g_flat's shape (3N,)
    v_flat = v.reshape(-1)
    Hv = torch.autograd.grad(g_flat, x, grad_outputs=v_flat, retain_graph=False)[0]
    return Hv.reshape_as(x)             # (N,3)

def lowest_eigvec_via_power(E_fn, x, iters=12):
    """Power iteration on -H to obtain the lowest-eigenvector of H."""
    N3 = x.numel()
    v = torch.randn(N3, device=x.device, dtype=x.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        Hv = hvp_energy(E_fn, x, v.reshape_as(x)).reshape(-1)
        w  = -Hv  # dominant eigvec of -H -> lowest eigenvector of H
        v  = w / (w.norm() + 1e-12)
    return v  # (3N,)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",   required=True)   # e.g. equiformer_v2.yml
    ap.add_argument("--ckpt",  required=True)   # e.g. ckpt/eqv2.ckpt
    ap.add_argument("--xyz",   required=True)   # TS guess .xyz
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--dt",    type=float, default=0.02)
    ap.add_argument("--traj",  default="traj_gad.xyz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model  = load_model(args.cfg, args.ckpt, device)

    # build one-graph batch
    atoms0 = read(args.xyz)
    data   = atoms_to_pyg(atoms0)
    batch  = Batch.from_data_list([data]).to(device)
    # add one graph-level 'ae' or else it crashes
    batch.ae = torch.zeros(batch.num_graphs, device=device, dtype=torch.float32)

    frames = []
    for _ in range(args.steps):
        print(f"[GAD] step {si+1}/{args.steps}", flush=True)
        # positions as leaf with grad
        x = batch.pos.detach().clone().requires_grad_(True)
        batch.pos = x  # DO NOT create a new Batch; reuse same batch so .ae stays

        # closure for energy at current x
        def E_fn(inp):
            batch.pos = inp
            return get_energy(model, batch)

        # F = negative grad E
        E = E_fn(x)
        (grad_pos,) = torch.autograd.grad(E, x, create_graph=False)
        F = -grad_pos.reshape(1, -1)  # (1, 3N)

        # lowest eigvec via HVP power iteration (no dense Hessian)
        v = lowest_eigvec_via_power(E_fn, x, iters=12).reshape(1, -1)
           # Rayleigh quotient = curvature along v
        Hv = hvp_energy(E_fn, x, v.reshape(-1)).reshape(-1)
        rq = (v.reshape(-1) @ Hv) / (v.reshape(-1) @ v.reshape(-1))

        print(f"   Energy={E.item():.6f}, |F|={F.norm().item():.3e}, λ_min≈{rq.item():.6f}")
        # GAD step: -F + 2 (F·v) v
        dot = (F * v).sum(dim=1, keepdim=True)
        dx  = (-F + 2.0 * dot * v).reshape(-1, 3)

        # Euler update
        batch.pos = (x + args.dt * dx).detach()

        frames.append(batch.pos.detach().cpu().numpy())

    # write trajectory
    from ase import Atoms
    Z = atoms0.get_atomic_numbers()
    traj_atoms = [Atoms(numbers=Z, positions=P, cell=atoms0.cell, pbc=atoms0.pbc) for P in frames]
    write(args.traj, traj_atoms)
    print(f"Wrote {len(traj_atoms)} frames to {args.traj}")

if __name__ == "__main__":
    main()

