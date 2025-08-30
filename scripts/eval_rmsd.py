# scripts/eval_rmsd.py
import argparse
import numpy as np
from ase.io import read

def kabsch(P, Q):
    """
    Find optimal rotation R that minimizes ||P R - Q||_F
    P, Q: (N,3) centered coordinates
    returns 3x3 rotation matrix
    """
    C = P.T @ Q
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return R

def rmsd_aligned(P, Q):
    """
    RMSD after optimal superposition (Kabsch).
    P, Q: (N,3)
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.shape == Q.shape and P.shape[1] == 3, "P/Q must be (N,3) and same shape"

    # center
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    # rotate P onto Q
    R = kabsch(Pc, Qc)
    P_rot = Pc @ R

    # RMSD
    diff = P_rot - Qc
    return np.sqrt((diff * diff).sum() / P.shape[0])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",  required=True, help="Reference .xyz (single frame)")
    ap.add_argument("--traj", required=True, help="Trajectory .xyz (multi-frame)")
    ap.add_argument("--index", default="-1", help="Which frame from traj to compare (default: last)")
    args = ap.parse_args()

    ref = read(args.ref)  # single Atoms
    frames = list(read(args.traj, index=":"))
    last = frames[int(args.index)]

    # quick sanity: same number/order of atoms
    if ref.get_number_of_atoms() != last.get_number_of_atoms():
        raise ValueError("Atom count mismatch between ref and traj frame.")

    P = ref.get_positions()
    Q = last.get_positions()

    val = rmsd_aligned(P, Q)
    print(f"RMSD(ref vs traj[{args.index}]) = {val:.6f} Å")

