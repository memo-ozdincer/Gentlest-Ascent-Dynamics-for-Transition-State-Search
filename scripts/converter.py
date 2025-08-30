# scripts/extract_ts_xyz.py
import h5py, argparse
from ase import Atoms, io

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", help="Path to RGD1_CHNO.h5", default="rgd1/RGD1_CHNO.h5")
    ap.add_argument("--group", default=None, help="Specific group (e.g. MR_115883_0). If omitted, pick first with TSG.")
    ap.add_argument("--out", default="rgd1/TSguess.xyz")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        gname = args.group
        if gname is None:
            # pick the first group that has TSG
            for name, obj in f.items():
                if isinstance(obj, h5py.Group) and "TSG" in obj:
                    gname = name
                    break
            if gname is None:
                raise RuntimeError("No group with TSG found in the file.")

        g = f[gname]
        Z   = g["elements"][...]          # (N,)
        pos = g["TSG"][...]               # (N,3)

    atoms = Atoms(numbers=Z, positions=pos)
    io.write(args.out, atoms)
    print(f"Wrote {args.out} from group {gname}")

if __name__ == "__main__":
    main()


