"""
quick_extract.py  <path_to_folder_with_out_and_xyz>

Creates descriptors.csv with:
ligand, vbur_aqua, sigma_e2, hbonds, delta_vbur, dE_bind
"""

import pathlib                # needed before DEFAULT_FOLDER is defined
VERBOSE = True  # set to False once things work
# Default data folder used when no CLI argument is given
DEFAULT_FOLDER = pathlib.Path(
    "/Users/thomasquaid/Library/CloudStorage/OneDrive-Personal/Documents/OrcaWS/Hephaestus/Test_set_working"
)
DIAG = True   # set False when troubleshooting is finished
# ---------- optional external libraries ---------- #
import sys, pathlib, re, joblib, pandas as pd
import subprocess, json, tempfile, shutil, os
import math, itertools
from cclib import io as ccopen

# ---- robust text reader (handles UTF‑8, UTF‑16 BOM, latin‑1) ----
def read_text_safe(path):
    """Return file content as str, trying UTF‑8 then latin‑1, finally utf‑16."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-16")

SAMBVCA_EXE = shutil.which("SambVca") or os.getenv("SAMBVCA_EXE")
if SAMBVCA_EXE is None and VERBOSE:
    print("[WARN] SambVca binary not found on PATH or SAMBVCA_EXE; %Vbur will be NaN")

try:
    import mdtraj as md                                    # H‑bond counts
except ImportError:
    md = None
md = None  # force-disable H-bond descriptor until topology issue is resolved
# ---------- descriptor helper functions ---------- #
# ---------------- fallback steric metric (pure‑Python) ---------------- #
import numpy as np
_VDW = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
        'P': 1.80, 'S': 1.80, 'Zn': 1.39}

def _golden_sphere(n=7000):
    """Return n quasi‑uniform points on the unit sphere (golden spiral)."""
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i + 0.5)/n)
    theta = np.pi * (1 + 5**0.5) * (i + 0.5)
    return np.vstack((np.cos(theta)*np.sin(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(phi))).T
_SPHERE = _golden_sphere()  # cached

def percent_buried(xyz):
    """
    Compute %Vbur with the original SambVca 2.1 Fortran binary (no CLI flags).
    It creates a temporary *.inp* file understood by SambVca, runs the program,
    and reads *sambvca.json* that the code writes by default.

    Returns NaN if the binary or JSON file is missing, or if any error occurs.
    """
    exe = SAMBVCA_EXE  # resolved at import time
    if exe is None or not xyz.exists():
        return float("nan")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            inp_file = tmpdir / "job.inp"

            # Minimal five‑line input: xyz path, center atom, sphere Å, vdW set, JSON flag
            inp_file.write_text(
                f"{xyz}\n"
                "Zn\n"
                "3.5\n"
                "bondi\n"
                "json\n"
            )

            # SambVca 2.1 takes the input filename as its sole argument
            subprocess.run(
                [exe, str(inp_file)],
                cwd=tmpdir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            out_json = tmpdir / "sambvca.json"
            if not out_json.exists():
                raise FileNotFoundError("sambvca.json not produced")

            data = json.loads(out_json.read_text())
            return data.get("percent_buried", float("nan"))

    except Exception as err:
        if VERBOSE:
            print(f"[ERR ] SambVca v2.1 failed on {xyz.name}: {err}")
        # fallback: compute purely in NumPy
        try:
            atoms, coords = [], []
            with xyz.open() as fh:
                _ = fh.readline(); _ = fh.readline()       # skip header
                for ln in fh:
                    p = ln.split()
                    if len(p) < 4: continue
                    atoms.append(p[0]); coords.append(list(map(float, p[1:4])))
            atoms = np.array(atoms)
            coords = np.array(coords)
            Zn = coords[atoms == "Zn"][0]
            pts = _SPHERE*3.5 + Zn
            d2 = ((pts[:, None, :] - coords[None, :, :])**2).sum(-1)
            rad2 = np.array([_VDW.get(a, 1.7)**2 for a in atoms])
            blocked = (d2 <= rad2).any(1)
            return blocked.mean()*100
        except Exception:
            return float("nan")


def sigma_donation(out: pathlib.Path) -> float:
    """
    Sum of E(2) donor→Zn terms (kcal mol⁻¹) in **all** ‘SECOND ORDER
    PERTURBATION THEORY ANALYSIS …’ tables.  Works with ORCA 6 where the
    tables are split into “from unit  X to unit  Y” blocks and the token
    ‘E(2)=’ is absent.

    Parsing strategy:
      • locate the header line once;
      • for every subsequent line that contains “ Zn”,
        take the third‑to‑last numeric column (the E(2) value)
        and add it to the running total.

    Returns NaN if no Zn rows are found.
    """
    try:
        txt = read_text_safe(out)
        hdr = re.search(r"SECOND\s+ORDER\s+PERTURBATION\s+THEORY\s+ANALYSIS",
                        txt, re.I)
        if not hdr:
            return math.nan

        e2_total = 0.0
        float_rx = re.compile(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?")
        for ln in txt[hdr.end():].splitlines():
            if "Zn" not in ln:                   # accept Zn with or without space
                continue
            nums = float_rx.findall(ln)
            if len(nums) < 3:
                continue
            e2_total += float(nums[-3])          # E(2) is third‑to‑last numeric
        return e2_total if e2_total else math.nan

    except Exception as err:
        if VERBOSE:
            print(f"[ERR] sigma_donation in {out.name}: {err}")
        return math.nan

def hbond_count(xyz):
    """Return number of H‑bonds to the aqua ligand; NaN if mdtraj unavailable."""
    if md is None:
        return float("nan")
    try:
        t = md.load_xyz(str(xyz), top=None)
        hb = md.baker_hubbard(t, freq=0.0)           # default criteria
        return len(hb)
    except Exception as err:
        if VERBOSE:
            print(f"[ERR ] mdtraj failed on {xyz.name}: {err}")
        return float("nan")

# ---------- main driver ---------- #
def final_energy(out: pathlib.Path) -> float:
    """
    Return the last ‘FINAL SINGLE POINT ENERGY’ value (Hartree).
    Matches plain or scientific notation; ignores trailing units.
    """
    vals = re.findall(
        r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-]?\d+\.\d+(?:[Ee][-+]?\d+)?)",
        read_text_safe(out)
    )
    return float(vals[-1]) if vals else math.nan

def process(stem, folder):
    aqua_out   = folder / f"{stem}_H2O.out"
    aqua_xyz   = folder / f"{stem}_H2O.xyz"
    bicarb_xyz = folder / f"{stem}_HCO3.xyz"

    row = dict(ligand=stem)
    row["vbur_aqua"]  = percent_buried(aqua_xyz)
    row["sigma_e2"]   = sigma_donation(aqua_out)
    row["hbonds"]     = hbond_count(aqua_xyz)
    row["delta_vbur"] = percent_buried(bicarb_xyz) - row["vbur_aqua"]
    # Binding energy will be calculated separately; leave placeholder
    row["dE_bind"] = float("nan")

    # ---------- DIAGNOSTICS ----------
    if DIAG:
        print(f"\n--- DIAG {stem} ---")
        print(f"  vbur_aqua  : {row['vbur_aqua']}")
        print(f"  sigma_e2   : {row['sigma_e2']}")
        print(f"  dE_bind    : {row['dE_bind']}")

    return row

def main():
    # ---------------- resolve folder ----------------
    if len(sys.argv) > 1:
        folder = pathlib.Path(sys.argv[1]).expanduser().resolve()
    else:
        folder = DEFAULT_FOLDER.resolve()
        if VERBOSE:
            print(f"[INFO ] No folder argument; using DEFAULT_FOLDER = {folder}")

    if not folder.exists():
        sys.exit(f"[ERR ] Folder {folder} does not exist.")

    if VERBOSE:
        print(f"[DEBUG] Scanning folder: {folder}")
        print(f"[DEBUG] Exists? {folder.exists()} | Contains {len(list(folder.iterdir()))} items")

    rows = []
    aqua_files = list(folder.glob("*_H2O.out"))
    if VERBOSE:
        print(f"[DEBUG] Found {len(aqua_files)} *_H2O.out files")
        for f in aqua_files[:10]:
            print(f"        > {f.name}")

    for aqua in aqua_files:
        stem = re.sub("_H2O.out$", "", aqua.name)
        hco3_out = folder / f"{stem}_HCO3.out"
        axyz = folder / f"{stem}_H2O.xyz"
        bxyz = folder / f"{stem}_HCO3.xyz"

        if not hco3_out.exists():
            if VERBOSE:
                print(f"[WARN] Missing {hco3_out.name}; skipping {stem}")
            continue
        if not (axyz.exists() and bxyz.exists()):
            if VERBOSE:
                print(f"[WARN] Missing xyz for {stem}; skipping")
            continue
        rows.append(process(stem, folder))

    df = pd.DataFrame(rows)
    df.to_csv("descriptors.csv", index=False)
    print("✅  Saved descriptors.csv to project directory")

    if VERBOSE:
        print(f"[DEBUG] Wrote {len(df)} rows to descriptors.csv")

if __name__ == "__main__":
    main()