#!/usr/bin/env python3
__version__ = "0.1.0"
"""
descriptor_extract.py  <folder_with_out_and_xyz>

Creates descriptors.csv with the columns:
ligand, vbur_aqua, sigma_e2, zn_q, hbonds, delta_vbur
(dE_bind is left NaN so you can fill it in separately.)

• Works with ORCA 6 NBO output (“SECOND ORDER PERTURBATION THEORY …” blocks).
• Falls back to a pure-NumPy %Vbur if the SambVca 2.1 binary isn’t working.
• Prints a one-page diagnostic per ligand while you’re still debugging.

Author: (2025-06-24)  — drop-in replacement for quick_extract.py
"""

# ---------- standard library ----------
import sys, os, re, json, math, tempfile, subprocess, shutil, itertools, pathlib
import argparse
from   pathlib import Path
from typing import List

DEFAULT_DIR = pathlib.Path("/Users/thomasquaid/Library/CloudStorage/OneDrive-Personal/Documents/OrcaWS/Hephaestus/Test_set_working")

# ---------- third-party ----------
import numpy       as np
import pandas      as pd
try:                      import mdtraj as md  # for hydrogen-bond counting
except ImportError:       md = None

# ⇣⇣⇣ ---------- USER SETTINGS ---------- ⇣⇣⇣
VERBOSE      = True          # lots of debug noise
DIAG         = False          # prints per-ligand summary
SAMBVCA_EXE  = shutil.which("SambVca") or os.getenv("SAMBVCA_EXE")
if SAMBVCA_EXE is None and VERBOSE:
    print("[WARN] SambVca binary not found — will use NumPy fallback for %Vbur")

# ---------- helpers ----------
def read_text_safe(path: Path) -> str:
    """Return file text, trying UTF-8 → latin-1 → utf-16."""
    for enc in ("utf-8", "latin-1", "utf-16"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise

# ---- %Vbur (forbidden-cone)  ----
_VDW = {'H':1.20,'C':1.70,'N':1.55,'O':1.52,'F':1.47,'P':1.80,'S':1.80,'Zn':1.39}
_golden_cache = None
def _sphere_pts(n:int=7000) -> np.ndarray:
    global _golden_cache
    if _golden_cache is None:
        i = np.arange(n)
        phi   = np.arccos(1 - 2*(i+0.5)/n)
        theta = np.pi*(1+5**0.5)*(i+0.5)
        _golden_cache = np.vstack((np.cos(theta)*np.sin(phi),
                                   np.sin(theta)*np.sin(phi),
                                   np.cos(phi))).T
    return _golden_cache

def percent_buried(xyz: Path, center: str="Zn", r:float=3.5) -> float:
    """Try SambVca; fall back to analytic method."""
    if not xyz.exists():                 return math.nan
    if SAMBVCA_EXE:
        try:
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                inp = td/"job.inp"
                inp.write_text(f"{xyz}\n{center}\n{r}\nbondi\njson\n")
                subprocess.run([SAMBVCA_EXE, str(inp)], cwd=td,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=True)
                js = json.loads((td/"sambvca.json").read_text())
                return js.get("percent_buried", math.nan)
        except Exception as e:
            if VERBOSE: print(f"[ERR] SambVca on {xyz.name}: {e}")
    # --- pure NumPy backup ---
    try:
        coords, elems = [], []
        with xyz.open() as fh:
            _ = fh.readline(); _ = fh.readline()
            for ln in fh:
                t = ln.split();   elems.append(t[0]);   coords.append(list(map(float,t[1:4])))
        elems  = np.array(elems)
        coords = np.array(coords)
        C      = coords[elems==center][0]
        pts    = _sphere_pts()*r + C
        d2     = ((pts[:,None,:]-coords[None,:,:])**2).sum(-1)
        rad2   = np.array([_VDW.get(e,1.7)**2 for e in elems])
        return (d2<=rad2).any(1).mean()*100
    except Exception as e:
        if VERBOSE: print(f"[ERR] fallback %Vbur on {xyz.name}: {e}")
        return math.nan

# ---- σ-donation (Σ E(2) donor → Zn) ----
float_re = re.compile(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?")
def sigma_donation(out: Path) -> float:
    """Sum all E(2) values from donor→Zn rows (ORCA 6 style)."""
    if not out.exists():                 return math.nan
    try:
        txt = read_text_safe(out)
        mo  = re.search(r"SECOND\s+ORDER\s+PERTURBATION", txt, re.I)
        if not mo:                       return math.nan
        total = 0.0
        for ln in txt[mo.end():].splitlines():
            if "Zn" not in ln:           continue
            nums = float_re.findall(ln)
            if len(nums) >= 3:
                total += float(nums[-3])
        return total if total else math.nan
    except Exception as e:
        if VERBOSE: print(f"[ERR] σE2 in {out.name}: {e}")
        return math.nan

# ---- Zn natural charge (NBO) ----
_npa_re = re.compile(
    r"Natural\s+Population\s+Analysis.*?\bZn\b\s+\d+\s+([-+.\dEe]+)",
    re.S)

def zn_charge(out: Path) -> float:
    """Return Zn natural charge from ORCA‑6 NBO 'Natural Population Analysis' block."""
    if not out.exists():
        return math.nan
    try:
        txt = read_text_safe(out)
        m = _npa_re.search(txt)
        return float(m.group(1)) if m else math.nan
    except Exception as e:
        if VERBOSE:
            print(f"[ERR] Zn q in {out.name}: {e}")
        return math.nan

# ---- H-bond count (optional) ----
def hbonds(xyz: Path) -> float:
    if md is None or not xyz.exists():   return math.nan
    try:
        traj = md.load_xyz(str(xyz), top=None)
        return len(md.baker_hubbard(traj, freq=0.0))
    except Exception as e:
        if VERBOSE: print(f"[ERR] mdtraj on {xyz.name}: {e}")
        return math.nan

# ---------- driver ----------
def process(stem:str, folder:Path) -> dict:
    a_out = folder/f"{stem}_H2O.out"
    a_xyz = folder/f"{stem}_H2O.xyz"
    b_xyz = folder/f"{stem}_HCO3.xyz"

    row = dict(ligand=stem,
               vbur_aqua = percent_buried(a_xyz),
               sigma_e2  = sigma_donation(a_out),
               zn_q      = zn_charge(a_out),          # NEW COLUMN
               delta_vbur= percent_buried(b_xyz) - percent_buried(a_xyz),
               dE_bind   = math.nan)

    return row

def main(argv:List[str]):
    if len(argv) < 2:
        folder = DEFAULT_DIR
        print(f"[INFO] No folder argument; using DEFAULT_DIR = {folder}")
    else:
        folder = pathlib.Path(argv[1]).expanduser()
    if not folder.exists(): sys.exit(f"[ERR] dir {folder} not found")

    # ---- collect ligands -------------------------------------------------
    aqua_outs = list(folder.glob("*_H2O.out"))
    rows: list[dict] = []

    for a_out in aqua_outs:
        # derive the ligand stem (everything before '_H2O.out')
        stem = re.sub(r"_H2O\.out$", "", a_out.name)
        # require that the matching bicarbonate file exists
        if not (folder / f"{stem}_HCO3.out").exists():
            if VERBOSE:
                print(f"[WARN] skipping {stem}: no matching _HCO3.out")
            continue
        rows.append(process(stem, folder))

    # ---- write CSV -------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv("descriptors.csv", index=False)
    print(f"\n✅ descriptors.csv written with {len(rows)} ligands")

# ---------- CLI ----------
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract buried volume, σ‑donation, Zn charge and other descriptors "
                    "from ORCA output/XYZ pairs and write descriptors.csv")
    p.add_argument("folder", nargs="?", default=DEFAULT_DIR,
                   help=f"Folder containing *_H2O.out/xyz and *_HCO3.* files "
                        f"(default: {DEFAULT_DIR})")
    p.add_argument("-d", "--diag", action="store_true",
                   help="print per‑ligand diagnostic block")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress verbose logging")
    p.add_argument("--self-test", action="store_true",
                   help="run built‑in smoke‑test and exit")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    return p


def self_test() -> None:
    """Minimal smoke‑test: dummy ligand with fake ORCA/NBO lines."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # ---- dummy XYZ (Zn‑N‑H) ----
        xyz = td / "dummy_H2O.xyz"
        xyz.write_text("3\ncomment\nZn 0 0 0\nN 0 0 2\nH 0 0 4\n")

        # ---- dummy HCO3 XYZ ----
        (td/"dummy_HCO3.xyz").write_text(xyz.read_text())

        # ---- dummy OUT ----
        out = td / "dummy_H2O.out"
        out.write_text(""" SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS
  1. LP ( 1) N  1           97. LV ( 1)Zn  2           10.00    0.50   0.050
 Natural Population Analysis
 Zn          2    1.70000
""")
        (td/"dummy_HCO3.out").write_text(out.read_text())

        row = process("dummy", td)
        assert not math.isnan(row["vbur_aqua"]),   "vbur nan"
        assert row["sigma_e2"] == 10.0,            "σE2 wrong"
        assert row["zn_q"] == 1.7,                 "Zn q wrong"
        print("Self‑test OK")


if __name__ == "__main__":
    import argparse
    cli = _build_cli()
    args = cli.parse_args()

    # configure globals
    VERBOSE = not args.quiet
    DIAG    = args.diag

    if args.self_test:
        try:
            self_test()
            sys.exit(0)
        except Exception as e:
            print(f"[FAIL] self-test: {e}")
            sys.exit(1)

    main([sys.argv[0], str(args.folder)])