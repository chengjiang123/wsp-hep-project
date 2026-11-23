"""Convert the ParT authors' TopLandscape / QuarkGluon parquet files into padded .npy arrays.

Every model and every arm reads these same arrays, so all arms see exactly the same jets.
Output, per dataset and split, in <out>/<ds>/:
  <split>_vec.npy  [N, P, 6] float32  px, py, pz, E, deta, dphi (zeros for padding), P = 128 leading particles
  <split>_n.npy    [N]       int16    number of real particles (<= P)
  <split>_jet.npy  [N, 3]    float32  jet_pt, jet_eta, jet_energy
  <split>_y.npy    [N]       int8     1 = signal (top for TopLandscape, quark for QuarkGluon), as in the files' `label`
  <split>_pid.npy  [N, P, 7] int8     QuarkGluon only: charge, isChargedHadron, isNeutralHadron, isPhoton,
                                      isElectron, isMuon, OmniLearned integer PID (README recipe, see omni_pid)
Splits: TopLandscape pool = train_file, val = val_file, test = test_file.
        QuarkGluon pool = train_file_0..17, test = test_file_0..1 (no official val; full-N runs take 11.11% of the pool, as the ParT script).
Usage: python prep_data.py --src $SCRATCH/wsp_bench/data --out $SCRATCH/wsp_bench/cache
"""
import argparse
import glob
import os

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

P = 128
# OmniLearned README: experimentally accessible PIDs, integer id = position in the sorted unique list (0 = padding)
OMNI_PDG = np.array([-211, -13, -11, 0, 11, 13, 22, 130, 211])
REMAP = [(321, 211), (2212, 211), (-321, -211), (-2212, -211), (2112, 130), (-2112, 130)]


def pad(a, dtype=np.float32):
    return ak.to_numpy(ak.fill_none(ak.pad_none(a, P, clip=True), 0)).astype(dtype)


def omni_pid(pdg, charge):
    pdg = pdg.astype(np.int64).copy()
    for a, b in REMAP:
        pdg[pdg == a] = b
    idx = np.searchsorted(OMNI_PDG, pdg).clip(0, len(OMNI_PDG) - 1)
    ok = OMNI_PDG[idx] == pdg
    # anything outside the list (long-lived strange baryons etc.): charged -> pi+-, neutral -> K0L
    other = np.where(charge > 0, 211, np.where(charge < 0, -211, 130))
    idx[~ok] = np.searchsorted(OMNI_PDG, other[~ok])
    return idx.astype(np.int8)


def convert(files, out, split):
    n_tot = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    has_pid = "part_pid" in pq.ParquetFile(files[0]).schema_arrow.names
    mm = lambda name, shape, dt: np.lib.format.open_memmap(os.path.join(out, f"{split}_{name}.npy"), "w+", dt, shape)
    vec, n, jet, y = mm("vec", (n_tot, P, 6), np.float32), mm("n", (n_tot,), np.int16), mm("jet", (n_tot, 3), np.float32), mm("y", (n_tot,), np.int8)
    pid = mm("pid", (n_tot, P, 7), np.int8) if has_pid else None
    i = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=100_000):
            t = ak.from_arrow(batch)
            k = len(t)
            vec[i:i + k] = np.stack([pad(t[c]) for c in ("part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi")], -1)
            n[i:i + k] = np.minimum(ak.to_numpy(ak.num(t["part_px"])), P)
            jet[i:i + k] = np.stack([ak.to_numpy(t[c]) for c in ("jet_pt", "jet_eta", "jet_energy")], -1)
            y[i:i + k] = ak.to_numpy(t["label"]).astype(np.int8)
            if has_pid:
                charge = pad(t["part_charge"], np.int8)
                flags = [charge] + [pad(t[c], np.int8) for c in ("part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon")]
                flags.append(omni_pid(pad(t["part_pid"], np.int64), charge))
                pid[i:i + k] = np.stack(flags, -1)
            i += k
    for a in (vec, n, jet, y) + ((pid,) if has_pid else ()):
        a.flush()
    print(f"{out}/{split}: {n_tot} jets, signal fraction {float(np.asarray(y).mean()):.3f}, pid={has_pid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nat = lambda fs: sorted(fs, key=lambda s: int(s.rsplit("_", 1)[1].split(".")[0]))
    jobs = {
        "top": {"pool": [f"{a.src}/TopLandscape/train_file.parquet"], "val": [f"{a.src}/TopLandscape/val_file.parquet"],
                "test": [f"{a.src}/TopLandscape/test_file.parquet"]},
        "qg": {"pool": nat(glob.glob(f"{a.src}/QuarkGluon/train_file_*.parquet")), "test": nat(glob.glob(f"{a.src}/QuarkGluon/test_file_*.parquet"))},
    }
    for ds, splits in jobs.items():
        os.makedirs(f"{a.out}/{ds}", exist_ok=True)
        for split, files in splits.items():
            if os.path.exists(f"{a.out}/{ds}/{split}_y.npy"):
                print("exists", ds, split)
                continue
            convert(files, f"{a.out}/{ds}", split)


if __name__ == "__main__":
    main()
