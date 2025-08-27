import re
import numpy as np
import pandas as pd
from pathlib import Path

# Helper functions to create csv from .h5 file
def convert_dlc_h5s_to_csv(
    dest_dir,
    pcutoff=0.6,
    include_likelihood=True,
    filter_likelihood_with_cutoff=True,
    preserve_bodypart_order=True,   # keep DLC order; set False to sort alphabetically
) -> str:
    
    """
    Convert DeepLabCut/SuperAnimal .h5 predictions in `dest_dir` to CSVs with columns:
    nose_x, nose_y, nose_likelihood, ear_x, ear_y, ear_likelihood, ...

    - pcutoff: set x/y (and optionally likelihood) to NaN where likelihood < pcutoff
    - include_likelihood: keep or drop *_likelihood columns in output
    - filter_likelihood_with_cutoff: if True, likelihood values below cutoff become NaN
    - preserve_bodypart_order: keep bodyparts in the order they appear in the HDF
    """
    dest_dir = Path(dest_dir)
    h5_files = sorted(dest_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {dest_dir}")

    COORDS = ("x", "y", "likelihood")
    COORD_SET = set(COORDS)

    def _to_bpcoord(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns

        # Case 1: MultiIndex -> identify (bodypart, coord)
        if isinstance(cols, pd.MultiIndex):
            n = cols.nlevels

            # Find which level is coord (contains at least two of x/y/likelihood)
            coord_lvl = None
            for lvl in range(n):
                vals = {str(v).lower() for v in cols.get_level_values(lvl).unique()}
                if len(vals & COORD_SET) >= 2:
                    coord_lvl = lvl
                    break
            if coord_lvl is None:
                raise ValueError("Cannot identify coordinate level in MultiIndex columns.")

            # Choose bodypart level among the remaining levels: the one with most uniques
            candidate_bps = [lvl for lvl in range(n) if lvl != coord_lvl]
            uniq_counts = {lvl: cols.get_level_values(lvl).nunique() for lvl in candidate_bps}
            bodypart_lvl = max(uniq_counts, key=uniq_counts.get)

            new_cols = [(str(c[bodypart_lvl]), str(c[coord_lvl]).lower()) for c in cols]
            df2 = df.copy()
            df2.columns = pd.MultiIndex.from_tuples(new_cols, names=["bodypart", "coord"])
            # Deduplicate (e.g., multiple scorers) by keeping first
            df2 = df2.T.groupby(level=["bodypart", "coord"]).first().T
            return df2

        # Case 2: Flat columns -> parse "scorer_bp_x" / "bp_x" / "bp_likelihood"
        pairs = []
        for name in map(str, cols):
            parts = re.split(r'[|/\s\-]+|__|_', name.strip())
            parts = [p for p in parts if p]
            coord = None
            if parts:
                last = parts[-1].lower()
                if last in COORD_SET:
                    coord = last
                    bp = "_".join(parts[:-1]) or "unknown"
                else:
                    m = re.match(r'(.+?)(x|y|likelihood)$', last, flags=re.IGNORECASE)
                    if m:
                        coord = m.group(2).lower()
                        bp = "_".join(parts[:-1] + [m.group(1)])
                    else:
                        bp = name
            else:
                bp, coord = name, None
            pairs.append((bp, coord if coord in COORD_SET else "unknown"))

        df2 = df.copy()
        df2.columns = pd.MultiIndex.from_tuples(pairs, names=["bodypart", "coord"])
        df2 = df2.loc[:, df2.columns.get_level_values("coord").isin(COORDS)]
        df2 = df2.T.groupby(level=["bodypart", "coord"]).first().T
        return df2

    for h5 in h5_files:
        df = pd.read_hdf(h5)
        df_bc = _to_bpcoord(df)

        # Apply pcutoff: mask x/y (and optionally likelihood) where likelihood < cutoff
        if pcutoff is not None and ("likelihood" in df_bc.columns.get_level_values("coord")):
            for bp in df_bc.columns.get_level_values("bodypart").unique():
                like_col = (bp, "likelihood")
                if like_col in df_bc.columns:
                    mask = df_bc[like_col] < pcutoff
                    for coord in ("x", "y"):
                        col = (bp, coord)
                        if col in df_bc.columns:
                            df_bc.loc[mask, col] = np.nan
                    if include_likelihood and filter_likelihood_with_cutoff:
                        df_bc.loc[mask, like_col] = np.nan

        # Optionally drop likelihood columns
        if not include_likelihood:
            df_bc = df_bc.loc[:, df_bc.columns.get_level_values("coord").isin(["x", "y"])]

        # Build ordered column list: per bodypart -> x, y, likelihood
        if preserve_bodypart_order:
            # keep first-seen order
            seen = []
            for bp in [c[0] for c in df_bc.columns]:
                if bp not in seen:
                    seen.append(bp)
            bodyparts = seen
        else:
            bodyparts = sorted(df_bc.columns.get_level_values("bodypart").unique())

        ordered_cols = []
        for bp in bodyparts:
            for coord in COORDS:
                col = (bp, coord)
                if col in df_bc.columns:
                    ordered_cols.append(col)
        df_bc = df_bc[ordered_cols]

        # Flatten to "bodypart_coord" and drop rows with all-NaN (ignoring frame index)
        out = df_bc.copy()
        out.columns = [f"{bp}_{coord}" for bp, coord in out.columns]
        out = out.dropna(how="all")
 
        out_csv = h5.with_suffix(".csv")

        # # out_csv_short = Path(out_csv).parent/Path(out_csv).name.partition("_super")[0]+Path(out_csv).suffix
        # p = Path(out_csv)
        # new_name = p.stem.partition("_super")[0] + p.suffix   # cut at "_super", keep .csv
        # new_path = p.with_name(new_name)                      # same folder, new name
        # out_csv_short=p.rename(new_path)  

        out.to_csv(out_csv, index=True, index_label="frame")
        print(f"✅ Saved: {out_csv}")

    return out_csv
