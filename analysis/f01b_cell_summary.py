"""Fig. 1d,e input: frame size, monitored systems and observed exceedances in the eight source-by-size cells.

Reads the derived small-system frame (data_derived/small_frame_anchor.parquet); no fitting or weighting.
Counts reproduce Table S3 (frame N, observed n) and the 68 observed exceeding small systems.

Usage: python f01b_cell_summary.py --out <folder>   (ANCHOR_WORKDIR selects the project holding data_derived/)
"""
from pathlib import Path
import argparse
import pandas as pd
from project_paths import D

SIZES = ['<=100', '101-500', '501-1000', '1001-3300']


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, required=True)
    out = ap.parse_args().out; out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(D / 'small_frame_anchor.parquet', columns=['PWSID', 'src', 'scls', 'pop', 'in_ucmr5', 'y_mcl'])
    obs = d[d.in_ucmr5.astype(bool)]
    assert len(d) == 56591 and len(obs) == 970 and int(obs.y_mcl.sum()) == 68
    frame = d.groupby(['src', 'scls']).agg(frame_systems=('PWSID', 'size'), frame_people=('pop', 'sum'))
    seen = obs.groupby(['src', 'scls']).agg(observed_systems=('PWSID', 'size'), observed_exceeding=('y_mcl', 'sum'))
    t = frame.join(seen).reset_index().rename(columns={'src': 'source', 'scls': 'size_band'})
    t['observed_exceeding'] = t.observed_exceeding.astype(int)
    t['monitored_percent'] = 100 * t.observed_systems / t.frame_systems
    t['observed_exceedance_percent'] = 100 * t.observed_exceeding / t.observed_systems
    t['size_band'] = pd.Categorical(t.size_band, SIZES, ordered=True)
    t = t.sort_values(['source', 'size_band'])
    t.to_csv(out / 'fig1_cell_summary.csv', index=False)
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
