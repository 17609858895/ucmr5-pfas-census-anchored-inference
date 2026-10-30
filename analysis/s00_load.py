"""Read the official UCMR5 ZIP (read-only, no extraction to disk) and cache typed parquet copies."""
import zipfile, io, pandas as pd, pathlib
from project_paths import UCMR5_ZIP, D as OUT
z = zipfile.ZipFile(UCMR5_ZIP)
for member, name in [("UCMR5_All.txt", "all"), ("UCMR5_AddtlDataElem.txt", "addtl"), ("UCMR5_ZIPCodes.txt", "zip")]:
    with z.open(member) as f:
        df = pd.read_csv(io.TextIOWrapper(f, encoding="cp1252"), sep="\t", dtype=str, keep_default_na=False)
    df.to_parquet(OUT / f"ucmr5_{name}.parquet", index=False)
    print(member, df.shape)
