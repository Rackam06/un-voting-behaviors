import pyreadr
from pathlib import Path

data_dir = Path("data")
for rda in data_dir.glob("*.rda"):
    res = pyreadr.read_r(rda)
    for name, df in res.items():
        out = data_dir / f"{rda.stem}.csv"
        print(" - writing", out)
        df.to_csv(out, index=False, encoding="utf-8")
print("Done converting .rda -> .csv")
