import pandas as pd
from pathlib import Path

data_dir = Path("data")
votes_csv = data_dir / "un_votes.csv"
roll_calls_csv = data_dir / "un_roll_calls.csv"
issues_csv = data_dir / "un_roll_call_issues.csv"

votes = pd.read_csv(votes_csv)
roll_calls = pd.read_csv(roll_calls_csv)
issues = pd.read_csv(issues_csv)

print("votes shape:", votes.shape)
print("roll_calls shape:", roll_calls.shape)
print("issues shape:", issues.shape)

# Quick normalization of column names (stripping whitespace)
votes.columns = votes.columns.str.strip()
roll_calls.columns = roll_calls.columns.str.strip()
issues.columns = issues.columns.str.strip()

# Map vote text to numeric: yes -> 1, abstain -> 0, no -> -1
vote_map = {
    "yes": 1,
    "no": -1,
    "abstain": 0,
    # some datasets have 'nv' or 'not present' or uppercase
    "veto": -1,
    "absent": 0,
    "not present": 0
}

# Normalize vote to strings + lowercase then maps it to numeric value in vote_map
# Fallback to 0 in case of NaN (unknown) and cast column to int
votes["vote_norm"] = votes["vote"].astype(str).str.lower().map(vote_map).fillna(0).astype(int)

# Create the matrix rows = country, cols = rcid (resolutions)
matrix = votes.pivot_table(index="country", columns="rcid", values="vote_norm", aggfunc="first").fillna(0).astype(int)
print("Matrix shape (countries x rcid):", matrix.shape)

# Save the matrix
matrix.to_csv(data_dir / "csvs" / "country_by_rcid_matrix.csv")
print("Saved country_by_rcid_matrix.csv")


# ---- Time evolution (by decade) ----

# Parse dates
roll_calls["date"] = pd.to_datetime(roll_calls["date"], errors="coerce")
roll_calls["year"] = roll_calls["date"].dt.year

# Merge year into votes
votes = votes.merge(roll_calls[["rcid", "year"]], on="rcid", how="left")

votes = votes.dropna(subset=["year"])
votes["year"] = votes["year"].astype(int)
votes["decade"] = (votes["year"] // 10) * 10

# Build one matrix per decade
for decade, df_decade in votes.groupby("decade"):
    matrix_decade = df_decade.pivot_table(index="country", columns="rcid", values="vote_norm", aggfunc="first").fillna(0).astype(int)
    matrix_decade.to_csv(data_dir / "csvs" / f"country_by_rcid_{decade}.csv")
    print(f"Saved matrix for decade {decade}")