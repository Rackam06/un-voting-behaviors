import pandas as pd
from pathlib import Path

data_dir = Path("data")
matrix = pd.read_csv(data_dir / "country_by_rcid_matrix.csv", index_col=0)

print("Countries:", matrix.shape[0])
print("Resolutions:", matrix.shape[1])

# How many votes of each type per country
vote_counts = (matrix == 1).sum(axis=1).rename("yes_count").to_frame().join(
    (matrix == -1).sum(axis=1).rename("no_count"),
    how="left").join(
    (matrix == 0).sum(axis=1).rename("abstain_or_missing"), how="left")
print(vote_counts.sort_values("yes_count", ascending=False).head(10))

# How many countries didn't vote / abstained
absent_counts = (matrix == 0).sum(axis=0)
print("Roll-calls with most abstentions (top 10):")
print(absent_counts.sort_values(ascending=False).head(10))

# Distribution of votes overall (within each column and across columns)
total_yes = (matrix == 1).sum().sum()
total_no = (matrix == -1).sum().sum()
total_zero = (matrix == 0).sum().sum()
print("Total counts -> yes:", total_yes, "no:", total_no, "abstain/missing:", total_zero)
