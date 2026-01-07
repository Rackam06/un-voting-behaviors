import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style for plots (makes them look nicer but its optional)
sns.set_theme()

# List of countries we want to highlight on the plot because they are politically important (and relevant to analysis)
important_countries = [
    "United States", "Russia", "China",
    "France", "United Kingdom",
    "Germany", "Iran",
    "Egypt", "Saudi Arabia", "Vietnam", "Poland", "Hungary", "Iraq", "South Africa", "Italy"
]

# Define the directory where data is stored
data_dir = Path("data")

# Load the main matrix: rows are countries, columns are roll-call IDs (votes)
# index_col=0 means the first column (country names) is used as the index
matrix = pd.read_csv(data_dir / "csvs" / "country_by_rcid_matrix.csv", index_col=0)

# --- Step 1: Filter Data ---
# We want to remove columns (votes) where almost everyone abstained or was missing.
# min_nonzero_per_rc = 10 means we need at least 10 countries with a real vote (Yes/No) to keep the column.
min_nonzero_per_rc = 10
nonzero_counts = (matrix != 0).sum(axis=0) # Count non-zero values per column
keep_cols = nonzero_counts[nonzero_counts >= min_nonzero_per_rc].index # Keep columns that meet the threshold
matrix_f = matrix[keep_cols] # Create a filtered matrix
print(f"After filtering rcids with <{min_nonzero_per_rc} ; nonzero votes: {matrix_f.shape}")

# Convert the dataframe to a numpy array for calculations
X = matrix_f.values  # rows = countries, cols = votes (values are -1, 0, 1)

# --- Step 2: Normalization ---
# We normalize each country's vector to have a length of 1 (unit norm).
# This ensures that countries with more votes don't dominate just because they voted more often.
# It focuses on the direction of voting (pattern) rather than the magnitude.
# It makes them all the same "strength" even if they have different numbers of total votes, but their vector will still have the same direction.
row_norms = np.linalg.norm(X, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1 # Avoid division by zero if a country has all 0
X_norm = X / row_norms

# --- Step 3: Find Best Number of Clusters (k) ---
# We try different values of k (from 2 to 7) to see which one fits the data best.
# We use the "silhouette score" to measure quality (cohesion) within it's cluster, higher is better.
best_k = None
best_score = -1
results = []
for k in range(2, 8):
    km = KMeans(n_clusters=k) # Create KMeans model with k clusters
    labels = km.fit_predict(X_norm) # Fit model and get cluster labels
    score = silhouette_score(X_norm, labels, metric="euclidean") # Calculate score
    results.append((k, score))
    if score > best_score:
        best_score = score
        best_k = k
    print(f"k={k}, silhouette={score}")

print("Best k by silhouette:", best_k)

# --- Step 4: Final Clustering ---
# Use the best k found, or default to 2 if something went wrong
final_k = best_k or 2
km = KMeans(n_clusters=final_k)
labels = km.fit_predict(X_norm) # Get final cluster labels for all countries

# Save the results to a CSV file
countries = matrix_f.index
df_clusters = pd.DataFrame({"country": countries, "cluster": labels})
df_clusters.to_csv(data_dir / "csvs" / "country_clusters_kmeans.csv", index=False)
print("Saved country_clusters_kmeans.csv")

# --- Step 5: Visualization with PCA ---
# PCA (Principal Component Analysis) reduces the data to 2 dimensions (2D) so we can plot it.
# It finds the 2 main "directions" of variance in the voting data.
pca = PCA(n_components=2)
coords = pca.fit_transform(X_norm) # Get the 2D coordinates for each country

plt.figure(figsize=(10, 7))
palette = sns.color_palette("tab10", final_k) # Create a color palette

# Plot each cluster with a different color (autoamtically assigned with matplotlib)
for cl in range(final_k):
    mask = labels == cl # Select countries in this cluster : it's a boolean mask like True for countries in cluster cl, False otherwise
    plt.scatter(coords[mask, 0], coords[mask, 1], label=f"cluster {cl}", alpha=0.8) # coords[mask, 0] are x values, coords[mask, 1] are y values

# Add labels for important countries so we can see where they are
for i, country in enumerate(countries):
    if country in important_countries:
        plt.text(coords[i, 0], coords[i, 1], country, fontsize=8)

plt.xlabel("PC1") # PC1: First Principal Component (x-axis)
plt.ylabel("PC2") # PC2: Second Principal Component (y-axis)
plt.title(f"KMeans clusters (k={final_k}) with PCA projection")
plt.legend() # Show legend for clusters
plt.tight_layout() # Adjust padding between elements
plt.savefig(data_dir / "graphs" / "clusters_pca_plot.png", dpi=150)
print("Saved clusters_pca_plot.png")

# NB: The PCA coordinates values are not unit like "% agreement", they are just projections for visualization. 
# It is relative positions/distances (who is near whom), and which rcids load strongly on each PC

# --- Step 6: Per-Decade Analysis ---
# This function does the same thing as above, but for a specific decade's data.
def run_kmeans_pca_for_matrix(matrix_decade: pd.DataFrame, tag: str, k: int) -> None:
    # Filter columns with too few votes
    min_nonzero_per_rc = 10
    nonzero_counts = (matrix_decade != 0).sum(axis=0)
    keep_cols = nonzero_counts[nonzero_counts >= min_nonzero_per_rc].index
    matrix_f = matrix_decade[keep_cols]

    print(f"[{tag}] matrix after filtering: {matrix_f.shape}")

    # Safety check: if data is too small, skip this decade (it skips 2020 as it has only one resolution voted)
    if matrix_f.shape[0] < k:
        print(f"[{tag}] skipped (not enough countries for k={k})")
        return
    if matrix_f.shape[1] < 2:
        print(f"[{tag}] skipped (not enough rcids/features)")
        return

    X = matrix_f.values

    # Normalize data
    row_norms = np.linalg.norm(X, axis=1, keepdims=True) # Compute row norms with parameters axis=1 (rows), keepdims=True (keep 2D shape)
    row_norms[row_norms == 0] = 1 # Avoid division by zero
    X_norm = X / row_norms

    # Run KMeans with the fixed k (passed as parameter)
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(X_norm)

    # Save cluster results
    countries = matrix_f.index
    df_clusters = pd.DataFrame({"country": countries, "cluster": labels})
    out_clusters = data_dir / "csvs" / f"country_clusters_kmeans_{tag}.csv"
    df_clusters.to_csv(out_clusters, index=False)
    print(f"[{tag}] saved {out_clusters.name}")

    # Run PCA for plotting
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_norm)

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10", k)

    # Plot points
    for cl in range(k):
        mask = labels == cl
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f"cluster {cl}", alpha=0.8, color=palette[cl])
    
    # Label important countries
    for i, country in enumerate(countries):
        if country in important_countries:
            plt.text(coords[i, 0], coords[i, 1], country, fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"KMeans clusters (k={k}) with PCA projection — {tag}")
    plt.legend()
    plt.tight_layout()

    # Save plot
    out_plot = data_dir / "graphs" / f"clusters_pca_plot_{tag}.png"
    plt.savefig(out_plot)
    plt.close()
    print(f"[{tag}] saved {out_plot.name}")


# Loop through all decade files and run the analysis
for p in sorted((data_dir / "csvs").glob("country_by_rcid_*.csv")):
    if p.name == "country_by_rcid_matrix.csv":
        continue # Skip the main matrix file

    # Extract decade from filename (like "1940" from "country_by_rcid_1940.csv")
    try:
        decade = p.stem.split("_")[-1]
        int(decade)  # Verify it's a number
    except ValueError:
        continue

    # Load the decade's data and run the function
    matrix_decade = pd.read_csv(p, index_col=0)
    run_kmeans_pca_for_matrix(matrix_decade, tag=decade, k=final_k)
