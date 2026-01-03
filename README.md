# UN Voting Behaviors Analysis

**Author:** Wail Ameur  
**Course:** Data Mining Project

## Overview

This project analyzes voting data from the United Nations General Assembly to identify patterns in international relations. By examining how countries vote on resolutions, we can:
1.  Construct a network of alliances based on shared "Yes" votes.
2.  Identify "top allies" for key countries.
3.  Cluster countries into political communities using K-Means and PCA (for plotting only).

## Project Structure

- `data/`: Contains the input datasets and the output results.
    - `csvs/`: Generated data tables (matrices, clusters, top allies).
    - `graphs/`: Generated visualizations (heatmaps, network graphs, bar charts).
- `src/`: Python source code for the analysis.

## Setup

1.  **Prerequisites:** Make sure you have Python installed.
2.  **Install Dependencies:**
    Run the following command to install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Analysis

The analysis is split into several scripts. Run them in the following order:

### 1. Build the Voting Matrix
First, we process the raw voting data to create a matrix where rows are countries and columns are resolutions.
```bash
python src/build_matrix.py
```
*Output:* Saves `country_by_rcid_matrix.csv` and decade-specific matrices in `data/csvs/`.

### 2. Network Analysis & Top Allies
This script builds a bipartite graph (Countries <-> Resolutions) and projects it to find relationships between countries. It generates a network graph and a list of top allies for important countries.
```bash
python src/bipartite_projection.py
```
*Output:*
- `data/graphs/important_countries_graph.png`: Network visualization.
- `data/graphs/important_countries_top_allies_simple.png`: Bar chart of top allies.
- `data/csvs/important_countries_top_allies.csv`: Detailed list of allies.

### 3. Clustering (K-Means)
This script groups countries into clusters based on their voting similarity. It uses PCA to visualize these clusters in 2D space.
```bash
python src/cluster_kmeans.py
```
*Output:*
- `data/graphs/clusters_pca_plot.png`: Visualization of country clusters.
- `data/csvs/country_clusters_kmeans.csv`: List of countries and their assigned cluster.

It also contains those vizualizations per decades.

## Results

After running the scripts, check the `data/graphs/` folder for visualizations and `data/csvs/` for the processed data tables.

---
*Project created for educational purposes.*
