import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from pathlib import Path
import seaborn as sns

# --- Step 1: Load Data ---
# We define where our data is stored
data_dir = Path("data")

# Read the raw votes file (contains: rcid, country, vote, etc.)
df = pd.read_csv(data_dir / "un_votes.csv")

# --- Step 2: Filter for 'Yes' Votes ---
# We are interested in finding "allies" or countries that agree.
# So we only keep rows where the vote was "yes".
# If Country A and Country B both voted "yes" on Resolution X, they are connected.
df_yes = df[df["vote"] == "yes"]

# --- Step 3: Build a Bipartite Graph ---
# A "Bipartite Graph" is a graph with two distinct sets of nodes.
# Set 1: Countries
# Set 2: Resolutions (rcid)
# Edges only exist between a Country and a Resolution (meaning "Country voted Yes on Resolution").
B = nx.Graph()

# Get unique lists of countries and resolutions
countries = df_yes["country"].unique()
resolutions = df_yes["rcid"].unique()

# Add nodes to the graph, specifying which set they belong to (bipartite=0 or 1)
B.add_nodes_from(countries, bipartite=0) # Set 0: Countries
B.add_nodes_from(resolutions, bipartite=1) # Set 1: Resolutions

# Add edges: Connect each country to the resolutions they voted 'yes' on
# zip() pairs the country column with the rcid column row by row
B.add_edges_from(zip(df_yes["country"], df_yes["rcid"]))

# --- Step 4: Project to Country-Country Graph ---
# We want to see relationships between countries, not countries and resolutions.
# "Projecting" the graph means we create a new graph where:
# - Nodes are ONLY countries.
# - An edge exists between Country A and Country B if they share a neighbor in the original graph.
# - The "weight" of the edge is the NUMBER of shared neighbors (how many times they both voted 'yes').
G = bipartite.weighted_projected_graph(B, countries)

print(f"Graph stats: {len(G.nodes())} countries, {len(G.edges())} edges")

# --- Step 5: Select Important Countries ---
# The full graph is too big to plot clearly (too many edges).
# We focus on a specific list of politically important countries to make the visualization readable and interpret it on the report.
important_countries = [
    "United States", "Russia", "China",
    "France", "United Kingdom",
    "Germany", "Iran",
    "Egypt", "Saudi Arabia", "Vietnam", "Poland", "Hungary", "Iraq", "South Africa", "Italy"
]

# We must ensure these countries actually exist in our data (graph nodes)
valid_countries = [c for c in important_countries if c in G.nodes()]

# --- Step 6: Visualization (Network Graph / Bipartite Projection) ---
plt.figure(figsize=(12, 10))

# Create a "subgraph" containing ONLY the important countries and the edges between them.
H = G.subgraph(valid_countries)

# Choose a layout: "circular_layout" puts nodes in a circle.
# This is good for showing connections between a small group of peers.
pos = nx.circular_layout(H)

# Draw the nodes (countries)
nx.draw_networkx_nodes(H, pos, node_size=1000, node_color='lightblue')

# Draw the edges (connections)
# We want thicker lines for stronger alliances (higher weight).
weights = [H[u][v]['weight'] for u, v in H.edges()]

# Normalize weights to calculate width (so lines aren't too thick or thin)
if weights:
    max_w = max(weights)
    # Width is proportional to weight: (5 * weight / max_weight)
    widths = [5 * w / max_w for w in weights]
else:
    widths = 1

nx.draw_networkx_edges(H, pos, width=widths, alpha=0.5, edge_color='gray') # alpha is the edge transparency

# Add labels (country names)
nx.draw_networkx_labels(H, pos, font_size=10, font_weight='bold')

# Add edge labels (show the actual number of shared votes on the lines)
# u and v are the nodes (countries), d is the edge data dictionary (containing 'weight')
edge_labels = {(u, v): int(d['weight']) for u, v, d in H.edges(data=True)}
nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=8)

plt.title("Shared 'Yes' Votes Among Key Countries")
plt.axis("off") # Turn off the x/y axis numbers
plt.tight_layout() # Adjust padding between elements
plt.savefig(data_dir / "important_countries_graph.png")
print("Saved important_countries_graph.png")


# --- Step 7: Heatmap Visualization ---
# A heatmap is another way to show the same data but here it's a better visual display.
# It's a grid where color intensity represents the number of shared votes.

# Build the matrix manually for the selected countries
matrix_data = []
for c1 in valid_countries:
    row = []
    for c2 in valid_countries:
        if c1 == c2:
            row.append(0) # Diagonal is 0 (we don't count self-agreement ofc)
        elif H.has_edge(c1, c2):
            row.append(int(H[c1][c2]['weight'])) # Value = number of shared votes
        else:
            row.append(0) # No shared votes
    matrix_data.append(row)

# Create a DataFrame for the heatmap
df_matrix = pd.DataFrame(matrix_data, index=valid_countries, columns=valid_countries)
df_matrix.to_csv(data_dir / "important_countries_matrix.csv")
print("Saved important_countries_matrix.csv")

# Plot the heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(df_matrix, annot=True, fmt="d", cmap="YlGnBu") # annot=True shows numbers, fmt="d" means integer, cmap is just for colors
plt.title("Number of Shared 'Yes' Votes")
plt.tight_layout()
plt.savefig(data_dir / "important_countries_heatmap.png")
print("Saved important_countries_heatmap.png")


# --- Step 8: Top Allies Table ---
# For each important country, we want to find who their "best friends" are in the WHOLE UN.
# We look at the full graph G, not just the subgraph H.

top_allies_data = []

for country in valid_countries:
    # Get all neighbors of the country and the weight of the connection
    neighbors = []
    for neighbor, data in G[country].items():
        neighbors.append((neighbor, data['weight']))
    
    # Sort the list of neighbors by weight (highest first)
    neighbors.sort(key=lambda x: x[1], reverse=True)
    
    # Take the top 10 allies
    top_10 = neighbors[:10]
    
    # Prepare a row for the CSV file
    row = {"Country": country}
    for i, (ally, weight) in enumerate(top_10):
        row[f"Ally_{i+1}"] = ally
        row[f"Votes_{i+1}"] = weight
    
    top_allies_data.append(row)

# Save the top allies to a CSV file
df_allies = pd.DataFrame(top_allies_data)
df_allies.to_csv(data_dir / "important_countries_top_allies.csv", index=False)
print("Saved important_countries_top_allies.csv")
