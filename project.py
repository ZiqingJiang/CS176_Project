# File: project.py
# Author: Ziqing Jiang, Arianna Gonzalez
# Date: 2025-11-10
# Description: We use netscience as our project dataset. 

import networkx as nx
import os
import matplotlib.pyplot as plt
from scipy.io import mmread
import pandas as pd
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
import random
from networkx.algorithms.link_prediction import adamic_adar_index

# Helper methods for community detection
def set_node_community(G, communities):
    """Add community ID to node attributes."""
    for c, v_c in enumerate(communities):
        for v in v_c:
            G.nodes[v]['community'] = c + 1

def set_edge_community(G):
    """Mark internal and external edges by community."""
    for v, w in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
    """Generate distinct RGB color tuples."""
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

def smallest_largest(communities):
    sizes = [len(c) for c in communities]
    return min(sizes), max(sizes)

# Load dataset
matrix = mmread("ca-netscience.mtx")
netscience = nx.from_scipy_sparse_array(matrix)

# density
density = nx.density(netscience)
print("Density:", density)

# Average Clustering Coefficient
avg_clustering = nx.average_clustering(netscience)
print("Average clustering coefficient:", avg_clustering)

# ========  Girvan–Newman algorithm  ======== 
communities_gn = nx_comm.girvan_newman(netscience)
node_groups = [list(c) for c in next(communities_gn)]

#print results
print("# of communities: ", len(node_groups), ", modularity score: ", nx_comm.modularity(netscience, node_groups))
print("Smallest community size: ", smallest_largest(node_groups)[0])
print("Largest community size: ", smallest_largest(node_groups)[1])

# Layout for plotting
pos = nx.kamada_kawai_layout(netscience)

# Plot Girvan–Newman communities
plt.figure(figsize=(12, 12))
color_map = ["red" if node in node_groups[0] else "orange" for node in netscience]

nx.draw(
    netscience,
    pos=pos,
    node_color=color_map,
    node_size=50,
    width=0.5,
    with_labels=False,
)
plt.title("Netscience – Girvan–Newman Communities", fontsize=14)
plt.savefig("netscience_gn.png", dpi=300, bbox_inches="tight")
plt.show()

# ======== Louvain algorithm  ======== 
partition = nx_comm.louvain_communities(netscience, seed=123)
print("# of communities: ", len(partition), ", modularity score: ", nx_comm.modularity(netscience, partition))
print("Smallest community size: ", smallest_largest(partition)[0])
print("Largest community size: ", smallest_largest(partition)[1])

# Save communities to file
with open("communities.txt", "w") as fp:
    for item in partition:
        fp.write("%s\n" % item)
    print("Saved Louvain communities to communities.txt")

# Draw Louvain communities
set_node_community(netscience, partition)
set_edge_community(netscience)

node_color = [get_color(netscience.nodes[v]["community"]) for v in netscience.nodes]
external = [(v, w) for v, w in netscience.edges if netscience.edges[v, w]["community"] == 0]
internal = [(v, w) for v, w in netscience.edges if netscience.edges[v, w]["community"] > 0]
internal_color = [get_color(netscience.edges[e]["community"]) for e in internal]

plt.figure(figsize=(12, 12))

# External edges
nx.draw_networkx_edges(
    netscience,
    pos=pos,
    edgelist=external,
    edge_color="#333333",
    width=0.3
)

# Nodes + internal edges
nx.draw_networkx_nodes(
    netscience,
    pos=pos,
    node_color=node_color,
    node_size=50
)
nx.draw_networkx_edges(
    netscience,
    pos=pos,
    edgelist=internal,
    edge_color=internal_color,
    width=0.5
)

plt.title("Netscience - Louvain Communities", fontsize=24)
plt.axis('off')
plt.savefig("netscience_lv.png", dpi=300, bbox_inches="tight")
plt.show()

# Label Propagation Algorithm (LPA)
communities_lpa = list(nx_comm.label_propagation_communities(netscience))
print("# of communities: ", len(communities_lpa), ", modularity score: ", nx_comm.modularity(netscience, communities_lpa))
print("Smallest community size: ", smallest_largest(communities_lpa)[0])
print("Largest community size: ", smallest_largest(communities_lpa)[1])

# Save LPA communities to file
with open("communities_lpa.txt", "w") as fp:
    for item in communities_lpa:
        fp.write("%s\n" % sorted(list(item)))
print("Saved Label Propagation communities to communities_lpa.txt")

# Assign node + edge community attributes
set_node_community(netscience, communities_lpa)
set_edge_community(netscience)

# Colors for nodes
node_color_lpa = [get_color(netscience.nodes[v]["community"]) for v in netscience.nodes]

# Separate internal vs external edges
external_lpa = [(v, w) for v, w in netscience.edges if netscience.edges[v, w]["community"] == 0]
internal_lpa = [(v, w) for v, w in netscience.edges if netscience.edges[v, w]["community"] > 0]
internal_color_lpa = [get_color(netscience.edges[e]["community"]) for e in internal_lpa]


# Plot LPA Communities
plt.figure(figsize=(12, 12))
plt.axis("off")

# Draw external edges (grey)
nx.draw_networkx_edges(
    netscience,
    pos=pos,
    edgelist=external_lpa,
    edge_color="#666666",
    width=0.3
)

# Draw nodes + internal edges
nx.draw_networkx_nodes(
    netscience,
    pos=pos,
    node_color=node_color_lpa,
    node_size=50
)
nx.draw_networkx_edges(
    netscience,
    pos=pos,
    edgelist=internal_lpa,
    edge_color=internal_color_lpa,
    width=0.5
)

plt.title("Netscience - Label Propagation Communities", fontsize=24)
plt.savefig("netscience_lpa.png", dpi=300, bbox_inches="tight")
plt.show()


# ======== Centrality Analysis  ======== 
# Show the figure of power bus
nx.draw(netscience, with_labels=False, node_color="blue", node_size=20, edge_color="gray", width=0.2, alpha=0.3)
plt.savefig("netscience.png", dpi=300)
plt.show()

degree = dict(netscience.degree())
betweenness = nx.betweenness_centrality(netscience)
closeness = nx.closeness_centrality(netscience)
eigenvector = nx.eigenvector_centrality(netscience, max_iter=500)

# Helper: return top N nodes from a centrality dict
def top_n(cent_dict, n=10):
    return sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:n]


top_degree = top_n(degree)
top_between = top_n(betweenness)
top_close = top_n(closeness)
top_eigen = top_n(eigenvector)

print("Top 10 Degree:", top_degree)
print("Top 10 Betweenness:", top_between)
print("Top 10 Closeness:", top_close)
print("Top 10 Eigenvector:", top_eigen)

# draw each centrality

# degree centrality
plt.figure(figsize=(12, 12))
highlight = set([n for n, _ in top_degree])
node_colors = ["crimson" if n in highlight else "lightpink" for n in netscience.nodes()]
nx.draw(
    netscience,
    pos=pos,
    node_color=node_colors,
    node_size=50,
    edge_color="gray",
    width=0.5,
    with_labels=False
)
plt.title("Netscience - Top 10 Degree Centrality", fontsize=24)
plt.savefig("netscience_degree.png", dpi=300, bbox_inches="tight")
plt.show()

# betweenness centrality
plt.figure(figsize=(12, 12))
highlight = set([n for n, _ in top_between])
node_colors = ["green" if n in highlight else "lightgreen" for n in netscience.nodes()]
nx.draw(
    netscience,
    pos=pos,
    node_color=node_colors,
    node_size=50,
    edge_color="gray",
    width=0.5,
    with_labels=False
)
plt.title("Netscience - Top 10 Betweenness Centrality", fontsize=24)
plt.savefig("netscience_betweenness.png", dpi=300, bbox_inches="tight")
plt.show()

# closeness centrality
plt.figure(figsize=(12, 12))
highlight = set([n for n, _ in top_close])
node_colors = ["darkblue" if n in highlight else "lightskyblue" for n in netscience.nodes()]
nx.draw(
    netscience,
    pos=pos,
    node_color=node_colors,
    node_size=50,
    edge_color="gray",
    width=0.5,
    with_labels=False
)
plt.title("Netscience - Top 10 Closeness Centrality", fontsize=24)
plt.savefig("netscience_closeness.png", dpi=300, bbox_inches="tight")
plt.show()

# eigenvector centrality
plt.figure(figsize=(12, 12))
highlight = set([n for n, _ in top_eigen])
node_colors = ["green" if n in highlight else "lightgreen" for n in netscience.nodes()]
nx.draw(
    netscience,
    pos=pos,
    node_color=node_colors,
    node_size=50,
    edge_color="gray",
    width=0.5,
    with_labels=False
)
plt.title("Netscience - Top 10 Eigenvector Centrality", fontsize=24)
plt.savefig("netscience_eigenvector.png", dpi=300, bbox_inches="tight")
plt.show()

# ======== Link Prediction ======== 
# 
# 1. Compute Jaccard Coefficient Predictions
preds = nx.jaccard_coefficient(netscience)
top_preds = sorted(preds, key=lambda x: x[2], reverse=True)[:10]

print("\nTop 10 predicted links (Jaccard):")
for u, v, p in top_preds:
    print(f"({u}, {v}) → {p:.4f}")

pos = nx.spring_layout(netscience, seed=42)

jaccard_nodes = set()
for u, v, _ in top_preds:
    jaccard_nodes.add(u)
    jaccard_nodes.add(v)

pred_edges = [(u, v) for u, v, _ in top_preds]

plt.figure(figsize=(10, 10))

node_colors = ["red" if n in jaccard_nodes else "gold" for n in netscience.nodes()]
node_sizes = [140 if n in jaccard_nodes else 30 for n in netscience.nodes()]

nx.draw(
    netscience, pos,
    node_color=node_colors,
    node_size=node_sizes,
    edge_color="lightgray",
    alpha=0.7,
    with_labels=False
)

nx.draw_networkx_edges(
    netscience, pos,
    edgelist=pred_edges,
    edge_color='red',
    style='dashed',
    width=1.8
)

plt.title("Jaccard Coefficient — Top 10 Predicted Links (Highlighted Nodes)", fontsize=16)
plt.savefig("netscience_jaccard_highlighted.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Compute Adamic–Adar Predictions
aa_preds = list(adamic_adar_index(netscience))
aa_sorted = sorted(aa_preds, key=lambda x: x[2], reverse=True)
top_aa = aa_sorted[:10]

print("\nTop 10 predicted links (Adamic–Adar):")
for u, v, s in top_aa:
    print((u, v), "→", round(s, 4))

adamic_nodes = set()
for u, v, _ in top_aa:
    adamic_nodes.add(u)
    adamic_nodes.add(v)

predicted_edges_aa = [(u, v) for u, v, _ in top_aa]

plt.figure(figsize=(10, 10))

node_colors_aa = ["navy" if n in adamic_nodes else "#87CEFA" for n in netscience.nodes()]
node_sizes_aa = [140 if n in adamic_nodes else 30 for n in netscience.nodes()]

nx.draw(
    netscience, pos,
    node_color=node_colors_aa,
    node_size=node_sizes_aa,
    edge_color="lightgray",
    alpha=0.7,
    with_labels=False
)

nx.draw_networkx_edges(
    netscience, pos,
    edgelist=predicted_edges_aa,
    edge_color="blue",
    style='dashed',
    width=1.8
)

plt.title("Adamic–Adar — Top 10 Predicted Links (Highlighted Nodes)", fontsize=16)
plt.savefig("netscience_adamic_highlighted.png", dpi=300, bbox_inches="tight")
plt.show()

# ======== Page Rank ======== 
# Compute PageRank scores
pagerank_scores = nx.pagerank(netscience, alpha=0.85)

# Sort and get top 10
top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 nodes by PageRank:")
for node, score in top_pagerank:
    print(f"Node {node}: {score:.5f}")

# Extract top nodes
top_1_node = top_pagerank[0][0]                  # highest PageRank
top_10_nodes = [node for node, _ in top_pagerank]

# Assign colors:
# - red for top 1
# - orange for top 2–10
# - light blue for all others
node_colors = []
for n in netscience.nodes():
    if n == top_1_node:
        node_colors.append("red")
    elif n in top_10_nodes:
        node_colors.append("orange")
    else:
        node_colors.append("skyblue")

# Node sizes scaled by PageRank (same as your code)
node_sizes = [pagerank_scores[n] * 8000 for n in netscience.nodes()]

# Layout
pos = nx.spring_layout(netscience, seed=42)

plt.figure(figsize=(12, 12))

# Draw nodes with custom colors and sizes
nx.draw_networkx_nodes(
    netscience,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    alpha=0.85
)

# Draw edges
nx.draw_networkx_edges(netscience, pos, alpha=0.25)

plt.title("Top 10 PageRank Nodes Highlighted", fontsize=16)
plt.savefig("netscience_pagerank.png", dpi=300, bbox_inches="tight")
plt.show()
