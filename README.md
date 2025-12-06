# CS176 Intro Network Analysis Project

Team Name: The Node

Members: Ziqing Jiang, Arianna Gonzalez

## Project

This project analyzes the CA-NetScience collaboration network using several algorithms introduced in CS176. The goal is to understand the structural properties of the network, identify important researchers, detect community structure, analyze global influence, and predict potential future collaborations. All analyses were produced using Python and the NetworkX library.

GitHub Link: https://github.com/ZiqingJiang/CS176_Project

All files that have:

- Report(pdf)
- Presentation(pdf)
- Code(zip)
  - source code(project.py)
  - figures & data files
- Readme(md)

Note: Figures and data files are the output after running the code. It provides convenience if the user does not want to execute the script.

## Dataset

The dataset used in this project is: https://networkrepository.com/ca-netscience.php

Format: Matrix Market(mtx)

Type: Undirected, unweighted

Node: 379

Edges: 914

Structure: Node - Researcher, Edge - Co-authorship

## How to Run

1. Install the required Python packages:

```
pip install networkx matplotlib scipy pandas
```

2. Run the Script

```
python project.py
```

3. Output

The script ensures different visualizations and data files, and will store in the same location. That includes:

- Community detection: Gircan-Newwan, Louvain, LPA
- Centralities: degree, betweenness, closeness
- Link Prediciton: Jaccaed Coefficient, Adamic-Adar Index
- PageRank
- Results that provide in terminal

## Algorithms

1. Graph Properties
   - Density
   - Average clustering coefficient
2. Centrality
   - Degree Centrality
   - Betweenness Centrality
   - Closeness Centrality
   - Eigenvector Centrality
3. Community Detection
   - Girvan-NewWan
   - Louvain
   - Label Propagation Algorithm(LPA)
4. Link Prediction
   - Jaccard Coefficient
   - Adamic-Adar Index
5. PageRank

## Notes

- The network is treated as undirected and unweighted.
- All algorithms are applied using NetworkX’s built-in implementations.
- Layouts are generated using Kamada–Kawai or Spring Layout for visual clarity.
- Random seeds are set where appropriate to ensure reproducibility.
