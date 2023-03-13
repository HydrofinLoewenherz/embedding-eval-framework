# Topics

## Introduction
- why embeddings
- general idea

## Graph Embeddings
- graph is nodes connected by edges, nodes represent (complex) data objects
- embedding is mapping node to fixed-length feature vector that captures information
- like hashing?

## Problems
- high class imbalance
  - a lot more non-edges than edges for real life graphs (power law)
- graph dataset has `O(n²)` edges, therefore can't loop all edges each epoch

### Balancing
- weighted sampler over edges 
  - edges get a uniform higher weight than non-edges
- still a lot of edges to consider

### Graph Sampling
- instead subsample graph with `n_epoch << n`
- sample over nodes and keep all edges that connect the sampled nodes
- different methods of sampling (random walk, bfs, dfs, random)
- here: choose random walk with two degrees of freedom
  - sample size
  - connectivity (jump probability)
  - escape threshold (chosen dependent on connectivity)
- possibly rank nodes (didn't seem to give improvements)
- results in a smaller train-dataset and lowers class imbalance

## Examples

### Random Geometric Graph
- TODO

### Power Law Graph
- TODO

## Future
- correlation between (sub-)graph properties and learning performance
  - as a whole for the input
  - for the subsampling in epoch training