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
- splitting?
  - does splitting an embedding destroy the embedding? (unanswered but prob no)
  - minimize overfitting for general results
  - use for early stopping
  - without splitting slight overfitting

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

#### Sampling Params

- do grid search for degrees of freedom
  - search for sample size, connectivity
  - escape threshold should have little influence and will be constant
- results
  - higher sample size gives more consistent (see std) results
  - even a sample size of 100 gives good results
  - higher connectivity (see smaller alpha) seems to give better results
  - most runs reach quite high scores (around 0.96 mean score for the upper half of runs)
  - not many epochs needed -> use early stopping

#### Run time
- increases with graph and epoch graph size
- graph size has a lesser effect on the run time because of const size samples

## Examples

### Random Geometric Graph
- with no loss of generality: graph on unit-disc centered around `(0.5, 0.5)`
- generate random positions for nodes
- two nodes are connected, if the distance between them is smaller than a threshold

### Girgs (Power-Law-Graphs)
- see https://github.com/chistopher/girgs
- TODO

### Negative Test
- random graph without structure to its features
- here: power-law graph with random features
- no learning takes place (constant precision on validation set)
- train precision ~ percentage of edges

## Future
- correlation between (sub-)graph properties and learning performance
  - as a whole for the input
  - for the subsampling in epoch training