*These notes are not final statements*

# Deficit

As it turns out, the subsampling implementations used in this experiment for `bfs` and `random_walk` contained errors.
The implementations should have given a subgraph with the specified number of nodes, but instead only contained a 
fraction of this. For example, the `bfs` implementation only gave a subgraph with 20-30% of the expected number of nodes
(`bfs-100` had around 30 nodes, while `bfs-250` had only 50). These numbers are only an estimate based on later testing.

# Results

With this deficit the results have to be viewed differently: Instead of comparing only the subsampling methods, we have 
to consider the (big) difference in train data associated with the training runs.

For example, the `random_walk-250` method gave, even if seemingly unstable, results close to that of `node_degree-100` 
while containing only around 50% of the nodes and therefore way smaller train dataset. This could show, that randomly 
sampling the graph gives worse results.

But the graphs only show the performance based on the subgraph. To make more concrete predictions, we have to add more 
scores to evaluate on and also (at least once) evaluate on the complete graph.