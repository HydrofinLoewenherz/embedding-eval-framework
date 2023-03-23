# Towards the evaluation of graph embeddings with deep decoders

The Embedding-Eval-Framework is the bachelor thesis project of Paul Wagner.
**This is an unfinished state!**

## Abstract

This thesis presents a method for evaluating graph embeddings based on the performance of decoders. 
Graph embeddings are widely used for different tasks [...]. As such it is of interest to select an embedder, that is 
able to effectively embed graph structures.

The thesis examines how the learning efficiency and precision of decoders can be used as a score on how effectively 
information is stored in an embedding. The idea being, that if an embedding doesn't store its information effectively, 
than a decoder isn't able to learn the embedding. On the contrary, should an embedding store 
information effectively, than a decoder should be able to learn the embedding.

It provides a ready-to-use jupyter notebook to validate presented test results and enables third-parties to test on
other embeddings.

Most embeddings use general vicinity to represent information about connections. A generic decoder has the advantage, 
that it can learn embeddings that use different measures to represent information.


## Introduction

### Motivation

TBD
- Why embeddings? (What are they used for?)
- Differentiate between deterministic and machine learning based algorithms
- We see a difference in dimensionality in the result. Deterministic algorithms are generally low dimensional while ML algorithms produce high dimensional embeddings
- why do we need this


### Related Work

TBD


### Outline

TBD


## Preliminaries

### Graphs and Embeddings

A graph is a tuple `G = (V, E)` where `u, v \in V` is a set of nodes and `(v, u) = (u, v) = e \in E` a set of 
(undirected) edges between the nodes. A node represents a piece of information, while edges represent known relations 
between the pieces of information.

An embedder maps the sometimes unbound information vector to a fixed length vector, the feature vector, 
`f \in [0, 1]^n = F` where `n` is the dimension of the embedding. The embedding contains information about the edges in 
the original graph.


### Decoders

The decoder used in the thesis is a binary classificator. Given the features `f_u, f_v \in F` of two embedded nodes 
`u, v \in V`. It maps `[*f_u, *f_v]` to a number `p`, its prediction.

TBD
- loss function
- neural nets
- optimizer
- dataset splitting


## Evaluation

### Setup

The evaluator framework operates in four steps
1. Generate dataset from graph and embedding
2. Split dataset in train-, validation- and test-set
3. Train the model on the train-set while using the validation-set for early stopping
4. Test the model on the test-set to get a general score for the whole embedding

The dataset is split for two reasons
1. Ensure that no overfitting takes place.
   - An overfitted model could give good scores to bad embeddings as long as the embedding is not too complex
2. Reduce dataset size and class imbalance
   - A graph with `n` nodes contains up to `n^2` edges
   - The dataset size is too large for big graphs to loop all datapoints in each epoch
   - The number of edges in real world (power law) graphs increases [...], resulting in a class imbalance

TBD: graphs with number of edges to graph size of girgs and random geometric graphs


### Subgraph sampling

As described before, we have to look out for the dataset size and class imbalance. Would we just use the train-set 
directly, we would see many datapoints that are very similar. This would increase at least the train time and could even
lower the framework result performance. To mitigate these problems, the framework uses a new sub-graph for each epoch. 
These sub-graphs can be chosen with a variety of different subsampling methods. To name a few
- Weighted Random Selection
- BFS/DFS
- Random Walk
- Forest Fire

The weighted random sampling method weights all datapoints. Then a random but fixed length sample is chosen from the
datapoints. Weights could be chosen uniformly, giving all datapoints the same chance to appear. This would solve the 
dataset size problem but keeps the class imbalance untouched. The datapoints could be weighted based on their class.
Every datapoint would have a weight of `1 / size_of_datapoint_class`. the sum of all weights is here `2`, but based on 
the implementation used to sample, this is not a problem.
The weighted sample would have fixed size and balanced classes. But it doesn't represent the graph structure. Depending 
on the embedding used, this could be detrimental.

Another sampling method is breadth first search (BFS) and depth first search DFS. These search algorithms traverse a 
graph over its edges. The algorithms work as follows:
1. use two sets: `visited` and `visit`
2. pick one starting node at random and add it to the `visit` set
3. pop the first (for BFS) or last (for DFS) node from the `visit` set as the `current` node
4. add the `current` node to the `visited` set
5. push all neighbors of the `current` node that are not already inside `visited` to the back of the `visit` set
6. if the `visit` set is empty, choose a random node that is not in the `visited` set and add it to the `visit` set
7. stop if `visited` has reached the wanted size otherwise loop back to step (3)

Both of these sampling methods generate clusters of sub-graphs. The DFS generates lines while the BFS generates disc 
like structures. We need step (6) in case the graph consists of multiple smaller unconnected graphs or the search drives 
itself into a corner.
A negative element of these sampling methods id that we need to keep two sets of nodes to not have duplicates. Depending 
on the implementation of the random sampling it might not need any set. These sets can get, depending on the graph 
structure and size, quite big and have to be considered for time and space constraints.

We can eliminate the `visit` set by using a more relaxed sampling method such as random walk. Random walk at its base 
operates as follows
1. use one set `visited`
2. pick one starting node at random as the `current` node and add it to the `visited` set
3. choose one neighbor of the `current` node at random as the new `current` node and add it to the `visited` set
4. stop if `visited` has reached the wanted size otherwise loop back to step (3)

This sampling method has the advantage of only using one set. Additionally, walks can go "backwards" over nodes that 
were already visited, so it won't drive itself into a corner. But we can no longer have a fixed runtime constraint, we 
could even run in a "loop" for a considerable amount of time. And we cannot detect if there are even any new nodes 
reachable from the current position. The framework uses an extended version of the basic random walk that mitigates and 
eliminates some shortcomings, but more on that later.

TBD: describe forest fire

As described before, the framework uses an extended version of the basic random walk. It tries to tackle two shortcomings
1. getting stuck in disconnected graph parts
2. running in a loop for too long

Both shortcomings are solved by introducing jumps. The first change is that a relative threshold `bordem_pth` is 
introduced that tracks the amount of steps taken without seeing a previously not visited node relative to the amount of 
nodes that we want for the subgraph. The exact value of the relative threshold is not of interest. Depending on the task 
the framework uses constant values at around `0.8`. (It is mostly used to ensure that the algorithm doesn't get stuck)

In conjunction with that threshold there is also a jump probability `alpha` for every walk to be a jump instead.
The jump probability allows us to influence the number of disconnected (or loosely connected) components in the subgraph.
Setting the jump probability to `0.0` gives us a method close to the base random walk. Setting is to `1.0` effectively
gives us a (uniform) random sampling.

This gives us two (numerical) degrees of freedom for our subsampling: the jump probability and the subgraph size. 
Depending on the structure of the graph that is sampled we might want different settings. It also allows us to use grid 
search to find appropriate settings without completely changing sampling methods. Later more to actual tests with 
different settings on different graph structures.

TBD: subgraph generation source code


### Scoring

To score embeddings, we use the average precision. The Average precision is calculated with the so called 
precision-recall-curve.

TBD
- write more about precision-recall-curve and f1-score


### Embeddings

#### Random Geometric Graphs

Random geometric graphs are generated by generating random positions for nodes and connecting two nodes, when the 
distance between them is less than a threshold. The feature of a node would be its position (in 2d-space).

TBD: image of graph

Advantages are that the graph can be easily generated and that sub-graphs of random geometric graphs are themselves
random geometric graphs.


TBD
- other characteristics?


#### Girgs

Geometric inhomogeneous random graphs (GIRG) are graphs with the power-law characteristics. A girg uses its node 
positions and weights to generate edges. This enables us to draw the graph in a more comprehensive way while testing 
graphs with real-world characteristics.

We will later see, that girgs yield noticeably lower scores than random geometric graphs. Reasons for that could be the 
power-law characteristic, or that the edges "wrap around" on the plane.


#### Random Graphs

Random graphs are graphs, where the features don't represent the graph structure. They are used to represent inefficient 
embeddings as no information about the structure can be found in the features. We want the framework to give such graphs 
low scores. Would they not receive such low scores, then we could not be sure, that good scores indicate efficient 
embeddings.

To generate such graphs, the features of an embedding that is known to yield high scores are replaced with random 
features. Later we will see, that random geometric graphs yield quite high scores. So they will be used as a basis for
random graphs.


### Framework

The thesis also consists of a ready-to-use python framework to validate the provided test results and generate new 
tests. The framework is modular and intended to be used in the future to get further insights into the efficiency of 
graph embeddings.
The source code is open source on GitHub.


### Results

#### General Bounds

With the framework we want to be able to say: *if an embedding yields a high score, then it is efficient*. This is 
equivalent to *if an embedding is inefficient, then it yields a low score*. It is not feasible to look for all efficient
or inefficient embeddings and score them. Instead, we show the following necessary conditions
1. an approximate lower bound for scores of known inefficient embeddings
2. an approximate higher bound for scores of known efficient embeddings

We use the aforementioned random graphs for the lower bound and random geometric graphs for the higher bound. All tests
use embeddings in 2d-space. That is purely made for the tests to be able to show comprehensible graphics of the test 
results and not a constraint of the framework nor a performance decision.

First we look at the random graph. 

TBD: image of random graph; average precisions curves on validation set for different alphas and epoch sizes; best complete reconstruction
TBD: describe graph and how the score is low

Now we look at the random geometric graph.

TBD: image of random geometric graph; average precisions curves on validation set for different alphas and epoch sizes; best complete reconstruction
TBD: describe graph and how the score is high

As we just saw, the framework fulfills its necessary conditions. Next we look more closely into characteristics of the framework


#### Subgraph Characteristics

Question: How does the jump probability influence connectivity for sub-graphs and class imbalance?
Question: What (other) characteristics can we detect for sub-graphs with different alpha and size?


#### Large Graphs

Question: How does the runtime increase for bigger initial graphs?


#### Random Geometric Graphs

Question: Does it learn the threshold function for connecting nodes?
Question: How does the f1-score compare to the threshold used by the model (get threshold from model with statistics)


#### Girgs

Question: How does it perform on and score girgs?


#### Epoch Subsampling

Question: Does (and how much) the epoch subgraph sampling speed up the framework?
Question: Does (and how much) the epoch subgraph sampling improve the score?


## Conclusion

TBD


## Future

TBD
- power-law graph without "wrap around"
- other net structures
- preprocess node positions / features
  - directly give distance as feature
  - use polar coordinates


# Writing
- best 10-20 pages
- until 18.

# Random Notes and Topics

TBD
- expect higher `size` results in better representation of graph structure but also higher class imbalance and run time
- expect lower `alpha` results in higher connectivity of components but higher run time
  - *could be different based on embedding used, maybe test on multiple settings (like precision-recall-curve)*
- loss function
- precision-recall-curve and average precision
- f1-score and threshold
- use weights

