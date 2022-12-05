# Embedding Eval Framework
The Embedding-Eval-Framework is the bachelor thesis project of Paul Wagner.

## Motivation
- Why embeddings? (What are they used for?)
- Differentiate between deterministic and machine learning based algorithms
- We see a difference in dimensionality in the result. Deterministic algorithms are generally low dimensional while ML algorithms produce high dimensional embeddings


## Goal
- Create an evaluator framework that evaluates, if a generated embedding is "good"
- The evaluator takes the original graph (graph with nodes and edges) and an embedding (node positions)
- It tries to generate a decoder with ML that is capable of predicting graph edges
- Idea: if the decoder-setup is generally chosen "good", then it should be able to generate a decoder, that is able to predict edges. If the generated decoder cannot predict edges, then the embedding/embedder was chosen poorly.
- Question: How to make a good decoder-setup.
- This workflow will be presented in a visual application


## Roadmap
- [ ] Decide on an application framework (maybe [electron-app](https://www.electronjs.org/de/), [flask-api](https://github.com/pallets/flask) with [progress](https://github.com/simonw/datasette-app/issues/109) and [decoder-framework](?) )
- [ ] Answer: How to make a good decoder-setup?


___

*Further documentation and so on will be added later.*