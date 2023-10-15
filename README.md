# Overview
This repository contains JAX code for the implementation of the Deep Covering Eigenoptions (DCEO) algorithm.

**[Deep Laplacian-based Options for Temporally-Extended Exploration](https://proceedings.mlr.press/v202/klissarov23a/klissarov23a.pdf)**

by [Martin Klissarov](https://mklissa.github.io) and [Marlos C. Machado](https://webdocs.cs.ualberta.ca/~machado/). 

![dceo](https://github.com/mklissa/deco_dopamine/assets/22938475/285c7ed1-f1a3-499f-8655-5802ee4738c9)

DCEO is based on the idea of the [Representation Driven Option Disovery](https://medium.com/@marlos.cholodovskis/the-representation-driven-option-discovery-cycle-e3f5877696c2) cycle where options and representations are iteratively refined and bootstrap from eachother. In this work, we argue that the Laplacian representation (also referred to as [Proto-Value Functions](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PVF.pdf)) as it encodes the topology of the environment at various timescales. 

In the paper we investigate DCEO across a variety of environments where it outperforms hierarchical and flat baselines. In this repository, we can only share the code with respect to the Montezuma's Revenge experiments, which are built on the [Dopamine](https://github.com/google/dopamine) codebase.

# Installation
