## Fast approximations for geometric optimal transport

This repository contains the implementation of several fast approximate algorithms for geometric optimal transport (OT). Besides that, we allow to assemble these algorithms in pipelines that can be used for fast nearest neighbor search with respect to the OT distances. Most notably, we implement the classic QuadTree algorithm (Indyk, Thaper 2003) as well as its novel modification, which we call FlowTree. We also implement several other algorithms, see below for the references.

Authors: [Arturs Backurs](https://www.mit.edu/~backurs/), [Yihe Dong](https://yihedong.me/), [Piotr Indyk](https://people.csail.mit.edu/indyk/), [Ilya Razenshteyn](https://www.ilyaraz.org/), [Tal Wagner](http://www.mit.edu/~talw/).

This code accompanies our [paper](https://arxiv.org/abs/1910.04126) "Scalable Nearest Neighbor Search for Optimal Transport".

### Installation instructions

Start with cloning the repo. **It's really crucial to clone it with all the submodules as follows:**

```
git clone --recurse-submodules https://github.com/ilyaraz/ot_estimators.git
```

#### Linux (Ubuntu)
First, install the necessary dependencies:
```
sudo apt-get install -y g++ make cmake python3-dev python3-pip python3-numpy
sudo pip3 install cython pot
```
Then, to build the C++ extension, do the following:
  1. Create an empty directory and move there
  1. Run `cmake <path to the "native" directory from this repo>`
  1. Run `make`
  1. Copy the resulting `.so` file to the `"python"` directory from this repo, and move there
  1. To check that the extension loads, fire up ``python3`` and type ``import ot_estimators`` in the REPL that just opened

#### OS X, Windows (Cygwin)

For OS X and Cygwin, the steps are pretty much the same, except that you need to install the dependencies differently, through a combination of `pip3` and `brew` for OS X and the standard package manager for Cygwin. Note that in case of Cygwin, the result of running `make` has extension `.dll` rather than `.so`.

### Downloading a sample 20news dataset

Download the [archive](https://flowtree.s3-us-west-1.amazonaws.com/20news.tar.gz) and unpack it,
the result should be a directory called `20news`.

### Running the code

The file `algorithms.py` contains the implementations of the below algorithms:
  * `mean`: (Kusner et al., 2015), called there Word Centroid Distance (WCD)
  * `overlap`: baseline described in our paper, see below
  * `quadtree`: (Indyk and Thaper, 2003)
  * `flowtree`: a novel modification of QuadTree described in our paper, see below
  * `rwmd`: (Kusner et al., 2015), called there Relaxed WMD (R-WMD)
  * `lcwmd`: (Atasu and Mittelholzer, 2019), called there ACT
  * `sinkhorn`: (Cuturi 2013)
  * `exact`: exact OT computation using a combinatorial flow algorithm called in the `pot` library

Besides running the above individual algorithms, one can run a nearest neighbor search pipeline composed from several of these algorithms as follows.
```
python3 pipeline.py <path> <method list> <candidate number list> [<method parameter>]
```
  * `path` is a path to the data folder (e.g., the above `20news`, see `pipeline.py` for the exact data format).
  * `method list` is hyphen-separated list of pipeline method in the order they would run.
  * `candidate number list` is a hyphen-separated list of the number of output candidates of each method in the pipeline, corresponding to the method list.
  * `method parameter` (default is 1) is the number of iterations that would be used for both Sinkhorn and LC-WMD, if either is present in the pipeline method list (otherwise this argument has no effect).

Example:
```
python3 pipeline.py "./data/20news/" mean-sinkhorn-exact 1000-40-1 9
```
This runs a pipeline on the 20news dataset, which starts with mean narrowing the input dataset down to 1000 candidates, followed by Sinkhorn with 9 interations narrowing further to 40 candidates, ending with exact EMD computation narrowing down to a one candidate.
Output:
```
Nearest neighbors found by pipeline for query 0: [5286]
Nearest neighbors found by pipeline for query 1: [9950]
Nearest neighbors found by pipeline for query 2: [8972]

=== Output ===
Total time:  1.0380972082614899
Final accuracy: 0.901
Accuracy after each stage of the pipeline: [0.902, 0.902, 0.901]
```

### References

If you use our implementation or the FlowTree algorithm in your research, please cite our paper:

```
@article{FlowTree,
  title={Scalable nearest neighbor search for optimal transport},
  author={Backurs, Arturs and Dong, Yihe and Indyk, Piotr and Razenshteyn, Ilya and Wagner, Tal},
  journal={arXiv preprint arXiv:1910.04126},
  year={2019}
}
```

#### Other works

Besides the above paper, our implementation contains algorithms developed in the following papers:
  1. Indyk, Thaper &ndash; Fast image retrieval via embeddings, 3rd international workshop on statistical and computational theories of vision, 2003 (QuadTree).
  1. Cuturi &ndash; Sinkhorn distances: Lightspeed computation of optimal transport, NIPS 2013 (Sinkhorn).
  1. Kusner, Sun, Kolkin, Weinberger &ndash; From word embeddings to document distances, ICML 2015 (Mean, R-WMD).
  1. Atasu, Mittelholzer &ndash; Linear-complexity data-parallel earth mover’s distance approximations, ICML 2019 (LC-WMD).
