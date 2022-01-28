# Neuron Dependency Graphs

Code for paper **Neuron Dependency Graphs: A Causal Abstraction of Neural Networks**

Citation:

```
@inproceedings{hu2022ndg,
  author    = {Yaojie Hu and Jin Tian},
  title     = {Neuron Dependency Graphs: A Causal Abstraction of Neural Networks},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning, {ICML} 2022},
  publisher = {Proceedings of Machine Learning Research},
  year      = {2022}
}
```

## Abstract

We discover that neural networks exhibit approximate logical dependencies among neurons, and we introduce Neuron
Dependency Graphs (NDG) that extract and present them as directed graphs. In an NDG, each node corresponds to the
boolean activation value of a neuron, and each edge models an approximate logical implication from one node to another.
We show that the logical dependencies extracted from the training dataset generalize well to the test set. In addition
to providing symbolic explanations to the neural network's internal structure, NDGs can represent a Structural Causal
Model. We empirically show that an NDG is a causal abstraction of the corresponding neural network that
"unfolds" the same way under causal interventions using the theory by Geiger et al. (2021). Code is available
at https://github.com/phimachine/ndg

## Installation

We use PyTorch and provide conda installation instructions, assuming Linux operating system. MongoDB needs to be
installed for logging.

```bash
conda create -y -n ndg python=3.9
conda activate ndg
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y -c huggingface transformers
conda install -y pandas scipy pymongo
pip install ray[core] ml_collections sentence-transformers datasets imageio matplotlib tensorboardX networkx tqdm
pip install pytest pytest-pycharm graphviz tensorboard pydot
```

Before you run the python code, you need to set the global variables for your data path and checkpoint path
in `global_params.py`

Download CUB200 and ViT models and put them in respective directories based on TransFG codebase included in this
repository. Special thanks to TransFG authors.

Use ```tests/test_dataset.py``` and ```tests/test_model.py``` to ensure correct installation.

## Extract neuron dependency graphs
First, train the models. Run
```bash
python train_models.py
```

Build graphs and perform interchange intervention

```bash
python align.py
```

If you prefer to specify the configurations, modify the code to use argument parser.

Scripts used to perform analysis and create tables/plots used in the paper are included in the repository
`analysis.py`, `table_plots.py`, `plots.R`, and `plot.R.`
Dependencies for R can be found within the files.

## LICENSE

MIT