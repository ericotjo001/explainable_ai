This package contains the code to generate the results in [Generalization on the Enhancement of Layerwise Relevance Interpretability of Deep Neural Network](https://arxiv.org/abs/2009.02516).

For instructions to execute the code, see or run python main.py. The commands will be printed.

For external dependencies, see utils/utils.py. Please install the packages (they are common).

The main commands used to obtain the results in the papers are:<br>
```
python main.py --mode custom_sequence --submode smallnet_mnist --subsubmode RXXXX1
python main.py --mode custom_sequence --submode alexnet_mnist --subsubmode RXXXX1
```

The command for downloading data into the right place is included in the code. In fact, data will be automatically downloaded when you run the main process, if it does not exist in the directory yet.

No support will be provided for this package.

Pre-trained models: this project will no longer be updated, link to pre-trained model has been removed.
