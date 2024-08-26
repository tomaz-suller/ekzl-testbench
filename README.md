# ezkl-testbench

Code for collecting [`ezkl`](https://ezkl.xyz/) performance metrics on 
[Zero-Knowledge Proof](https://en.wikipedia.org/wiki/Zero-knowledge_proof) generation over inference tasks
for multiple deep neural network models following a previously-introduced methodology[^1].

## Usage

The entrypoint [`main.py`](src/main.py) provides a command-line interface powered by [Hydra](https://hydra.cc/),
which exposes configuration options that can be listed using the `--help` flag:
```shell
$ python src/main.py --help
```

Details on important settings are provided below.

### `model`
Controls which neural network model is used for proof generation and verification.
It accepts a list of strings, corresponding to the names of the models to be used.
For a list of supported models, see [`model.py`](src/model.py). 
If left unspecified (default), all models will be used.

[^1]: Cerioli, L.: A survey on Zero Knowledge proofs and their applications on machine
learning. Master’s thesis, Politecnico di Milano, Scuola di Ingegneria Industriale e
dell’Informazione (5 2023), https://hdl.handle.net/10589/209893
