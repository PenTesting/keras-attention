# Attention RNNs in Keras

Implementation and visualization of a custom RNN layer with attention in Keras .

```
Dataset - https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
```

## Setting up the repository

0. Make sure you have Python 3.4+ installed.

1. Clone this repository to your local system

```
git clone -b My_Dialog https://github.com/sunnysai12345/keras-attention.git
```

2. Install the requirements
(You can skip this step if you have all the requirements already installed)

We recommend using GPU's otherwise training might be prohbitively slow:

```
pip install -r requirements-gpu.txt
```

If you do not have a GPU or want to prototype on your local machine:

```
pip install -r requirements.txt
```


## Already existing data from stanford manual dialogue dataset

Download the files from https://drive.google.com/open?id=0ByDWT4kqM592bU9jOVhUSXZTa28 and place them in data folder.
1. `training.csv` - data to train the model
2. `validation.csv` - data to evaluate the model and compare performance
3. `vocabulary.json` - vocabulary file

`cd` into `data` and run

## Running the model

We highly recommending having a machine with a GPU to run this software, otherwise training might be prohibitively slow. To see what arguments are accepted you can run `python run.py -h` from the main directory:

```
usage: run.py [-h] [-e |] [-g |] [-p |] [-t |] [-v |] [-b |]

optional arguments:
  -h, --help            show this help message and exit

named arguments:
  -e |, --epochs |      Number of Epochs to Run
  -g |, --gpu |         GPU to use
  -p |, --padding |     Amount of padding to use
  -t |, --training-data |
                        Location of training data
  -v |, --validation-data |
                        Location of validation data
  -b |, --batch-size |  Location of validation data
```

All parameters have default values, so if you want to just run it, you can type `python run.py`. You can always stop running the model early using `Ctrl+C`.

## Visualizing Attention

You can use the script `visualize.py` to visualize the attention map. We have provided sample weights and vocabularies in `data/` and `weights/` so that this script can run automatically using just an example. Run with the `-h` argument to see what is accepted:

```
usage: visualize.py [-h] -e | [-w |] [-p |] [-hv |] [-mv |]

optional arguments:
  -h, --help            show this help message and exit

named arguments:
  -e |, --examples |    Example string/file to visualize attention map for If
                        file, it must end with '.txt'
  -w |, --weights |     Location of weights
  -p |, --padding |     Length of padding
  -hv |, --human-vocab |
                        Path to the human vocabulary
  -mv |, --machine-vocab |
                        Path to the machine vocabulary
```

The default `padding` parameters correspond between `run.py` and `visualize.py` and therefore, if you change this make sure to note it. You must supply the path to the weights you want to use and an example/file of examples. An example file is provided in `examples.txt`. 


### Help

Start an issue if you find a bug or would like to contribute!


### Acknowledgements

As with all open source code, we could not have built this without other code out there. Special thanks to:

1. [rasmusbergpalm/normalization](https://github.com/rasmusbergpalm/normalization/blob/master/babel_data.py) - for some of the data generation code.
2. [joke2k/faker](https://github.com/joke2k/faker) for their fake data generator.
3. datalogue 

### References

Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 
["Neural machine translation by jointly learning to align and translate." 
arXiv preprint arXiv:1409.0473 (2014).](https://arxiv.org/abs/1409.0473)
