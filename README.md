# Improving Variational Autoencoder for Text Modelling withTimestep-Wise Regularisation

This repository includes the source code to reproduce the results presented in the paper [Improving Variational Autoencoder for Text Modelling with Timestep-Wise Regularisation](https://arxiv.org/pdf/2011.01136.pdf) (COLING 2020)


## Contents
1. [Language modelling](#Language-modelling)

2. [Dialogue response generation](#Dialogue-response-generation)

    2.1. [Dialogue response generation on Switchboard](#Dialogue-response-generation-on-Switchboard)

    2.2. [Dialogue response generation on DailyDialog](#Dialogue-response-generation-on-Dailydialog)


## 1. Language modelling



To train the TWR-VAE on PTB/Yelp/Yahoo

```
cd lang_model/
python main.py -dt ptb/yelp/yahoo --z_type normal
```

To load trained model
```
python main.py -dt ptb/yelp/yahoo -l --model_dir path-to-the-trained-model/
```

To train the TWR-VAE-mean or TWR-VAE-sum on PTB/Yelp/Yahoo

```
python main.py -dt ptb/yelp/yahoo --z_type mean/sum
```

To train the TWR-VAE-LSTM-last25 or TWR-VAE-LSTM-last50 or TWR-VAE-LSTM-last75 on PTB/Yelp/Yahoo

```
python main.py -dt ptb/yelp/yahoo --z_type normal -par --partial_type last25/last50/last75
```

## 2. Dialogue response generation

Use pre-trained Word2vec: download Glove word embeddings `glove.twitter.27B.200d.txt` from https://nlp.stanford.edu/projects/glove/ and save it to the `./data` folder. The default setting use 200 dimension word embedding trained on Twitter.

### 2.1 Dialogue response generation on Switchboard

To train TWR-VAE on Switchboard
```
cd dialogue_switchboard/
python train_swda.py
```

### 2.2 Dialogue response generation on Dailydialog

To train TWR-VAE on Dailydialog
```
cd dialogue_dd/
python train_dailydial.py
```

## Acknowledgements
Thanks for the code published on Github repositories:
- https://github.com/guxd/DialogWAE
- https://github.com/fangleai/Implicit-LVM

```
@inproceedings{Li_TWRVAE_2020,
  title={Improving Variational Autoencoder for Text Modelling withTimestep-Wise Regularisation},
  author={Li, Ruizhe and Li, Xiao and Chen, Guanyi and Lin, Chenghua},
  booktitle={COLING},
  year={2020}
}
```
