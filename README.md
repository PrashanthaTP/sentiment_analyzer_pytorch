# Tweet Sentiment Classification

<!--[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)-->
<!-- for badges visit : https://github.com/alexandresanlim/Badges4-README.md-Profile-->

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

---

This repo contains Python implementation of Sentiment Analysis of Tweets using Neural Networks with Pytorch.

## Overview

- Framework used : Pytorch
- Architectures : LSTM,CNN(2D), and a hybrid of these two
- Dataset : [Sentiment140](http://help.sentiment140.com/for-students/)
- Final accuracy : 82.157433 %
- [Report](docs/Sentiment_Analysis_Report.pdf)

### Scripts

- Preprocessing : [ setup_dataset.py ](main/setup_dataset.py)
- Embedding Training : [ train_embedding.py ](main/train_embedding.py)
- Hybrid Model Training : [ train_hybrid.py ](main/train_hybrid.py)

## #Architectures :

### CNN based Model

![CNN architecture](docs/cnn_arch.png)

### LSTM based Model

![LSTM architecture](docs/bilstm_arch.png)

### Hybrid Model

![Hybrid Model](docs/hybrid_arch_v2.png)

## #Results

### Accuracy

- CNN based model : 79.28%
- LSTM based model : 82.5%
- Hybrid Model : 82.15%

### Confusion Matrices

- For cnn based model

  ![CNN based model cm](docs/cnn_confusion_matrix.png)

- For LSTM based model

  ![LSTM based model cm](docs/lstm_confusion_matrix.png)

- For hybrid model

  ![Hybrid Model cm](docs/hybrid_confusion_matrix.png)

# References

---

- Dataset

```
@ONLINE {Sentiment140,
    author = "Go, Alec and Bhayani, Richa and Huang, Lei",
    title  = "Twitter Sentiment Classification using Distant Supervision",
    year   = "2009",
    url    = "http://help.sentiment140.com/home"
}
```
