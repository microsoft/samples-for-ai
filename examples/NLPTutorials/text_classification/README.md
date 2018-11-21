#A PyTorch based demo for text classification.
## Introduction
Text classification is a core problem to many applications like e.g. spam filtering, email routing, book classification. 
The task aim to train a classifier using labelled dataset containing text documents and their labels, which can be 
a web page, paper, email, reviewer etc. 

This demo is designed to assign documents to 20 different newsgroups, each corresponding to a different topic using a 
text classification method. The newsgroups data is simply the raw text:

```
From: cmk@athena.mit.edu (Charles M Kozierok)
Subject: Re: bosio's no-hitter
Date: 24 Apr 1993 03:59:58 GMT
Organization: Massachusetts Institute of Technology

I watched the final inning of Bosio's no-hitter with several people at work. After Vizquel made that barehanded 
grab of the chopper up the middle, someone remarked that if he had fielded it with his glove, he wouldn't have 
had time to throw Riles out. Yet, the throw beat Riles by about two steps. I wonder how many others who watched 
the final out think Vizquel had no choice but to make the play with his bare hand.

In this morning's paper (or was it on the radio?), Vizquel was quoted as saying that he could have fielded the 
ball with his glove and still easily thrown out Riles, that he barehanded it instead so as to make the final 
play more memorable.  Seems a litle cocky to me, but he made it work so he's entitled. i guess so.

still, that's kind of a stupid move, IMO. he'd be singing a different tune if he had booted it, and the next 
guy up had hit a bloop single. stranger things have happened (hey, i used to be a big Dave Stieb fan...) and 
unfortunately, there's no such thing as an "unearned hit". :^)
cheers,
-*-
charles
```

This text describes about baseball. The goal of this demo is to learn to tag it with rec.sport.baseball. Category 
includes 6 major categories and 20 fine-grained categories, See the section on Dataset for more information about 
labels. 
 

## Dataset
The 20 Newsgroups data set contains 20000 newsgroup documents collected across 20 different newsgroups. Here is a list 
of the 20 newsgroups partitioned (more or less) according to subject matter:

<table border="1" cellpadding="0">
    <tbody>
        <tr>
            <td>
                <p align="left">
                    comp.graphics<br>
                    comp.os.ms-windows.misc<br>
                    comp.sys.ibm.pc.hardware<br>
                    comp.sys.mac.hardware<br>
                    comp.windows.x
                </p>
            </td>
            <td>
                <p align="left">
                    rec.autos<br>
                    rec.motorcycles<br>
                    rec.sport.baseball<br>
                    rec.sport.hockey
                </p>
            </td>
            <td>
                <p align="left">
                    sci.crypt<br>
                    sci.electronics<br>
                    sci.med<br>
                    sci.space
                </p>
            </td>
        </tr>
        <tr>
            <td>
                <p align="left">
                    misc.forsale
                </p>
            </td>
            <td>
                <p align="left">
                    talk.politics.misc<br>
                    talk.politics.guns<br>
                    talk.politics.mideast
                </p>
            </td>
            <td>
                <p align="left">
                    talk.religion.misc<br>
                    alt.atheism<br>
                    soc.religion.christian
                </p>
            </td>
        </tr>
    </tbody>
</table>

## Requirements
- [PyTorch](http://pytorch.org/) Deep learning library, should install follow the offical web.
- numpy==1.15.3
- Pillow==5.3.0
- requests==2.20.0
- scikit-learn==0.20.0
- scipy==1.1.0
- six==1.11.0
- sklearn==0.0
- torchtext==0.2.3
- tqdm==4.28.1
- urllib3==1.24

## Runtime requirements
Using the default parameters: 
- GPU mode need about 2GB cuda memory.
- CPU mode takes tiny CPU utilization and less than 500 MB of main memory usage. 

## Usage

```
python/python3 text_classification.py --model_name <TextCNN|LSTMSelfAttentionHighway> --batch_size 128 --epochs 3
```

This downloads the following data automatically:
  - [Twenty Newsgroups Dataset](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) (This dataset consists of 20000 messages taken from 20 newsgroups)


## Results

with the two methods we provide(TextCNN, LSTMSelfAttentionHighway), you will get the following results on valid dataset:

```
methods                         accuracy        loss
TextCNN                         97%             0.000836
LSTMSelfAttentionHighway        93%             0.003445
```

## FAQ
#### I am getting out-of-memory errors, what is wrong?
You are likely to encounter out-of-momery issues using the default parameters if your GPU momery less than 2GB. 
The factors that affect memory usage are batch_size, hidden_dim, embed_dim, filters, and so on, you can try to reduce
these parameters.

## Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>
* [A pytorch implementation of CNNText classification](https://github.com/Shawn1993/cnn-text-classification-pytorch)
* [A pytorch implementation of LSTM + Self Attention classification](https://github.com/nn116003/self-attention-classification)

## Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
* [A structured self-attention sentence embedding](https://arxiv.org/pdf/1703.03130.pdf)
* [Highway Network](https://arxiv.org/abs/1505.00387)
* [NewsWeeder: Learning to Filter Netnews](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=F78E1658C9E109F677438D805DF0BF9E?doi=10.1.1.22.6286&rep=rep1&type=pdf)

## License
MIT
