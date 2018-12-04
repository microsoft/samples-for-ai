#A PyTorch based demo for language model.
## Introduction
The language model is to simulate human language learning through specific models, which can be used in language generation, speech recognition, machine translation, information retrieval and other applications.

This demo is designed to learn language model from WikiText2 dataset and generate sentence from the trained model 

## Dataset
The WikiText2 dataset is an English data of 10,000 words extracted from Wikipedia's high-quality articles and benchmarks. Each vocabulary also retains the original article that produced the vocabulary.

This data is simply the raw text like:
```
 = Valkyria Chronicles III = 
 
 Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . <unk> the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven " . 
 The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more <unk> for series newcomers . Character designer <unk> Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . 
 It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .
```

### Requirements
- [PyTorch](http://pytorch.org/) Deep learning library, should install follow the offical web.
- numpy==1.15.3
- torchtext==0.2.3

### Runtime requirements
Using the default parameters: 
- GPU mode need about 1GB cuda memory.
- CPU mode takes tiny CPU utilization and less than 500 MB of main memory usage. 

## Usage

```
python/python3 language_model.py --batch_size 20 --epochs 20 --gen_word_len 15
```

This downloads the following data automatically:
  - [WikiText2 Data Set]('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip')

## Results
with the methods we provide(GRU, LSTM) and use the default parameters, you will get the following results on valid and test dataset:
```
[Epoch: 20] val-loss: 4.48 | val-pp:87.80
[!] saving model
[!] training done
test-loss: 4.41 | test-pp:82.04
[!] generating...
The games is approximately one of the website 's first in a meeting
```

## FAQ
#### 1.I am getting out-of-memory errors, what is wrong?
You are likely to encounter out-of-momery issues using the default parameters if your GPU momery less than 1GB. 
The factors that affect memory usage are batch_size, hidden_dim, embed_dim and so on, you can try to reduce
these parameters.


### Acknowledgements
* [Pytorch team](https://github.com/pytorch/pytorch#the-team) for Python library<br>

### License
MIT
