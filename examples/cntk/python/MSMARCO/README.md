# R-NET reimplemention
## Prepare data
### Download the MS MARCO  dataset
```
https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz
https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz
https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz
```
### Download GloVe vectors
```
http://nlp.stanford.edu/data/glove.840B.300d.zip
https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt
```
download and unzip glove.840B.300d.zip in script/
### Download NLTK punkt
```
python3 -m nltk.downloader -d $HOME/nltk_data punkt
```
or if you use windows
```
python -m nltk.downloader -d %USERPROFILE%/nltk_data punkt
```

### Run script to process data
1. edit on of config*.pys or let it remain default.
2. `python convert_msmarco.py source dest`
	It should create `train.tsv`, `dev.tsv`, and `test.tsv` and `vocab.pkl`.
	It does a fair amount of preprocessing therefore converting data to cntk text format reader starts from these files
3. run `python tsv2ctf.py`
	It creates a `vocabs.pkl` and `train.ctf`, `val.ctf`, and `dev.ctf`

### Train a model
```
python train_rnet.py --logdir ./logs/
 mpirun -npernode 4 python train_rnet.py --logdir ./logs/
```
### Generate a Result and Test It
```
python train_pm.py --test dev.tsv
```
If you want to test it , please refer to [msmarco](http://www.msmarco.org/submission.aspx)
```
sh ./run.sh dev_as_references.json pm.model_out.json

{'testlen': 108948, 'reflen': 111890, 'guess': [108948, 103691, 99301, 94994], 'correct': [41880, 22693, 18713, 16777]}
ratio: 0.9737063187058631
############################
bleu_1: 0.3741621473845382
bleu_2: 0.28231995043944164
bleu_3: 0.24452041494674448
bleu_4: 0.2239025217932479
rouge_l: 0.302060927811
############################
```