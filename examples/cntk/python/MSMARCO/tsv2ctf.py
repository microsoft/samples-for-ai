from collections import defaultdict
from itertools import count, zip_longest
from config import *
import pickle
import numpy as np

word_count_threshold = data_config['word_count_threshold']
char_count_threshold = data_config['char_count_threshold']
word_size = 50

sanitize = str.maketrans({"|": None, "\n": None})
tsvs = 'train', 'dev', 'test'
unk = '<UNK>'
pad = '<PAD>'
EMPTY_TOKEN = '<NULL>'
START_TOKEN = '<S>'
END_TOKEN = '</S>'
WORD_START='<W>'
WORD_END='</W>'

# pad (or trim) to word_size characters
pad_spec = '{0:<%d.%d}' % (word_size, word_size)

class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260 # <padding>

    beginning_of_sentence_characters = [258, 256, 259]+[260]*47 
    end_of_sentence_characters = [258, 257, 259]+[260]*47

    bos_token = '<S>'
    eos_token = '</S>'

    @staticmethod
    def convert_word_to_char_ids(word):
        if word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode('utf-8', 'ignore')[:(ELMoCharacterMapper.max_word_length-2)]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

class FeatureMaker(object):
    @staticmethod
    def co_occurence(query_tokens, doc_tokens):
        doc_len = len(doc_tokens)
        d_exists = [1 if w in query_tokens else 0 for w in doc_tokens]
        wdcnt = defaultdict(int)
        for t in doc_tokens:
            wdcnt[t] += 1
        q_exists = [wdcnt[w]/doc_len for w in query_tokens]
        assert len(d_exists)==len(doc_tokens)
        assert len(q_exists)==len(query_tokens)
        return np.array(q_exists), np.array(d_exists)
    @staticmethod
    def edit(query_tokens, doc_tokens):
        '''
        ni hao ma
        wo bu hao
        '''
        table = np.zeros((len(query_tokens)+1, len(doc_tokens)+1))
        table[0,:] = np.array(range(len(doc_tokens)+1))
        table[:,0] = np.array(range(len(query_tokens)+1))
        for i in range(1,len(query_tokens)+1): # every row
            for j in range(1,len(doc_tokens)+1): # every column
                left_up = table[i-1][j-1]+1 if query_tokens[i-1]!=doc_tokens[j-1] else table[i-1][j-1]
                table[i][j] = min([left_up, table[i-1][j]+1, table[i][j-1]+1])

        return table[len(query_tokens)][len(doc_tokens)]
    @staticmethod
    def jaccard_and_edit(query_tokens, doc_tokens):
        window_len = len(query_tokens)
        jaccard = np.zeros(len(doc_tokens))
        edit = np.zeros(len(doc_tokens))
        q_set = set(query_tokens)
        for i in range(len(doc_tokens)):
            low = max(0,i-window_len//2)
            high = i+window_len//2 + 1
            window_doc = doc_tokens[low:high]
            doc_set = set(window_doc)
            a = len(doc_set.intersection(q_set)); b = len(doc_set-q_set); c = len(q_set-doc_set)
            jaccard[i] = a/(a+b+c)
            edit[i] = FeatureMaker.edit(query_tokens, window_doc)
        return np.vstack((jaccard, edit))
                
    @staticmethod
    def extract_feature(query_tokens, doc_tokens):
        '''[Str]->[Str]->[Float]'''
        query_tokens = list(map(lambda x:x.lower() ,query_tokens))
        doc_tokens = list(map(lambda x:x.lower(), doc_tokens))
        q_exists, c_exists = FeatureMaker.co_occurence(query_tokens, doc_tokens)
        simi = FeatureMaker.jaccard_and_edit(query_tokens, doc_tokens)
        return q_exists, np.vstack((c_exists, simi))
             
def populate_dicts(files):
    vocab = defaultdict(count().__next__)
    # chars = defaultdict(count().__next__)
    chars = {}
    wdcnt = defaultdict(int)
    chcnt = defaultdict(int)
    test_wdcnt = defaultdict(int) # all glove words in test/dev should be added to known, but non-glove words in test/dev should be kept unknown

    # count the words and characters to find the ones with cardinality above the thresholds
    for f in files:
        with open('%s.tsv' % f, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                if 'test' in f:
                    uid, title, context, query = line.split('\t')
                else:
                    uid, title, context, query, answer, raw_context, begin_answer, end_answer, raw_answer = line.split('\t')
                tokens = context.split(' ')+query.split(' ')
                if 'train' in f:
                    for t in tokens:
                        wdcnt[t.lower()] += 1
                        for c in t: chcnt[c] += 1
                else:
                    for t in tokens:
                        test_wdcnt[t.lower()] += 1

    # add all words that are both in glove and the vocabulary first
    with open('glove.840B.300d.txt', encoding='utf-8') as f:
        for line in f:
            word = line.split()[0].lower()
            # polymath adds word to dict regardless of word_count_threshold when it's in GloVe
            if wdcnt[word] >= 1 or test_wdcnt[word] >= 1:
                _ = vocab[word]
    known =len(vocab)

    # add the special markers
    _ = vocab[unk]; unkid = vocab[unk]
    _ = vocab[pad]
    # compatible with elmo
    # _ = chars[unk]; unkcid = chars[unk]
    chars[pad] = 260; chars[START_TOKEN]=256; chars[END_TOKEN]=257
    chars[WORD_START]=258; chars[WORD_END]=259
    _ = vocab[START_TOKEN]
    _ = vocab[END_TOKEN]

    #finally add all words that are not in yet
    _  = [vocab[word] for word in wdcnt if word not in vocab and wdcnt[word] > word_count_threshold]
    for i in range(256):
        chars[chr(i)]=i
    #_  = [chars[c]    for c    in chcnt if c    not in chars and chcnt[c]    > char_count_threshold]
    # return as defaultdict(int) so that new keys will return id which is the value for <unknown>
    return known, dict(vocab), dict(chars)

def tsv_iter(line, vocab, chars, is_test=False, misc={}):
    unk_w = vocab[unk]
    # unk_c = chars[unk]

    if is_test:
        uid, title, context, query = line.split('\t')
        begin_answer=0
        end_answer=1
        answer="<S>"
        # change for dev.tsv
        '''
        split_line=line.strip().split('\t')
        if len(split_line)==9:
            uid, title, context, query, answer, raw_context, begin_answer, end_answer, raw_answer = split_line
        else:
            uid, title, context, query, answer, raw_context, begin_answer, end_answer = split_line
            answer='<S>'
        '''
    else:
        split_line=line.strip().split('\t')
        if len(split_line)==9:
            uid, title, context, query, answer, raw_context, begin_answer, end_answer, raw_answer = split_line
        else:
            uid, title, context, query, answer, raw_context, begin_answer, end_answer = split_line
            answer='<S>'
    print('[TSV ITER]{} {}'.format(uid, answer))
    ctokens = context.split(' ')
    qtokens = query.split(' ')

    #replace EMPTY_TOKEN with ''
    ctokens = [t if t != EMPTY_TOKEN else '' for t in ctokens]
    qtokens = [t if t != EMPTY_TOKEN else '' for t in qtokens]


    cwids = [vocab.get(t.lower(), unk_w) for t in ctokens]
    qwids = [vocab.get(t.lower(), unk_w) for t in qtokens]
    # discard unknown char
    ccids = [ELMoCharacterMapper.convert_word_to_char_ids(t) for t in ctokens] #clamp at word_size
    qcids = [ELMoCharacterMapper.convert_word_to_char_ids(t) for t in qtokens]


    ba, ea = int(begin_answer), int(end_answer) - 1 # the end from tsv is exclusive
    if ba > ea:
        ea=ba+1

    # if word is on begin/end position
    baidx = [0 if i != ba else 1 for i,t in enumerate(ctokens)]
    eaidx = [0 if i != ea else 1 for i,t in enumerate(ctokens)]

    atokens = answer.split(' ')

    # change for enable is_selected
    if not is_test and sum(eaidx) == 0:
        #raise ValueError('problem with input line:\n%s' % line)
        print('============================')
    if is_test and misc.keys():
        misc['uid'].append(uid)
        misc['answer']+=[answer]
        misc['rawctx']+=[context]
        misc['ctoken']+=[ctokens]

    qs,ds = FeatureMaker.extract_feature(qtokens, ctokens)

    return ctokens, qtokens, atokens, cwids, qwids, baidx, eaidx, ccids, qcids, qs, ds.T

def tsv_to_ctf(f, g, vocab, chars, is_test):
    print("Known words: %d" % known)
    print("Vocab size: %d" % len(vocab))
    print("Char size: %d" % len(chars))
    for lineno, line in enumerate(f):
        ctokens, qtokens, atokens, cwids, qwids,  baidx,   eaidx, ccids, qcids, qs, ds = tsv_iter(line, vocab, chars, is_test)
        count = 0
        for     ctoken,  qtoken,  atoken,  cwid,  qwid,   begin,   end,   ccid,  qcid, qf, df in zip_longest(
                ctokens, qtokens, atokens, cwids, qwids,  baidx,   eaidx, ccids, qcids, qs, ds):
            out = [str(lineno)]
            if ctoken is not None:
                out.append('|# %s' % pad_spec.format(ctoken.translate(sanitize)))
            if qtoken is not None:
                out.append('|# %s' % pad_spec.format(qtoken.translate(sanitize)))
            if atoken is not None:
                out.append('|# %s' % pad_spec.format(atoken.translate(sanitize)))
            if begin is not None:
                out.append('|ab %3d' % begin)
            if end is not None:
                out.append('|ae %3d' % end)
            if cwid is not None:
                if cwid >= known:
                    out.append('|cgw {}:{}'.format(0, 0))
                    out.append('|cnw {}:{}'.format(cwid - known, 1))
                else:
                    out.append('|cgw {}:{}'.format(cwid, 1))
                    out.append('|cnw {}:{}'.format(0, 0))
            if qwid is not None:
                if qwid >= known:
                    out.append('|qgw {}:{}'.format(0, 0))
                    out.append('|qnw {}:{}'.format(qwid - known, 1))
                else:
                    out.append('|qgw {}:{}'.format(qwid, 1))
                    out.append('|qnw {}:{}'.format(0, 0))
            if ccid is not None:
                outc = ' '.join(['%d' % c for c in ccid+[0]*max(word_size - len(ccid), 0)])
                out.append('|cc %s' % outc)
            if qcid is not None:
                outq = ' '.join(['%d' % c for c in qcid+[0]*max(word_size - len(qcid), 0)])
                out.append('|qc %s' % outq)
            if qf is not None:
                out.append('|qf %f' % qf)
            if df is not None:
                str_df = [str(x) for x in df]
                out.append('|df '+' '.join(str_df))
            
            g.write('\t'.join(out))
            g.write('\n')

if __name__=='__main__':
    try:
        known, vocab, chars = pickle.load(open('vocabs.pkl', 'rb'))
    except:
        known, vocab, chars = populate_dicts(tsvs)
        f = open('vocabs.pkl', 'wb')
        pickle.dump((known, vocab, chars), f)
        f.close()

    for tsv in tsvs:
        with open('%s.tsv' % tsv, 'r', encoding='utf-8') as f:
            with open('%s.ctf' % tsv, 'w', encoding='utf-8') as g:
                tsv_to_ctf(f, g, vocab, chars, tsv == 'test')
