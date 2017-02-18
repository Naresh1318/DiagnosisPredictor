Word2Vec in Gensim:

Modified two files in the word2vec of Gensim:

1 word2vec.py

(1)In the class of "class Word2Vec(utils.SaveLoad):", I added two parameters "array_pre" and "array_suf" to store the "window" sizes.

For event i:
array_pre[i]: Prefix window size for event i;
array_suf[i]: Suffix window size for event i.

(2)Input:
time_win_pre.csv: Prefix window size
time_win_suf.csv: Suffix window size

(3)Code
#prepare the window parameters
self.array_pre=[]
self.array_suf=[]
win = pd.read_csv('time_win_pre.csv', sep=',',header=None)
for i in range(len(win)):
    results = map(int, win.ix[i,0].split())
    self.array_pre.append(results)

win = pd.read_csv('time_win_suf.csv', sep=',',header=None)
for i in range(len(win)):
    results = map(int, win.ix[i,0].split())
    self.array_suf.append(results)


2 word2vec_inner.pyx

Modified the "train_batch_cbow" function:

(1)Initialize two parameters.

#Define the window_pre and window_suf parameters based on "array_pre" and "array_suf".
cdef int[:,:] window_pre = model.array_pre
cdef int[:,:] window_suf = model.array_suf

(2) Given different window_pre and window_suf values for each event.

#Set i and j parameters for each event. i is the window size before event, and j is the window size after event.

# release GIL & train on all sentences
with nogil:
for sent_idx in range(effective_sentences):
    idx_start = sentence_idx[sent_idx]
    idx_end = sentence_idx[sent_idx + 1]
    for i in range(idx_start, idx_end):
        j = i - window_pre[sent_idx][i]
        if j < idx_start:
            j = idx_start
        k = i + window_suf[sent_idx][i]
        if k > idx_end:
            k = idx_end
        if hs:
            fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k, cbow_mean, word_locks)
        if negative:
            next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random, word_locks)
        
return effective_words