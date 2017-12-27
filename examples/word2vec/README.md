# Word Embedding

This is an example of word embedding.
We implemented Mikolov's Skip-gram model and Continuous-BoW model with Hierarchical softmax and Negative sampling.

Run `train_word2vec.py` to train and get `word2vec.model` which includes embedding data.
You can find top-5 nearest embedding vectors using `search.py`.

This example is based on the following word embedding implementation in C++.
https://code.google.com/p/word2vec/

# Todo
- <eos>
- Iterator 分かりにくい。素直に、データセットを細切れに分けて、複数回スクリプトを叩くほうがよい。
    * デメリット
        - vocabを静的につくらないといけない。
        - データセットを適切な量毎に分ける作業。
        - lrateをなめらかにつなぐ作業。
- Iterator
- 以下は微妙。valを得るには一度舐めないといけないし、valを一度でも使うとexhaustされちゃう。
    * val = itertools.islice(wid_generator(), 0, min(int(num_tokens*0.05), 10000))

- 以下も微妙。sliceを取ることでiteratorになってしまう。iteratorになってしまうと、exhaustしてしまう。1epochしか回らない。
``` python
    train = itertools.islice(wids_generator(), min(int(num_tokens*0.05), 10000), sys.maxsize)
    val = itertools.islice(wids_generator(), 0, min(int(num_tokens*0.05), 10000))
    train_iter = WindowIteratorIterator(train, args.window, args.batchsize)
    val_iter = WindowIteratorIterator(val, args.window, args.batchsize, repeat=False)
```
