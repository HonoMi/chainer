#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
"""
import argparse
import collections

import numpy as np
import sys
import re
import copy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
import itertools
from chainer import reporter
from chainer import training
from chainer.training import extensions
from word2vec_module import word2vec_module


class ContinuousBoW(chainer.Chain):
    """Definition of Continuous Bag of Words Model"""

    def __init__(self, n_vocab, n_units, loss_func):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.loss_func = loss_func

    def __call__(self, x, contexts):
        e = self.embed(contexts)
        h = F.sum(e, axis=1) * (1. / contexts.shape[1])
        loss = self.loss_func(h, x)
        reporter.report({'loss': loss}, self)
        return loss


class SkipGram(chainer.Chain):
    """Definition of Skip-gram Model"""

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.loss_func = loss_func

    def __call__(self, x, contexts):
        e = self.embed(contexts)
        batch_size, n_context, n_units = e.shape
        x = F.broadcast_to(x[:, None], (batch_size, n_context))
        e = F.reshape(e, (batch_size * n_context, n_units))
        x = F.reshape(x, (batch_size * n_context,))
        loss = self.loss_func(e, x)
        reporter.report({'loss': loss}, self)
        return loss


class SoftmaxCrossEntropyLoss(chainer.Chain):
    """Softmax cross entropy loss function preceded by linear transformation.

    """

    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        with self.init_scope():
            self.out = L.Linear(n_in, n_out, initialW=0)

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


# class WindowIteratorIterator(chainer.dataset.Iterator):
# 
#     def __init__(self, dataset_generator, window, batch_size, repeat=True):
#         self.dataset_generator = dataset_generator
#         self.window = window  # size of context window
#         self.batch_size = batch_size
#         self._repeat = repeat
#         self.epoch = 0
#         self.is_new_epoch = False
#         self.current_position = 0
#         if not self._repeat and self.epoch > 0:
#             raise StopIteration
# 
#     def __next__(self):
#         """This iterator returns a list representing a mini-batch.
# 
#         Each item indicates a different position in the original sequence.
#         """
#         if not self._repeat and self.epoch > 0:
#             raise StopIteration
#         for dataset in self.dataset_generator:
#             window_iterator = WindowIterator(dataset, self.window, self.batch_size, repeat=False)
#             for center, contexts in window_iterator:
#                 print(center.shape, contexts.shape)
#                 print('hogeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
#                 yield center, contexts
#         self.epoch += 1
#         self.is_new_epoch = True
# 
#     @property
#     def epoch_detail(self):
#         # return self.epoch + float(self.current_position) / len(self.order)
#         return self.epoch


class WindowIteratorIterator(chainer.dataset.Iterator):

    def __init__(self, wid_generator, window, batch_size, repeat=True):
        self.wid_generator = wid_generator
        self.window = window  # size of context window
        self.batch_size = batch_size
        self._repeat = repeat
        self.epoch = 0
        self.is_new_epoch = False
        self.current_position = 0
        self._my_iterator = self.__my_generator()
        if not self._repeat and self.epoch > 0:
            raise StopIteration

    def __next__(self):
        """This iterator returns a list representing a mini-batch.

        Each item indicates a different position in the original sequence.
        """
        if not self._repeat and self.epoch > 0:
            raise StopIteration
        try:
            return self._my_iterator.__next__()
        except Exception as e:
            self._my_iterator = self.__my_generator()
        self.epoch += 1
        self.is_new_epoch = True
        return self._my_iterator.__next__()

    def __my_generator(self):
        def batch():
            i = 0
            batch = []
            batch_pre = []
            for i, wid in enumerate(self.wid_generator()):
                batch.append(wid)
                if i % self.batch_size + 1 == 0:
                    batch_pre = copy.deepcopy(batch)
                    batch = []
                    yield batch_pre
        for wid_batch in batch():
            window_iterator = WindowIterator(wid_batch, self.window, self.batch_size, repeat=False)()
            for center, contexts in window_iterator:
                yield center, contexts

    @property
    def epoch_detail(self):
        # return self.epoch + float(self.current_position) / len(self.order)
        return self.epoch



class WindowIterator(chainer.dataset.Iterator):
    """Dataet iterator to create a batch of sequences at different positions.

    This iterator retuns a pair of the current words and the context words.
    """

    def __init__(self, dataset, window, batch_size, repeat=True):
        self.dataset = np.array(dataset, np.int32)
        self.window = window  # size of context window
        self.batch_size = batch_size
        self._repeat = repeat
        # order is the array which is shuffled ``[window, window + 1, ...,
        # len(dataset) - window - 1]``
        self.order = np.random.permutation(
            len(dataset) - window * 2).astype(np.int32)
        self.order += window
        self.current_position = 0
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False

    def __next__(self):
        """This iterator returns a list representing a mini-batch.

        Each item indicates a different position in the original sequence.
        """
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]
        contexts = self.dataset.take(pos)
        center = self.dataset.take(position)

        if i_end >= len(self.order):
            np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return center, contexts

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


def convert(batch, device):
    print('== convert')
    center, contexts = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        contexts = cuda.to_gpu(contexts)
    return center, contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=100, type=int,
                        help='number of units')
    parser.add_argument('--window', '-w', default=5, type=int,
                        help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                        default='skipgram',
                        help='model type ("skipgram", "cbow")')
    parser.add_argument('--negative-size', default=5, type=int,
                        help='number of negative samples')
    parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'],
                        default='hsm',
                        help='output model type ("hsm": hierarchical softmax, '
                        '"ns": negative sampling, "original": '
                        'no approximation)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--wakati_corpus_list')
    parser.add_argument('--num_tokens', type=int, default=None, help='If not set, we count words as the 1st-pash.')
    parser.add_argument('--word_count_threshold', default=5, type=int)
    parser.set_defaults(test=False)
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Training model: {}'.format(args.model))
    print('Output type: {}'.format(args.out_type))
    print('')

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    wakati_corpus_list = [line.rstrip() for line in open(args.wakati_corpus_list, 'r').readlines() if not re.match('^\s*#', line)]

    # Create vocab.
    vocab = word2vec_module.create_vocab(wakati_corpus_list, count_threshold=args.word_count_threshold)
    index2word = dict([(wid, word) for (word, wid) in vocab.items()])

    # Load the dataset
    words_generator = word2vec_module.WordsGenerator(wakati_corpus_list, batch_size=1000)

    class WidsGenerator:
    
        def __init__(self, words_generator, vocab):
            self.words_generator = words_generator
            self.vocab = vocab

        def __call__(self):
            for words in self.words_generator():
                wids = [vocab[word] if word in vocab else 0 for word in words]
                yield wids

    class WidGenerator:
 
        def __init__(self, wids_generator):
            self.wids_generator = wids_generator

        def __call__(self):
            for wids in self.wids_generator():
                for wid in wids:
                    yield wid

    wids_generator = WidsGenerator(words_generator, vocab)   # Generator call returns iterator object.
    wid_generator = WidGenerator(wids_generator)
    # train, val, _ = chainer.datasets.get_ptb_words()
    num_tokens = len([wid for wid in wid_generator()]) if args.num_tokens is None else args.num_tokens
    train = itertools.islice(wid_generator(), min(int(num_tokens*0.05), 10000), sys.maxsize)
    val = itertools.islice(wid_generator(), 0, min(int(num_tokens*0.05), 10000))
    counts = collections.Counter(wid_generator())
    # counts.update(collections.Counter(WidGenerator(val)()))
    # n_vocab = max(train) + 1
    n_vocab = len(vocab)

    # if args.test:
    #     train = train[:100]
    #     val = val[:100]

    print('n_vocab: %d' % n_vocab)
    # print('data length: %d' % len(train))

    if args.out_type == 'hsm':
        HSM = L.BinaryHierarchicalSoftmax
        tree = HSM.create_huffman_tree(counts)
        loss_func = HSM(args.unit, tree)
        loss_func.W.data[...] = 0
    elif args.out_type == 'ns':
        cs = [counts[w] for w in range(len(counts))]
        loss_func = L.NegativeSampling(args.unit, cs, args.negative_size)
        loss_func.W.data[...] = 0
    elif args.out_type == 'original':
        loss_func = SoftmaxCrossEntropyLoss(args.unit, n_vocab)
    else:
        raise Exception('Unknown output type: {}'.format(args.out_type))

    # Choose the model
    if args.model == 'skipgram':
        model = SkipGram(n_vocab, args.unit, loss_func)
    elif args.model == 'cbow':
        model = ContinuousBoW(n_vocab, args.unit, loss_func)
    else:
        raise Exception('Unknown model type: {}'.format(args.model))

    if args.gpu >= 0:
        model.to_gpu()

    # Set up an optimizer
    optimizer = O.Adam()
    optimizer.setup(model)

    # Set up an iterator
    train = itertools.islice(wids_generator(), min(int(num_tokens*0.05), 10000), sys.maxsize)
    val = itertools.islice(wids_generator(), 0, min(int(num_tokens*0.05), 10000))
    train_iter = WindowIteratorIterator(train, args.window, args.batchsize)
    val_iter = WindowIteratorIterator(val, args.window, args.batchsize, repeat=False)
    # train_iter = WindowIterator(train, args.window, args.batchsize)
    # val_iter = WindowIterator(val, args.window, args.batchsize, repeat=False)

    # Set up an updater
    updater = training.updater.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)

    # Set up a trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(
        val_iter, model, converter=convert, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    # Save the word2vec model
    with open('word2vec.model', 'w') as f:
        f.write('%d %d\n' % (len(index2word), args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))


if __name__ == '__main__':
    main()
