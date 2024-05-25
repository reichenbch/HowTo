import os
import re
import collections
import numpy as np
from trie import Trie
from scipy.special import digamma


class SentencePieceTrainer:
    def __init__(self):
        self.trie = None
        self.maxlen = None
        self.vocab = None

    def _initialize_trie(self, tokens):
        """

        :param tokens:
        :return:
        """
        trie = Trie()
        norm = sum(list(tokens.values()))
        logsum = digamma(norm)

        maxlen = 0
        for tok, val in tokens.items():
            trie.add(tok, digamma(val)-logsum)
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

    def forward_step(self, text, trie):
        """

        :param text:
        :param trie:
        :return:
        """
        N = len(text)

        # d[i] contains the maximum log_prob of any tokenization of text[:i], initialized to 0.
        # log(0) = -infinity

        d = [-np.inf] * (N+1)

        # p[i] (stands for parent) contains the number of characters of the final token in the most likely
        # sequence that ends at index i.
        p = [None] * (N+1)

        d[0] = 0

        for i in range(N+1):
            # find all possible final words. Have to look back a distance set by the length of the longest token.

            for j in range(max(i-self.maxlen, 0), i):
                final_token = text[j:i]
                final_value = trie.get_value(final_token)

                if final_value and d[j] + final_value > d[i]:
                    d[i] = d[j] + final_value
                    p[i] = len(final_token)

            if p[i] is None:
                raise ValueError(f"Encountered unknown token'{text[i-1]}'.")

        loss = d[-1]

        return loss, p

    def backward_step(self, text, p):
        """

        :param text:
        :param p:
        :return:
        """
        idx = len(p)
        tokenization = []

        while idx > 1:
            # move back the number of steps the p tells us to.
            next_idx = idx - p[idx - 1]
            tok = text[next_idx-1:idx-1]

            tokenization.append(tok)

            idx = next_idx

        tokenization = list(reversed(tokenization))

        return tokenization

    def E_step(self, tokenization, trie):
        """

        :param tokenization:
        :param trie:
        :return:
        """
        # get new token count based on updated tokenization

        counts = collections.Counter(tokenization)
        norm = sum(list(counts.values()))

        # Bayesianify them: https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        # https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc
        # we are returning the log probabilties here (alpha=0 prior)
        logsum = digamma(norm)

        for k,v in counts.items():
            counts[k] = digamma(v) - logsum

        for k, v in counts.items():
            trie.set_value(k, v)

        return trie

    def M_step(self, text, trie):
        """

        :param text:
        :param trie:
        :return:
        """
        loss, p = self.forward_step(text, trie)
        tokenization = self.backward_step(text, p)

        return tokenization, loss

    def EM_step(self, text, tokenization, trie):
        """

        :param text:
        :param tokenization:
        :param trie:
        :return:
        """
        trie = self.E_step(tokenization, trie)
        tokenization, loss = self.M_step(text, tokenization)

        return loss, tokenization, trie

    def EM_round(self, text, tokens, delta=0.01, max_iter=10):
        """

        :param text:
        :param tokens:
        :param delta:
        :param max_iter:
        :return:
        """
        tokenization, old_loss = self.M_step(text, self.trie)

        for step in range(max_iter):
            print(f"EM iter {step}: ", end='')
            loss, tokenization, trie = self.EM_step(text, tokenization, self.trie)
            print(f"Loss={loss:.2f}")

            if abs(old_loss - loss) < delta:
                break

            old_loss = loss

    def prune_tokens(self, tokens, characters, vocab_size, trim_factor=0.2):
        """
            Tokens are passed by reference and modified in place.

        :param tokens:
        :param characters:
        :param vocab_size:
        :param trim_factor:
        :return:
            True: to indicate to caller that more rounds are needed
            False: to indicate we successfully hit the target vocab size
            ValueError: if the vocab size cannot be reached.
        """
        sorted_tokens = tokens.most_common()

        N = len(sorted_tokens)
        n_trim = int(trim_factor * N)

        for i in reversed(range(N)):
            if N <= vocab_size:
                return False

            if n_trim <= 0:
                return True

            tok = sorted_tokens[i][0]

            if tok not in characters:
                self.trie.set_value(tok, 0) # we need to delete it from the trie (that sticks around)
                tokens.pop(tok) # also need to delete from tokens, so the next round doesn't see it
                n_trim -= 1
                N -= 1

        if n_trim > 0:
            raise ValueError('Could not reduce tokens further. Please increase vocab size')

        return False

    def fit(self, text, tokens, characters, vocab_size, delta=0.01, max_iter=5, max_rounds=5):
        """
            To turn off pruning just set max_rounds = 1

        :param text:
        :param tokens:
        :param characters:
        :param vocab_size:
        :param delta:
        :param max_iter:
        :param max_rounds:
        :return:
        """
        text = re.sub(' ', '_', text)

        if vocab_size > len(tokens):
            raise ValueError(f"Vocab size is larger than the available number of tokens - {len(tokens)}.")

        self.trie, self.maxlen = self._initialize_trie(tokens)

        for i in range(1, max_rounds + 1):
            print(f"--- Round {i}. Vocab size: {len(tokens)} ---")
            self.EM_round(text, tokens, delta, max_iter)

            if not self.prune_tokens(tokens, characters, vocab_size):
                break

        self.vocab_size = len(tokens)

    def generalized_forward_step(self, text, trie, nbest_size=1):
        """

        :param text:
        :param trie:
        :param nbest_size:
        :return:
        """
        N = len(text)
        d = [-np.inf] * (N + 1)
        p = [None] * (N + 1)

        d[0] = 0

        for i in range(1, N+1):
            d_queue = []
            p_queue = []

            for j in range(max(i - self.maxlen, 0), i):
                final_token = text[j:i]
                final_value = trie.get(final_token)

                if final_value:
                    curr_d = d[j] + final_value
                    curr_p = len(final_token)

                    d[i] = max(d[i], curr_d)

                    d_queue.append(curr_d)
                    p_queue.append(curr_p)

            ids = np.argsort(d_queue)[-nbest_size:]
            p[i] = [p_queue[id] for id in ids]

        return p

    def generalized_backward_step(self, text, p):
        """

        :param text:
        :param p:
        :return:
        """
        idx = len(p)
        tokenization = []

        while idx > 1:
            back_steps = np.random.choice(p[idx-1])
            next_idx = idx - back_steps

            tok = text[next_idx-1: idx -1]
            tokenization.append(tok)

            idx = next_idx
        tokenization = list(reversed(tokenization))
        return tokenization

    def tokenize(self, text, nbest_size=1):
        """

        :param text:
        :param nbest_size:
        :return:
        """

        text = re.sub(' ', '_', text)
        if self.trie is None:
            raise ValueError("Trainer has not yet been fit. Cannot tokenize.")

        p = self.generalized_forward_step(text, self.trie, nbest_size)
        tokenization = self.generalized_backward_step(text, p)

        return tokenization