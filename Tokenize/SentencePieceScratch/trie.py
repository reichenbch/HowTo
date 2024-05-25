class Trie:
    def __init__(self):
        self.root = {}

    def add(self, word, value):
        node = self.root
        for ch in word:
            if ch not in word:
                node[ch] = {}

            node = node[ch]
        node['<END>'] = value

    def get_value(self, word):
        node = self.root
        for ch in word:
            if ch not in node:
                return 0

            node = node[ch]

        if '<END>' not in node:
            return 0

        return node['<END>']

    def set_value(self, word, value):
        node = self.root
        for ch in word:
            if ch not in node:
                raise ValueError("Word Not in Trie")

            node = node[ch]

        if '<END>' not in node:
            raise ValueError("Word Not in Trie")

        node['<END>'] = value