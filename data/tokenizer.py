import re
from collections import Counter


PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

_TOKENIZE_PATTERN = re.compile(
    r"(\\[a-zA-Z]+)"   # LaTeX-команды: \frac, \text, \alpha, ...
    r"|(\\[^a-zA-Z])"  # escaped символы: \{, \\, \_, ...
    r"|(\s)"            # пробел — разделитель слов в русском тексте
    r"|([^\s])"         # любой одиночный символ: кириллица, латиница, цифры, скобки
)


class LaTeXTokenizer:

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.vocab_size: int = 0

        for idx, token in enumerate(SPECIAL_TOKENS):
            self.token2id[token] = idx
            self.id2token[idx] = token
        self.vocab_size = len(SPECIAL_TOKENS)

    @staticmethod
    def tokenize(formula: str) -> list[str]:
        tokens = []
        for match in _TOKENIZE_PATTERN.finditer(formula):
            token = match.group(1) or match.group(2) or match.group(3) or match.group(4)
            tokens.append(token)
        return tokens

    def build_vocab(self, formulas: list[str], min_freq: int = 2) -> None:
        counter: Counter[str] = Counter()
        for formula in formulas:
            counter.update(self.tokenize(formula))

        self.token2id = {}
        self.id2token = {}
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.token2id[token] = idx
            self.id2token[idx] = token

        idx = len(SPECIAL_TOKENS)
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.token2id:
                self.token2id[token] = idx
                self.id2token[idx] = token
                idx += 1

        self.vocab_size = len(self.token2id)

    def encode(self, formula: str, max_len: int = 512) -> list[int]:
        tokens = self.tokenize(formula)
        ids = [SOS_ID]
        for token in tokens[: max_len - 2]:
            ids.append(self.token2id.get(token, UNK_ID))
        ids.append(EOS_ID)
        ids += [PAD_ID] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        tokens = []
        for idx in ids:
            token = self.id2token.get(idx, UNK_TOKEN)
            if strip_special:
                if token == EOS_TOKEN:
                    break
                if token in (PAD_TOKEN, SOS_TOKEN):
                    continue
            tokens.append(token)
        return "".join(tokens)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"LaTeXTokenizer(vocab_size={self.vocab_size})"
