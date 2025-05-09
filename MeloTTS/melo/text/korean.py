# melo/text/korean.py

import re
import unicodedata
import types

import MeCab
from transformers import AutoTokenizer

from . import punctuation, symbols
from num2words import num2words
from melo.text.ko_dictionary import english_dictionary, etc_dictionary
from anyascii import anyascii
from jamo import hangul_to_jamo

# -----------------------------------------------------------------------------
# Initialize G2P for Korean, monkey-patching MeCab.Tagger to provide `.pos()`
# -----------------------------------------------------------------------------
from g2pkk import G2p

# 1) Create G2p instance
g2p_kr = G2p()

# 2) Create a MeCab Tagger using the default Korean dictionary (mecab-ko-dic)
_mecab_tagger = MeCab.Tagger()

# 3) Define a `.pos()` method wrapping parseToNode()
def _pos(self, text):
    node = self.parseToNode(text)
    tokens = []
    while node:
        surface = node.surface
        if surface:
            # feature format: "POS,*,*,*,*,*,*,*,*"
            pos_tag = node.feature.split(',')[0]
            tokens.append((surface, pos_tag))
        node = node.next
    return tokens

# 4) Monkey-patch the Tagger
_mecab_tagger.pos = types.MethodType(_pos, _mecab_tagger)

# 5) Override G2p's internal mecab with our patched tagger
g2p_kr.mecab = _mecab_tagger

# -----------------------------------------------------------------------------
# Normalization routines
# -----------------------------------------------------------------------------
def normalize(text: str) -> str:
    text = text.strip()
    # remove obscure unicode blocks
    text = re.sub(r"[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]", "", text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    return text.lower()


def normalize_with_dictionary(text: str, dic: dict[str, str]) -> str:
    if any(k in text for k in dic):
        pattern = re.compile("|".join(re.escape(k) for k in dic))
        return pattern.sub(lambda m: dic[m.group()], text)
    return text


def normalize_english(text: str) -> str:
    def repl(m):
        w = m.group()
        return english_dictionary.get(w, w)
    return re.sub(r"([A-Za-z]+)", repl, text)


# -----------------------------------------------------------------------------
# Korean text → Jamo phonemes
# -----------------------------------------------------------------------------
def korean_text_to_phonemes(text: str, character: str = "hangeul") -> str:
    """
    Convert Korean text into a sequence of Hangul Jamo characters.
    If character=="english", do English G2P + ascii transliteration.
    """
    text = normalize(text)

    if character == "english":
        phones = g2p_kr(text)
        return anyascii(phones)

    # Hangul branch
    phones = g2p_kr(text)               # e.g. "안녕" → "안녕" in phoneme string
    jamo_seq = list(hangul_to_jamo(phones))
    return "".join(jamo_seq)


# -----------------------------------------------------------------------------
# Full-text normalization entrypoint
# -----------------------------------------------------------------------------
def text_normalize(text: str) -> str:
    return normalize(text)


# -----------------------------------------------------------------------------
# Distribute N phones evenly across M word-pieces
# -----------------------------------------------------------------------------
def distribute_phone(n_phone: int, n_word: int) -> list[int]:
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        idx = phones_per_word.index(min(phones_per_word))
        phones_per_word[idx] += 1
    return phones_per_word


# -----------------------------------------------------------------------------
# Tokenizer & main G2P pipeline (word2ph mapping)
# -----------------------------------------------------------------------------
model_id = "kykim/bert-kor-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def g2p(norm_text: str):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    word2ph = []
    ph_groups = []

    # group subword tokens into words
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))

    for group in ph_groups:
        word = "".join(group)
        if word == "[UNK]":
            phs.append("_")
            word2ph.append(1)
            continue
        if word in punctuation:
            phs.append(word)
            word2ph.append(1)
            continue

        # get phonemes (Jamo) for each group
        phonemes = korean_text_to_phonemes(word)
        phone_len = len(phonemes)
        word_len = len(group)
        counts = distribute_phone(phone_len, word_len)

        phs.extend(phonemes)
        word2ph.extend(counts)

    # add boundary symbols
    phones = ["_"] + phs + ["_"]
    tones = [0] * len(phones)
    word2ph = [1] + word2ph + [1]

    assert len(word2ph) == len(tokenized) + 2
    return phones, tones, word2ph


# -----------------------------------------------------------------------------
# BERT feature extraction
# -----------------------------------------------------------------------------
def get_bert_feature(text: str, word2ph: list[int], device: str = "cuda"):
    from . import japanese_bert
    return japanese_bert.get_bert_feature(
        text, word2ph, device=device, model_id=model_id
    )


# -----------------------------------------------------------------------------
# Debug / standalone test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample = "안녕하세요! 오늘은 날씨가 정말 좋네요."
    phones, tones, word2ph = g2p(text_normalize(sample))
    print("Phones:", phones)
    print("Word2Ph:", word2ph)
    feat = get_bert_feature(sample, word2ph, device="cpu")
    print("BERT feature shape:", feat.shape)
