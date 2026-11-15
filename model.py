"""Shared model definition: tokeniser, feature hashing, integer reference inference, packing.

The integer inference here is the specification the 6502 code must match bit for bit.
"""
import collections, random, re
import numpy as np
import data as D
from data import MAXLEN, norm

K = 2                 # previous output tokens fed back in
M = 2                 # "mood" inputs (pick reply variant)
S2 = 2                # right shift after layer 2
N2 = 20               # a token id is two "digits": d1 = id // N2, d2 = id % N2
IN_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789"   # code = index
PUNCT = ".,?!"


def tokenise(reply):
    return re.findall(r"[A-Z0-9'\-]+|[.,?!]", reply)


def build_vocab():
    c = collections.Counter()
    reps = [r for _, _, rs in D.INTENTS for r in rs] + D.FALLBACKS
    for r in reps:
        c.update(t for t in tokenise(r) if t not in PUNCT)
    words = [w for w, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
    vocab = ["<end>"] + list(PUNCT) + words          # id 0 = end, 1..4 punctuation
    maxtok = max(len(tokenise(r)) for r in reps)
    return vocab, maxtok


VOCAB, MAXTOK = build_vocab()
TOK = {w: i for i, w in enumerate(VOCAB)}
NV = len(VOCAB)
N1 = -(-NV // N2)
NOUT = N1 + N2
P = MAXTOK + 1        # position one-hots
CTXW = N1 + N2        # rows per context slot


def encode(reply):
    return [TOK[t] for t in tokenise(reply)]


def detok(ids):
    s = ""
    for n, i in enumerate(ids):
        if i <= len(PUNCT):
            s += VOCAB[i]
        else:
            s += (" " if n else "") + VOCAB[i]
    return s


def hash_tables(seed=1985):
    rng = random.Random(seed)
    return [[rng.randrange(256) for _ in IN_CHARS] for _ in range(3)]


HT = hash_tables()


def buckets(q, Q):
    """Set of active hash buckets for a normalised query."""
    s = " " + q + " "
    c = [IN_CHARS.index(ch) for ch in s]
    out = set()
    for i in range(len(c) - 2):       # two hashes per trigram (same tables, roles rotated)
        out.add((HT[0][c[i]] ^ HT[1][c[i + 1]] ^ HT[2][c[i + 2]]) & (Q - 1))
        out.add((HT[1][c[i]] ^ HT[2][c[i + 1]] ^ HT[0][c[i + 2]]) & (Q - 1))
    return sorted(out)


SQ, SZ, QC = 1, 3, 64  # shifts / clamp in the question encoder
CR = 8                 # weight of context, position and mood rows


class Net:
    """Integer-only network.

    Question encoder (run once):  hashed trigrams -> HQ units (4-bit weights) -> H1 (ternary)
    Generator (run per token):    + context/position rows (ternary) -> H2 (ternary) -> two digit groups
    """
    NAMES = ["Wq", "bq", "Wz", "bz", "W1", "W2", "b2", "W3", "b3"]

    def __init__(self, Q, **a):
        self.Q = Q
        for n in self.NAMES: setattr(self, n, np.asarray(a[n], dtype=np.int32))
        self.HQ, self.H1, self.H2 = self.Wq.shape[1], self.W1.shape[1], self.W2.shape[0]
        self.NC = K * CTXW + P + M
        assert self.Wq.shape[0] == Q and self.W1.shape[0] == self.NC and self.W3.shape[0] == NOUT
        assert np.abs(self.Wq).max() <= 7
        assert all(np.abs(getattr(self, n)).max() <= 1 for n in ('Wz', 'W1', 'W2', 'W3')), 'ternary layers must be -1/0/+1'

    def rows_ctx(self, k, tok): return [k * CTXW + tok // N2, k * CTXW + N1 + tok % N2]
    def row_pos(self, p): return K * CTXW + p
    def row_mood(self, m): return K * CTXW + P + m

    def encode_query(self, text, mood=0):
        aq = self.bq.copy()
        for b in buckets(norm(text), self.Q):
            aq += self.Wq[b]
        z = np.clip(aq >> SQ, 0, 127)
        t = self.Wz @ z + self.bz
        assert np.abs(t).max() < 32768
        return np.clip(t >> SZ, -QC, QC) + CR * self.W1[self.row_mood(mood)]

    def reply_ids(self, text, mood=0, trace=None):
        qacc = self.encode_query(text, mood)
        ctx = [0] * K
        out = []
        for pos in range(MAXTOK):
            a1 = qacc.copy()
            for k in range(K):
                for r in self.rows_ctx(k, ctx[k]):
                    a1 += CR * self.W1[r]
            a1 += CR * self.W1[self.row_pos(pos)]
            assert np.abs(a1).max() <= 127, "layer-1 overflow"
            h1 = np.maximum(a1, 0)
            a2 = self.W2 @ h1 + self.b2
            assert np.abs(a2).max() < 32768
            h2 = np.clip(a2 >> S2, 0, 127)
            a3 = self.W3 @ h2 + self.b3
            assert np.abs(a3).max() < 32768
            tok = int(np.argmax(a3[:N1])) * N2 + int(np.argmax(a3[N1:]))   # first maximum wins
            if trace is not None: trace.append(tok)
            if tok == 0 or tok >= NV:
                break
            out.append(tok)
            ctx = ctx[1:] + [tok]
        return out

    def reply(self, text, mood=0):
        return detok(self.reply_ids(text, mood))

    @staticmethod
    def pack_trits(W):
        """5 ternary weights per byte: digit 0 -> 0, 1 -> +1, 2 -> -1."""
        W = np.asarray(W); n = W.shape[1]
        assert n % 5 == 0
        t = np.where(W < 0, 2, W).astype(np.int32).reshape(W.shape[0], n // 5, 5)
        return (t * np.array([1, 3, 9, 27, 81])).sum(-1).astype(np.uint8)

    @staticmethod
    def pack_nibbles(W):
        """2 weights (-7..7) per byte, each stored as value+8."""
        W = np.asarray(W) + 8
        assert W.shape[1] % 2 == 0
        return (W[:, 0::2] | (W[:, 1::2] << 4)).astype(np.uint8)

    def save(self, path):
        np.savez(path, Q=self.Q, **{n: getattr(self, n) for n in self.NAMES})

    @staticmethod
    def load(path):
        d = np.load(path)
        return Net(int(d["Q"]), **{n: d[n] for n in Net.NAMES})

    def weight_bytes(self):
        return (self.Q * self.HQ // 2 + self.H1 * self.HQ // 5 + self.NC * self.H1 // 5 + self.H2 * self.H1 // 5
                + NOUT * self.H2 // 5 + 2 * self.HQ + 2 * self.H1 + 2 * self.H2 + 2 * NOUT)


if __name__ == "__main__":
    reps = [r for _, _, rs in D.INTENTS for r in rs] + D.FALLBACKS
    assert all(detok(encode(r)) == r for r in reps), [r for r in reps if detok(encode(r)) != r]
    print("vocab", NV, "N1", N1, "max tokens", MAXTOK, "vocab text bytes", sum(len(w) for w in VOCAB[1:]))
