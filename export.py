"""Write the trained network out as binary blobs + constants for the assembler."""
import os, sys
import numpy as np
import model as Mo
from model import Net, HT
from data import MAXLEN

net = Net.load(sys.argv[1] if len(sys.argv) > 1 else "net.npz")
bplus = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # 1 = build for the B+ with W2 in paged RAM
os.makedirs("build", exist_ok=True)
def w(name, arr): open(f"build/{name}.bin", "wb").write(bytes((np.asarray(arr).astype(np.int64) & 255).ravel().tolist()))
# dense layers are stored one row per INPUT, so the 6502 can skip inputs that are zero
assert Mo.NOUT % 5 == 0
w("wq", Net.pack_nibbles(net.Wq)); w("wz", Net.pack_trits(net.Wz.T)); w("w1", Net.pack_trits(net.W1))
w("w2", Net.pack_trits(net.W2.T)); w("w3", Net.pack_trits(net.W3.T))
for n in ("bq", "bz", "b2", "b3"):
    b = getattr(net, n); assert np.abs(b).max() < 32768
    w(n + "lo", b & 255); w(n + "hi", (b >> 8) & 255)
for i in range(3): w(f"ht{i}", HT[i])
words = bytearray()
for wd in Mo.VOCAB[1 + len(Mo.PUNCT):]:
    words += wd[:-1].encode() + bytes([ord(wd[-1]) | 0x80])
open("build/words.bin", "wb").write(words)
c = dict(Q=net.Q, HQ=net.HQ, H1=net.H1, H2=net.H2, K=Mo.K, P=Mo.P, M=Mo.M, CR=Mo.CR, S2=Mo.S2, SQ=Mo.SQ, SZ=Mo.SZ, QC=Mo.QC,
         N1=Mo.N1, N2=Mo.N2, NOUT=Mo.NOUT, CTXW=Mo.CTXW, NV=Mo.NV, MAXTOK=Mo.MAXTOK, MAXLEN=MAXLEN,
         ROWBQ=net.HQ // 2, ROWB1=net.H1 // 5, AMAX=max(net.HQ, net.H1, net.H2, Mo.NOUT), BPLUS=bplus)
open("build/consts.asm", "w").write("".join(f"{k} = {v}\n" for k, v in c.items()))
open("boot.txt", "w").write("*RUN BEEBLM\n")
nw = sum(getattr(net, n).size for n in ("Wq", "Wz", "W1", "W2", "W3"))
print(c); print("weights:", nw, "; weight+bias bytes:", net.weight_bytes(), "; word list bytes:", len(words))
