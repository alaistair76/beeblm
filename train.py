"""Quantisation-aware training: the forward pass uses the same integers the 6502 will."""
import argparse, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import data as D
from model import Net, buckets, encode, K, P, M, CR, S2, SQ, SZ, QC, N1, N2, NOUT, CTXW, MAXTOK

ap = argparse.ArgumentParser()
ap.add_argument("--Q", type=int, default=256)
ap.add_argument("--H1", type=int, default=160)
ap.add_argument("--H2", type=int, default=220)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--bs", type=int, default=2048)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--warm", type=float, default=0.05)
ap.add_argument("--ramp", type=float, default=0.1)
ap.add_argument("--core", type=float, default=0.0, help="share of each batch drawn from the shortest phrasings of every intent")
ap.add_argument("--init", default="", help="continue from a saved .pt, fully quantised, stressing replies it still gets wrong")
ap.add_argument("--fp", default="")
ap.add_argument("--arch", type=int, default=2)
ap.add_argument("--HQ", type=int, default=40)
ap.add_argument("--noeval", action="store_true")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="net.npz")
ap.add_argument("--full", action="store_true", help="train on everything (no held-out split)")
args = ap.parse_args()
torch.manual_seed(args.seed); rng = random.Random(args.seed); torch.set_num_threads(args.threads)
Q, H1, H2, HQ = args.Q, args.H1, args.H2, args.HQ
R = Q + K * CTXW + P + M

# ---------------- data ----------------
replies = []                 # list of reply-variant lists
known, heldout = [], []      # (query, reply_set_index)
known_words = set()
for name, tpls, reps in D.INTENTS:
    qs = D.expand_intent(tpls); rng.shuffle(qs)
    nh = 0 if args.full else max(1, len(qs) // 12)
    ri = len(replies); replies.append(reps)
    pool = []
    for q in qs[nh:]:
        pool.append(q)
        w = D.norm(rng.choice(D.WRAP_PRE) + q + rng.choice(D.WRAP_POST)); pool.append(w)
        pool.append(D.norm(D.typo(rng, q)))
    known.append((pool, ri)); heldout += [(q, ri) for q in qs[:nh]]
    for q in qs: known_words.update(q.split())
FB = len(replies); replies.append(D.FALLBACKS)
D.UNK_WORDS = [w for w in D.UNK_WORDS if w not in known_words and w + "s" not in known_words]
all_known = {q for pool, _ in known for q in pool} | {q for q, _ in heldout}
unk = []
while len(unk) < 16000:
    q = D.norm(D.unknown_query(rng))
    if q and q not in all_known: unk.append(q)
unk_train, unk_test = unk[:15000], unk[15000:]
print(f"{len(D.INTENTS)} intents, {sum(len(p) for p,_ in known)} known train queries, {len(heldout)} held out, {len(D.UNK_WORDS)} unknown words")

def qvec(q):
    v = torch.zeros(Q); v[buckets(q, Q)] = 1.0; return v

# flatten queries into a tensor table
qlist, qreply, intent_slices = [], [], []
for pool, ri in known:
    intent_slices.append((len(qlist), len(pool))); qlist += pool; qreply += [ri] * len(pool)
CORE = torch.tensor([st + 3 * j for (st, ln), (pool, _) in zip(intent_slices, known) for j in sorted(range(ln // 3), key=lambda j: (len(pool[3 * j]), pool[3 * j]))[:6]])
unk_start = len(qlist); qlist += unk_train; qreply += [FB] * len(unk_train)
QF = torch.stack([qvec(q) for q in qlist]); QR = torch.tensor(qreply)

# reply tables: for (reply set, mood, pos) -> ctx tokens, target
nrep = len(replies)
CTX = torch.zeros((nrep, M, P, K), dtype=torch.long); TGT = torch.full((nrep, M, P), -1, dtype=torch.long); LEN = torch.zeros((nrep, M), dtype=torch.long)
for ri, reps in enumerate(replies):
    for m in range(M):
        toks = encode(reps[m % len(reps)]) + [0]
        LEN[ri, m] = len(toks)
        for p, t in enumerate(toks):
            hist = [0] * K + toks[:p]
            CTX[ri, m, p] = torch.tensor(hist[-K:]); TGT[ri, m, p] = t

def batch(bs):
    nk = int(bs * 0.8)
    it = torch.multinomial(INTENT_W, nk, replacement=True)
    st = torch.tensor([intent_slices[i][0] for i in it]); ln = torch.tensor([intent_slices[i][1] for i in it])
    variant = torch.multinomial(torch.tensor([0.5, 0.3, 0.2]), nk, replacement=True)    # clean / wrapped / typo
    qi = torch.cat([st + (torch.rand(nk) * (ln // 3)).long() * 3 + variant, unk_start + torch.randint(len(unk_train), (bs - nk,))])
    nc = int(bs * args.core)
    if nc: qi[:nc] = CORE[torch.randint(len(CORE), (nc,))]
    ri = QR[qi]; m = torch.randint(M, (bs,))
    p = (torch.rand(bs) * LEN[ri, m]).long()
    p = torch.where(torch.rand(bs) < 0.25, torch.zeros_like(p), p)   # stress the opening token
    ctx = CTX[ri, m, p]; tgt = TGT[ri, m, p]
    x = torch.zeros(bs, R); x[:, :Q] = QF[qi]
    ar = torch.arange(bs)
    for k in range(K):
        x[ar, Q + k * CTXW + ctx[:, k] // N2] = CRV; x[ar, Q + k * CTXW + N1 + ctx[:, k] % N2] = CRV
    x[ar, Q + K * CTXW + p] = CRV; x[ar, Q + K * CTXW + P + m] = CRV
    return x, tgt

# ---------------- model ----------------
LAM = 1.0   # 0 = float network, 1 = exact integer network

def tern(w, name=''):
    lam = 0.0 if (name and name in args.fp.split(',')) else LAM
    """Ternarise. The float stand-in is w/alpha so both versions have the same scale."""
    d = 0.7 * w.abs().mean().detach()
    mask = (w.abs() > d).float()
    q = torch.sign(w) * mask
    alpha = ((w.abs() * mask).sum() / mask.sum().clamp(min=1)).detach()
    wf = w / alpha
    return wf + lam * (q - wf).detach()

def rnd(b): return b + LAM * (torch.round(b) - b).detach()

CRV = CR
NC = R - Q            # context + position + mood rows

def q4(w):
    """4-bit weights -7..7; float stand-in on the same scale."""
    sc = (3.0 * w.std() / 7.0).detach()
    wf = w / sc
    q = torch.round(wf).clamp(-7, 7)
    return wf + LAM * (q - wf).detach()

def fl(z): return z + LAM * (torch.floor(z) - z).detach()

class QNet(nn.Module):
    def __init__(s):
        super().__init__()
        u = lambda *sh: nn.Parameter(torch.empty(*sh).uniform_(-1, 1))
        s.wq = u(Q, HQ); s.bq = nn.Parameter(torch.zeros(HQ))
        s.wz = u(H1, HQ); s.bz = nn.Parameter(torch.zeros(H1))
        s.w1 = u(NC, H1)
        s.w2 = u(H2, H1); s.b2 = nn.Parameter(torch.zeros(H2))
        s.w3 = u(NOUT, H2); s.b3 = nn.Parameter(torch.zeros(NOUT))
        s.g = nn.Parameter(torch.tensor(-4.0))
    def export(s):
        global LAM
        with torch.no_grad():
            LAM = 1.0; i = lambda t: t.round().int().numpy().copy()
            return Net(Q, Wq=i(q4(s.wq)), bq=i(s.bq), Wz=i(tern(s.wz)), bz=i(s.bz), W1=i(tern(s.w1)), W2=i(tern(s.w2)), b2=i(s.b2), W3=i(tern(s.w3)), b3=i(s.b3))
    def forward(s, x):
        xq, xc = x[:, :Q], x[:, Q:]
        aq = xq @ q4(s.wq) + rnd(s.bq)
        z = fl(aq / (1 << SQ)).clamp(0, 127)
        t = z @ tern(s.wz, 'wz').t() + rnd(s.bz)
        qa = fl(t / (1 << SZ)).clamp(-QC, QC)
        a1 = qa + xc @ tern(s.w1, 'w1')
        h1 = F.relu(a1)
        a2 = h1 @ tern(s.w2, 'w2').t() + rnd(s.b2)
        h2 = fl(a2 / (1 << S2)).clamp(0, 127)
        a3 = h2 @ tern(s.w3, 'w3').t() + rnd(s.b3)
        return a3 * torch.exp(s.g)

net = QNet()
INTENT_W = torch.ones(len(intent_slices))
if args.init:
    net.load_state_dict(torch.load(args.init)); args.warm, args.ramp = 0.0, 1e-9
    n0 = net.export(); r0 = random.Random(11)
    for i, (pool, ri) in enumerate(known):
        qs = r0.sample(pool[0::3], min(20, len(pool) // 3))
        INTENT_W[i] = 1.0 + 4.0 * (1.0 - sum(n0.reply(q, 0) == replies[ri][0] for q in qs) / len(qs))
    print('focus weights: mean', float(INTENT_W.mean()), flush=True)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
import math
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, t / 300) * (1.0 if t < 0.6 * args.steps else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t / args.steps - 0.6) / 0.4))))

def evaluate(n, tag):
    r = random.Random(7)
    def acc(items, nmax=400):
        items = r.sample(items, min(nmax, len(items))); ok = 0
        for q, ri in items:
            m = r.randrange(M)
            try: ok += n.reply(q, m) == replies[ri][m % len(replies[ri])]
            except AssertionError: pass
        return 100.0 * ok / max(1, len(items))
    tr = [(r.choice(p), ri) for p, ri in known for _ in range(3)]
    clean = [(r.choice(p[0::3]), ri) for p, ri in known for _ in range(3)]
    ho_typo = [(D.norm(D.typo(r, q)), ri) for q, ri in heldout]
    core = [(qlist[i], int(QR[i])) for i in CORE.tolist()]
    res = dict(core=acc(core, 2000), clean=acc(clean), train=acc(tr), heldout=acc(heldout), heldout_typo=acc(ho_typo), unknown=acc([(q, FB) for q in unk_test]))
    print(tag, " ".join(f"{k}={v:.1f}%" for k, v in res.items()), flush=True)
    return res

t0 = time.time()
for step in range(1, args.steps + 1):
    f = step / args.steps
    LAM = 0.0 if f < args.warm else min(1.0, (f - args.warm) / args.ramp)
    x, y = batch(args.bs)
    out = net(x)
    loss = F.cross_entropy(out[:, :N1], y // N2) + F.cross_entropy(out[:, N1:], y % N2)
    hit = (out[:, :N1].argmax(1) * N2 + out[:, N1:].argmax(1)) == y
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pass
    if step % 500 == 0:
        p0 = x[:, Q + K * CTXW] > 0
        print(f"step {step} lam {LAM:.2f} loss {loss.item():.4f} acc {hit.float().mean().item():.3f} first-token acc {hit[p0].float().mean().item():.3f} {time.time()-t0:.0f}s", flush=True)
torch.save(net.state_dict(), args.out + '.pt')
n = net.export(); n.save(args.out); print('saved', args.out, 'weight bytes', n.weight_bytes())
evaluate(n, 'eval')
