# BeebLM

A tiny neural language model that runs on a BBC Micro (Model B or B+, DFS) in
main memory, written in hand-assembled 6502. It is a chatbot called BEEB: you
type a line, it replies one word at a time.

## Run it

Two disc images:

- `beeblm-bplus.ssd` - BBC B+ only. Bigger model; the largest layer lives in the
  B+'s 12K of paged RAM at &8000-&AFFF. Files: BEEBLM (loader), W2, MAIN.
- `beeblm.ssd` - any Model B or B+ with DFS. Smaller model, main memory only.

SHIFT+BREAK to boot, or `*RUN BEEBLM`. ESCAPE leaves and restarts BASIC.

## How it works

Everything is integer arithmetic. `model.py` is the reference; `beeblm.asm`
matches it bit for bit.

1. The typed line is lower-cased and stripped of punctuation, then every
   character trigram is hashed (twice) into 256 buckets.
2. **Question encoder, once per line:** active buckets add rows of 4-bit weights
   into HQ units; a ternary layer expands that to H1 units.
3. **Generator, once per word:** ternary rows for the two previous words, the
   position and a "mood" bit are added in; then ReLU, a ternary H1 -> H2
   layer, and a ternary output layer whose two groups of outputs give the two
   digits of the next word's id. Word id 0 ends the reply.
4. Ternary weights (-1, 0, +1) are packed five to a byte, so the dense layers
   need only add and subtract: the 6502 has no multiply. Dense layers are
   stored one row per input, so inputs that are zero (most of them, after
   ReLU) are skipped entirely.

| build | HQ | H1 | H2 | weights | weight bytes |
|---|---|---|---|---|---|
| Model B | 40 | 160 | 220 | 75,680 | 19.1KB |
| B+ | 60 | 240 | 255 | 123,720 | 30.5KB |

## Rebuild

    python3 train.py --out net.npz                                   # Model B model
    python3 train.py --HQ 60 --H1 240 --H2 255 --steps 40000 --out net2.npz   # B+ model
    python3 train.py --HQ 60 --H1 240 --H2 255 --init net2.npz.pt --core 0.15 --steps 12000 --lr 0.002 --out net2.npz   # B+ fine-tune on short phrasings
    python3 export.py net.npz 0  &&  beebasm -i beeblm.asm -do beeblm.ssd -opt 3 -title BEEBLM
    python3 export.py net2.npz 1 &&  beebasm -i beeblm.asm -do beeblm-bplus.ssd -opt 3 -title BEEBLM

Checks:

    node run_beeb.mjs beeblm.ssd questions.txt <clock address> > out.jsonl   # jsbeeb, Model B build
    python3 verify.py net.npz out.jsonl
    python3 run_py65.py beeblm-bplus.ssd net2.npz questions.txt              # bare 6502 + B+ paging model

`data.py` holds the persona: intents, question templates and replies. Edit it
and retrain to change what BEEB knows.
