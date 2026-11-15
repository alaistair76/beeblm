"""Compare what the emulated BBC Micro printed with the Python integer reference."""
import json, sys
from model import Net
net = Net.load(sys.argv[1]); bad = n = 0; secs = []; toks = 0
for line in open(sys.argv[2]):
    d = json.loads(line)
    if "q" not in d: continue
    want = net.reply(d["q"], d["mood"]); n += 1; secs.append(d["secs"]); toks += len(net.reply_ids(d["q"], d["mood"])) + 1
    if want != d["reply"]:
        bad += 1; print("MISMATCH", d["q"], "| 6502:", d["reply"], "| ref:", want)
    elif "-v" in sys.argv: print(f'{d["secs"]:5.1f}s  > {d["q"]}\n        {d["reply"]}')
print(f"{n - bad}/{n} replies identical to the reference; mean {sum(secs)/max(1,n):.1f}s per reply, {sum(secs)/max(1,toks):.2f}s per word")
