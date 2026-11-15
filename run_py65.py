"""Run the B+ disc image on a bare 6502 (py65) with the B+'s paged RAM and stub OS calls,
and compare every reply with the Python reference. usage: run_py65.py disc.ssd net.npz questions.txt"""
import sys, time
from py65.devices.mpu6502 import MPU
from model import Net

ssd = open(sys.argv[1], "rb").read(); net = Net.load(sys.argv[2])
questions = [l for l in open(sys.argv[3]).read().split("\n") if l]

files = {}
for i in range(ssd[0x105] // 8):
    e = 8 + 8 * i
    name = ssd[e:e + 7].decode().strip()
    load = ssd[0x100 + e] | ssd[0x101 + e] << 8; ex = ssd[0x102 + e] | ssd[0x103 + e] << 8
    ln = ssd[0x104 + e] | ssd[0x105 + e] << 8; sec = ssd[0x107 + e] | (ssd[0x106 + e] & 3) << 8
    files[name] = (load, ex, ssd[sec * 256: sec * 256 + ln])

class BPlusMemory:
    """64K with ROM (reads &FF, ignores writes) at &8000-&BFFF unless ROMSEL bit 7 pages 12K RAM in."""
    def __init__(s): s.main = [0] * 0x10000; s.paged = [0] * 0x3000; s.romsel = 0x0F
    def __len__(s): return 0x10000
    def __getitem__(s, a):
        if isinstance(a, slice): return [s[i] for i in range(*a.indices(0x10000))]
        if 0x8000 <= a < 0xC000:
            return s.paged[a - 0x8000] if (s.romsel & 0x80 and a < 0xB000) else 0xFF
        return s.main[a]
    def __setitem__(s, a, v):
        if a == 0xFE30: s.romsel = v
        if 0x8000 <= a < 0xC000:
            if s.romsel & 0x80 and a < 0xB000: s.paged[a - 0x8000] = v
            return
        s.main[a] = v

mem = BPlusMemory(); mem.main[0xF4] = 0x0F
OSASCI, OSNEWL, OSWRCH, OSWORD, OSBYTE, OSCLI = 0xFFE3, 0xFFE7, 0xFFEE, 0xFFF1, 0xFFF4, 0xFFF7
for a in (OSASCI, OSNEWL, OSWRCH, OSWORD, OSBYTE, OSCLI): mem.main[a] = 0x60      # RTS
cpu = MPU(memory=mem); cpu.sp = 0xFF
def load(name, at=None):
    l, ex, data = files[name]
    for i, b in enumerate(data): mem[(at or l) + i] = b
    return ex
cpu.pc = load("BEEBLM")
out = []; pending = list(questions); results = []; mood = 0; asked = None; t0 = 0
while True:
    pc = cpu.pc
    if pc >= 0xFFE0:
        if pc == OSWRCH or pc == OSASCI: out.append(chr(cpu.a) if 32 <= cpu.a < 127 else ("\n" if cpu.a == 13 else ""))
        elif pc == OSNEWL: out.append("\n")
        elif pc == OSCLI:
            cmd = "".join(chr(mem[(cpu.x | cpu.y << 8) + i]) for i in range(20)).split("\r")[0].split()
            if cmd[0] == "LOAD": load(cmd[1])
            elif cmd[0] == "RUN": cpu.pc = load(cmd[1]); continue
            else: break                                           # *BASIC: finished
        elif pc == OSWORD and cpu.a == 1: mem[cpu.x | cpu.y << 8] = mood
        elif pc == OSWORD and cpu.a == 0:
            if asked is not None:
                results.append((asked, mood, "".join(out).strip().split("\n")[-2 if "".join(out).rstrip().endswith(">") else -1].strip(), (cpu.processorCycles - t0) / 2e6))
            out.clear()
            if not pending: cpu.p |= 1; cpu.step(); continue       # carry set = Escape
            asked = pending.pop(0); mood = len(results) & 1; t0 = cpu.processorCycles
            buf = mem[cpu.x | cpu.y << 8] | mem[(cpu.x | cpu.y << 8) + 1] << 8
            for i, ch in enumerate(asked + "\r"): mem[buf + i] = ord(ch)
            cpu.y = len(asked); cpu.p &= ~1
    cpu.step()
bad = 0
for q, m, got, secs in results:
    want = net.reply(q, m); bad += got != want
    print(f"{secs:5.1f}s  > {q}\n        {got}" + ("" if got == want else f"\n   REF: {want}"))
print(f"{len(results) - bad}/{len(results)} replies identical to the reference; paged RAM selected at exit: {bool(mem.romsel & 0x80)}")
