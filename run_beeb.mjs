// Boot a BBC Micro in jsbeeb, run BEEBLM from the disc image and type questions at it.
// usage: node run_beeb.mjs disc.ssd questions.txt  -> JSON lines {q, mood, reply, secs}
import fs from "node:fs";
import { TestMachine } from "../tools/jsbeeb/src/test-machine.js";

const [ssd, qfile, clockAddr = "0x8a"] = process.argv.slice(2);
const questions = fs.readFileSync(qfile, "utf8").split("\n").filter((l) => l.length);
const log = console.log; console.log = () => {};
const m = new TestMachine("B-DFS1.2");
await m.initialise();
await m.runUntilInput();
m.loadDiscData(fs.readFileSync(ssd));
m.startCapture();
let buf = "";
async function until(pred, maxSecs) {
    for (let c = 0; c < maxSecs * 2e6; c += 20000) {
        await m.runFor(20000);
        buf += m.drainText({ raw: true });
        if (pred(buf)) return true;
    }
    return false;
}
await m.type("*RUN BEEBLM");
if (!(await until((b) => b.endsWith("> "), 30))) { log(JSON.stringify({ error: "no prompt", buf })); process.exit(1); }
log(JSON.stringify({ banner: buf }));
for (const q of questions) {
    buf = "";
    await m.type(q);
    await until((b) => b.includes("\n"), 5);          // echo finished
    const t0 = m.elapsedCycles;
    const ok = await until((b) => b.endsWith("> "), 120);
    const secs = (m.elapsedCycles - t0) / 2e6;
    const lines = buf.split("\n");
    log(JSON.stringify({ q, mood: m.readbyte(Number(clockAddr)) & 1, reply: lines[1] ?? "", echo: lines[0], secs: +secs.toFixed(2), ok }));
}
process.exit(0);
