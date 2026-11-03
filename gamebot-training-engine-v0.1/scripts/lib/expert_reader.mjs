// Native external expert process (ExternalReader.exe) used by both the Phase 1
// collector and the DAgger collector. It speaks one JSON object per line.
import {spawn} from 'node:child_process';
import {appendFileSync} from 'node:fs';
import {createInterface} from 'node:readline';

export class ExpertReader {
  constructor({exe, semantics, logPath}) {
    this.logPath = logPath;
    this.pending = null;
    this.dead = false;
    this.child = spawn(exe, [semantics], {windowsHide: true, stdio: ['pipe', 'pipe', 'pipe']});
    createInterface({input: this.child.stdout}).on('line', line => {
      const pending = this.pending;
      if (!pending) return;
      this.pending = null;
      clearTimeout(pending.timer);
      try { pending.resolve(JSON.parse(line)); }
      catch (error) { pending.reject(new Error(`expert reader returned invalid JSON: ${error.message}`)); }
    });
    this.child.stderr.on('data', chunk => appendFileSync(this.logPath, chunk));
    const fail = error => {
      this.dead = true;
      const pending = this.pending;
      if (!pending) return;
      this.pending = null;
      clearTimeout(pending.timer);
      pending.reject(error);
    };
    this.child.on('error', fail);
    this.child.on('exit', code => fail(new Error(`expert reader exited with ${code}`)));
  }

  capture(payload, timeoutMs = 180000) {
    if (this.dead) return Promise.reject(new Error('expert reader is not running'));
    if (this.pending) return Promise.reject(new Error('expert reader capture already pending'));
    return new Promise((resolveCapture, rejectCapture) => {
      const timer = setTimeout(() => {
        this.pending = null;
        this.dead = true;
        this.child.kill();
        rejectCapture(new Error(`expert reader did not answer within ${timeoutMs} ms`));
      }, timeoutMs);
      this.pending = {resolve: resolveCapture, reject: rejectCapture, timer};
      this.child.stdin.write(`${JSON.stringify(payload)}\n`);
    });
  }

  async close() {
    if (this.child.exitCode !== null) return;
    await new Promise(done => {
      this.child.once('exit', done);
      this.child.stdin.end();
      setTimeout(() => { try { this.child.kill(); } catch { /* already gone */ } }, 5000).unref();
    });
  }
}
