import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const serverPath = resolve(packageRoot, 'src', 'stdio_server.mjs');
const child = spawn(process.execPath, [serverPath], {
  cwd: packageRoot,
  windowsHide: true,
  env: process.env,
  stdio: ['pipe', 'pipe', 'pipe'],
});
const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
const iterator = lines[Symbol.asyncIterator]();
let stderr = '';
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
child.stderr.setEncoding('utf8');
child.stderr.on('data', value => { stderr += value; });

async function request(message) {
  child.stdin.write(`${JSON.stringify(message)}\n`);
  const next = await iterator.next();
  if (next.done) throw new Error(`server exited before replying: ${stderr}`);
  const result = JSON.parse(next.value);
  if (!result.ok) throw new Error(result.error);
  return result;
}

try {
  const reset = await request({ command: 'reset', seed: '3JY1 6FLR' });
  let step;
  let sawPlayerTear = false;
  for (let index = 0; index < 16; index += 1) {
    step = await request({ command: 'step', action: { move: 'right', fire: 'down' } });
    sawPlayerTear ||= step.observation.entities.some(entity => entity.type === 2);
  }
  const observed = await request({ command: 'observe' });
  assert.equal(step.observation.frame, reset.observation.frame + 64);
  assert.equal(observed.observation.frame, step.observation.frame);
  assert.equal(step.info.framesAdvanced, 4);
  assert.equal(step.observation.room.variant, 1010);
  assert.ok(step.observation.player.position.x > reset.observation.player.position.x + 20);
  assert.equal(sawPlayerTear, true);
  assert.equal(step.observation.entities.some(entity => [4, 5].includes(entity.type)), false);

  const snapshot = await request({ command: 'snapshot', label: 'stdio-smoke-root' });
  const fork = await request({ command: 'fork', snapshotId: snapshot.snapshot.id, label: 'stdio-smoke-branch' });
  let branchStep;
  for (let index = 0; index < 8; index += 1) {
    branchStep = await request({ command: 'step', action: { move: 'left', fire: 'up' } });
  }
  const branchRestore = await request({ command: 'restore', fork: fork.branch.id });
  assert.equal(hash(branchRestore.observation), snapshot.snapshot.publicObservationHash);
  assert.notEqual(branchStep.observation.frame, branchRestore.observation.frame);

  const parallel = await request({ command: 'fork_parallel', snapshotId: snapshot.snapshot.id, label: 'stdio-parallel' });
  assert.notEqual(parallel.workerPid, null);
  assert.equal(hash(parallel.observation), snapshot.snapshot.publicObservationHash);
  const parallelStep = await request({
    command: 'branch_step', branchId: parallel.branchId,
    action: { move: 'left', fire: 'down' },
  });
  assert.equal(parallelStep.observation.frame, snapshot.snapshot.frame + 4);
  await request({ command: 'branch_close', branchId: parallel.branchId });

  const warmReset = await request({ command: 'reset', seed: '7W3L P1Z9', scene: 'larryJr' });
  assert.equal(warmReset.resetMode, 'warm-console');
  assert.equal(warmReset.observation.room.variant, 1020);
  assert.ok(warmReset.observation.entities.some(entity => entity.type === 19));
  const exited = new Promise((resolveExit, reject) => {
    child.once('error', reject);
    child.once('exit', code => resolveExit(code));
  });
  await request({ command: 'close' });
  child.stdin.end();
  const exitCode = await exited;
  assert.equal(exitCode, 0, stderr);
  process.stdout.write(`${JSON.stringify({
    pass: true,
    initialFrame: reset.observation.frame,
    finalFrame: step.observation.frame,
    externalActions: 16,
    framesPerExternalAction: step.info.framesAdvanced,
    playerXStart: reset.observation.player.position.x,
    playerXEnd: step.observation.player.position.x,
    playerTearObserved: sawPlayerTear,
    room: step.observation.room,
    reportedEntityCount: step.observation.entities.length,
    snapshotBytes: snapshot.snapshot.bytes,
    forkBranch: fork.branch.id,
    restoredFrame: branchRestore.observation.frame,
    parallelBranch: parallel.branchId,
    parallelBranchFrame: parallelStep.observation.frame,
    warmResetScene: warmReset.observation.room.variant,
    warmResetMode: warmReset.resetMode,
    runDirectory: reset.runDirectory,
    trajectoryFiles: ['trajectory.jsonl', 'trajectory-0002.jsonl'],
  })}\n`);
} finally {
  if (child.exitCode === null) child.kill();
  lines.close();
}
