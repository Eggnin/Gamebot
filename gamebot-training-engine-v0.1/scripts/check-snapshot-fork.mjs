import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GamebotTrainingEnvironment } from '../src/environment.mjs';

const env = new GamebotTrainingEnvironment({ actionRepeat: 1, maxSteps: 2000 });
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const report = { schema: 'gamebot-snapshot-fork-check/1', pass: false, checks: {} };

async function rollout(actions) {
  const hashes = [];
  for (const action of actions) {
    const result = await env.step(action);
    hashes.push(hash(result.observation));
    if (result.terminated || result.truncated) break;
  }
  return hashes;
}

try {
  const coldStart = performance.now();
  const initial = await env.reset({ seed: '3JY1 6FLR', scene: 'monstro' });
  report.coldSetupMs = performance.now() - coldStart;
  const anchor = env.anchorSnapshotId;
  assert.ok(anchor !== null);

  const prefill = Array.from({ length: 30 }, (_, i) => ({
    move: ['none', 'right', 'up', 'left', 'down'][i % 5],
    fire: ['up', 'down', 'left', 'right'][i % 4],
  }));
  await rollout(prefill);
  const snapshotStarted = performance.now();
  const checkpoint = await env.snapshot('mid-room-checkpoint');
  report.snapshotCaptureMs = performance.now() - snapshotStarted;
  report.snapshotBytes = checkpoint.bytes;
  report.snapshotMemorySha256 = checkpoint.memorySha256;
  report.checks.snapshotCapturedAtFrame = checkpoint.frame;

  const sequenceA = Array.from({ length: 60 }, (_, i) => ({
    move: i % 40 < 20 ? 'right' : 'left', fire: 'up',
  }));
  const snapshotObservationHash = checkpoint.publicObservationHash;
  const branchAFirst = await rollout(sequenceA);
  const restoreStarted = performance.now();
  const restored = await env.restore(checkpoint.id);
  report.restoreMs = performance.now() - restoreStarted;
  report.checks.restoreMatchedCheckpoint = hash(restored) === snapshotObservationHash;
  const branchAReplay = await rollout(sequenceA);
  report.checks.restoreReplayMatched = JSON.stringify(branchAFirst) === JSON.stringify(branchAReplay);

  const left = await env.fork(checkpoint.id, 'left-branch');
  const right = await env.fork(checkpoint.id, 'right-branch');
  const branchBSequence = Array.from({ length: 60 }, (_, i) => ({
    move: i % 40 < 20 ? 'left' : 'right', fire: 'down',
  }));
  await env.restore(left);
  const branchA = await rollout(sequenceA);
  await env.restore(right);
  const branchB = await rollout(branchBSequence);
  await env.restore(left);
  const branchAAgain = await rollout(sequenceA);
  report.checks.forksShareCheckpoint = left.snapshotId === right.snapshotId && left.snapshotId === checkpoint.id;
  report.checks.leftBranchWasIndependent = JSON.stringify(branchA) === JSON.stringify(branchAAgain);
  report.checks.branchesHadDistinctTrajectories = JSON.stringify(branchA) !== JSON.stringify(branchB);
  report.forks = [left, right];

  await env.restore(anchor);
  report.checks.anchorRestoreMatchedColdReset = hash(await env.observe()) === hash(initial);
  const resetStarted = performance.now();
  const resetObservation = await env.reset({ seed: '3JY1 6FLR', scene: 'monstro' });
  report.snapshotResetMs = performance.now() - resetStarted;
  report.checks.automaticResetUsedSnapshot = env.resetMode === 'snapshot-restore';
  report.checks.automaticResetMatchedInitial = hash(resetObservation) === hash(initial);
  report.pass = Object.values(report.checks).every(value => value === true || typeof value === 'number');
  report.checks = { ...report.checks, snapshotRestore: true, forkReplay: true, reset: true };
} finally {
  await env.close();
}

const output = resolve(process.env.GAMEBOT_SNAPSHOT_REPORT ?? 'evidence/snapshot-fork-check.json');
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  pass: report.pass,
  coldSetupMs: report.coldSetupMs,
  snapshotCaptureMs: report.snapshotCaptureMs,
  restoreMs: report.restoreMs,
  snapshotResetMs: report.snapshotResetMs,
  snapshotBytes: report.snapshotBytes,
  checks: report.checks,
})}\n`);
