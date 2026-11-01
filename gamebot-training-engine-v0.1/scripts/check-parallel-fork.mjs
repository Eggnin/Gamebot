import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GamebotTrainingEnvironment } from '../src/environment.mjs';

const runRoot = process.env.GAMEBOT_RUN_ROOT;
const envOptions = { actionRepeat: 1, maxSteps: 2000, ...(runRoot ? { runRoot } : {}) };
const base = new GamebotTrainingEnvironment(envOptions);
let left;
let right;
const digest = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const runActions = async (env, actions) => {
  const hashes = [];
  for (const action of actions) {
    const result = await env.step(action);
    hashes.push(digest(result.observation));
    if (result.terminated || result.truncated) break;
  }
  return hashes;
};

const report = { schema: 'gamebot-parallel-fork-check/1', pass: false };
try {
  const initial = await base.reset({ seed: '3JY1 6FLR', scene: 'monstro' });
  for (let i = 0; i < 30; i += 1) {
    await base.step({ move: ['none', 'right', 'up'][i % 3], fire: 'up' });
  }
  const checkpoint = await base.snapshot('parallel-fork-root');
  const exportStarted = performance.now();
  const exportedSnapshot = await base.exportSnapshot(checkpoint.id);
  const [leftBranch, rightBranch] = await Promise.all([
    base.forkParallel(checkpoint.id, 'parallel-left'),
    base.forkParallel(checkpoint.id, 'parallel-right'),
  ]);
  left = leftBranch;
  right = rightBranch;
  report.snapshotExportAndForkMs = performance.now() - exportStarted;
  report.snapshotBytes = checkpoint.bytes;
  report.distinctWorkerPids = left.workerPid !== right.workerPid;
  report.distinctRunDirectories = left.runDirectory !== right.runDirectory;
  report.leftStartMatchesSnapshot = digest(await left.observe()) === checkpoint.publicObservationHash;
  report.rightStartMatchesSnapshot = digest(await right.observe()) === checkpoint.publicObservationHash;
  assert.equal(report.distinctWorkerPids, true);
  assert.equal(report.distinctRunDirectories, true);
  assert.equal(report.leftStartMatchesSnapshot, true);
  assert.equal(report.rightStartMatchesSnapshot, true);

  const actionsLeft = Array.from({ length: 30 }, (_, i) => ({
    move: i < 15 ? 'right' : 'left', fire: 'up',
  }));
  const actionsRight = Array.from({ length: 30 }, (_, i) => ({
    move: i < 15 ? 'left' : 'right', fire: 'down',
  }));
  const [baseHashes, leftHashes, rightHashes] = await Promise.all([
    runActions(base, actionsLeft),
    runActions(left, actionsLeft),
    runActions(right, actionsRight),
  ]);
  report.crossProcessReplayMatched = JSON.stringify(baseHashes) === JSON.stringify(leftHashes);
  assert.equal(report.crossProcessReplayMatched, true);
  assert.notDeepEqual(leftHashes, rightHashes);
  const restoreStarted = performance.now();
  const leftRestored = await left.restore(left.anchorSnapshotId);
  report.fileBackedRestoreMs = performance.now() - restoreStarted;
  report.fileBackedRestoreMatchesSnapshot = digest(leftRestored) === checkpoint.publicObservationHash;
  assert.equal(report.fileBackedRestoreMatchesSnapshot, true);
  const leftReplay = await runActions(left, actionsLeft);
  report.leftBranchReplayMatched = JSON.stringify(leftHashes) === JSON.stringify(leftReplay);
  report.rightBranchStayedIndependent = digest(await right.observe()) !== checkpoint.publicObservationHash;
  assert.equal(report.leftBranchReplayMatched, true);
  assert.equal(report.rightBranchStayedIndependent, true);
  report.branchFrames = { left: leftHashes.length, right: rightHashes.length };
  const leftSnapshotPath = left.snapshots.get(left.anchorSnapshotId).exportedPath;
  const rightSnapshotPath = right.snapshots.get(right.anchorSnapshotId).exportedPath;
  await base.dropSnapshot(checkpoint.id);
  report.exportedSnapshotRemovedOnDrop = !existsSync(exportedSnapshot.path) &&
    !existsSync(`${exportedSnapshot.path}.json`);
  assert.equal(report.exportedSnapshotRemovedOnDrop, true);
  const leftAfterSourceDrop = await left.restore(left.anchorSnapshotId);
  report.branchSnapshotSurvivedSourceDrop = digest(leftAfterSourceDrop) === checkpoint.publicObservationHash &&
    existsSync(leftSnapshotPath) && existsSync(rightSnapshotPath);
  assert.equal(report.branchSnapshotSurvivedSourceDrop, true);
  await Promise.all([
    left.dropSnapshot(left.anchorSnapshotId),
    right.dropSnapshot(right.anchorSnapshotId),
  ]);
  report.branchSnapshotFilesCleaned = !existsSync(leftSnapshotPath) && !existsSync(rightSnapshotPath);
  assert.equal(report.branchSnapshotFilesCleaned, true);
  report.pass = true;
} finally {
  await Promise.all([base.close(), left?.close(), right?.close()]);
}

const output = resolve(process.env.GAMEBOT_PARALLEL_FORK_REPORT ?? 'evidence/parallel-fork-check.json');
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(report)}\n`);
