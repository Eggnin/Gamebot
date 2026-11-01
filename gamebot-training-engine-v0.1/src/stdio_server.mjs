import { createInterface } from 'node:readline';
import { GamebotTrainingEnvironment } from './environment.mjs';

const environment = new GamebotTrainingEnvironment();
const parallelForks = new Map();
let nextParallelFork = 1;
const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
const send = value => process.stdout.write(`${JSON.stringify(value)}\n`);

try {
  for await (const line of input) {
    if (!line.trim()) continue;
    let message;
    try {
      message = JSON.parse(line);
      if (message.command === 'reset') {
        const observation = await environment.reset({ seed: message.seed, scene: message.scene, room: message.room });
        send({ ok: true, observation, runDirectory: environment.runDirectory, resetMode: environment.resetMode });
      } else if (message.command === 'observe') {
        send({ ok: true, observation: await environment.observe() });
      } else if (message.command === 'step') {
        const result = await environment.step(message.action);
        send({ ok: true, ...result });
      } else if (message.command === 'snapshot') {
        send({ ok: true, snapshot: await environment.snapshot(message.label) });
      } else if (message.command === 'restore') {
        send({ ok: true, observation: await environment.restore(message.snapshotId ?? message.fork) });
      } else if (message.command === 'fork') {
        send({ ok: true, branch: await environment.fork(message.snapshotId, message.label) });
      } else if (message.command === 'fork_parallel') {
        const branchId = `parallel-${nextParallelFork++}`;
        const branch = await environment.forkParallel(message.snapshotId, message.label);
        parallelForks.set(branchId, branch);
        send({ ok: true, branchId, seed: branch.seed, scene: branch.scene.key,
          workerPid: branch.workerPid, runDirectory: branch.runDirectory,
          observation: await branch.observe() });
      } else if (message.command === 'branch_step') {
        const branch = parallelForks.get(message.branchId);
        if (!branch) throw new TypeError(`unknown parallel branch: ${message.branchId}`);
        send({ ok: true, ...await branch.step(message.action) });
      } else if (message.command === 'parallel_step') {
        const requests = message.actions;
        if (!Array.isArray(requests) || requests.some(row => !parallelForks.has(row.branchId))) {
          throw new TypeError('actions must list known parallel branchId/action pairs');
        }
        const results = await Promise.all(requests.map(async row => ({
          branchId: row.branchId,
          ...(await parallelForks.get(row.branchId).step(row.action)),
        })));
        send({ ok: true, results });
      } else if (message.command === 'branch_restore') {
        const branch = parallelForks.get(message.branchId);
        if (!branch) throw new TypeError(`unknown parallel branch: ${message.branchId}`);
        send({ ok: true, observation: await branch.restore(message.snapshotId ?? branch.anchorSnapshotId) });
      } else if (message.command === 'branch_close') {
        const branch = parallelForks.get(message.branchId);
        if (!branch) throw new TypeError(`unknown parallel branch: ${message.branchId}`);
        await branch.close();
        parallelForks.delete(message.branchId);
        send({ ok: true, closed: true });
      } else if (message.command === 'drop_snapshot') {
        send({ ok: true, dropped: await environment.dropSnapshot(message.snapshotId ?? message.fork) });
      } else if (message.command === 'close') {
        await Promise.all([...parallelForks.values()].map(branch => branch.close()));
        parallelForks.clear();
        await environment.close();
        send({ ok: true, closed: true });
        break;
      } else {
        throw new TypeError('command must be reset, observe, step, snapshot, restore, fork, drop_snapshot, or close');
      }
    } catch (error) {
      send({ ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  }
} finally {
  await Promise.all([...parallelForks.values()].map(branch => branch.close()));
  parallelForks.clear();
  await environment.close();
  input.close();
}
