import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { GamebotTrainingEnvironment, sampleWorkerMemory } from '../src/environment.mjs';

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const arg = name => process.argv.slice(2).find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3);
const instances = Number(arg('instances') ?? 2);
const decisions = Number(arg('decisions') ?? 120);
const actionRepeat = Number(arg('action-repeat') ?? 4);
const maxSteps = Number(arg('max-frames') ?? decisions * actionRepeat + 1);
const seeds = ['3JY1 6FLR', '7W3L P1Z9', 'SWGF G7WV', 'P1Z9 7W3L'];
if (!Number.isInteger(instances) || instances < 1 || instances > seeds.length) throw new RangeError('instances must be 1–4');
if (!Number.isInteger(decisions) || decisions < 1) throw new RangeError('decisions must be positive');
if (!Number.isInteger(actionRepeat) || actionRepeat < 1 || actionRepeat > 16) throw new RangeError('action-repeat must be 1–16');

const envs = Array.from({ length: instances }, () => new GamebotTrainingEnvironment({ actionRepeat, maxSteps }));
const runs = [];
try {
  const setupStarted = performance.now();
  const initial = await Promise.all(envs.map((env, i) => env.reset({ seed: seeds[i] })));
  const setupMs = performance.now() - setupStarted;
  const workerPids = envs.map(env => env.workerPid);
  const memoryBefore = workerPids.map(sampleWorkerMemory);
  const counters = envs.map((env, index) => ({
    index,
    seed: seeds[index],
    runDirectory: env.runDirectory,
    workerPid: workerPids[index],
    initialFrame: initial[index].frame,
    initialRoom: initial[index].room,
    frames: 0,
    decisions: 0,
    reward: 0,
    last: null,
    memoryBefore: memoryBefore[index],
  }));

  const moves = ['none', 'up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left'];
  const fires = ['down', 'left', 'up', 'right'];
  const start = performance.now();
  for (let decision = 0; decision < decisions; decision += 1) {
    const active = counters.filter(row => !row.last?.terminated && !row.last?.truncated);
    if (!active.length) break;
    await Promise.all(active.map(async row => {
      const env = envs[row.index];
      const result = await env.step({
        move: moves[(decision + row.index * 2) % moves.length],
        fire: fires[(decision + row.index) % fires.length],
      });
      row.frames += result.info.framesAdvanced;
      row.decisions += 1;
      row.reward += result.reward;
      row.last = result;
    }));
  }
  const elapsedMs = performance.now() - start;
  const memoryAfter = workerPids.map(sampleWorkerMemory);
  for (let index = 0; index < counters.length; index += 1) counters[index].memoryAfter = memoryAfter[index];
  const totalFrames = counters.reduce((sum, row) => sum + row.frames, 0);
  const elapsedSeconds = elapsedMs / 1000;
  const report = {
    schema: 'gamebot-training-benchmark/1',
    engine: 'The Browsing of Isaac training WASM / Node headless runtime',
    gameBuildSha256: '7122AC28779925B24E23E2416F231322B1470388BD25E2C08665AD8D53B3EA4F',
    segmentImageSha256: envs[0].runtimeMetadata.segmentImageSha256,
    bootWasmSha256: envs[0].runtimeMetadata.bootWasmSha256,
    room: 's.boss.1010',
    actionRepeat,
    requestedDecisionsPerInstance: decisions,
    setupMs,
    elapsedMs,
    totalFrames,
    aggregateFramesPerSecond: totalFrames / elapsedSeconds,
    aggregateRealtimeMultipleAt30Hz: totalFrames / elapsedSeconds / 30,
    instances: counters.map(row => ({
      index: row.index,
      seed: row.seed,
      runDirectory: row.runDirectory,
      workerPid: row.workerPid,
      initialFrame: row.initialFrame,
      finalFrame: row.last?.observation?.frame ?? row.initialFrame,
      frames: row.frames,
      decisions: row.decisions,
      framesPerSecond: row.frames / elapsedSeconds,
      realtimeMultipleAt30Hz: row.frames / elapsedSeconds / 30,
      totalReward: row.reward,
      terminated: row.last?.terminated ?? false,
      truncated: row.last?.truncated ?? false,
      healthEnd: row.last?.observation?.player?.health ?? initial[row.index].player.health,
      memoryBefore: row.memoryBefore,
      memoryAfter: row.memoryAfter,
      memorySampleMethod: 'Windows Get-Process WorkingSet64 and PrivateMemorySize64',
    })),
    limitations: [
      'Short deterministic action sweep, not a policy-quality benchmark.',
      'Memory values are point samples before and after the timed rollout, not a sampled peak.',
      'Real-time multiple uses the verified 30 Hz GameFrame clock.',
    ],
  };
  const out = resolve(arg('output') ?? process.env.GAMEBOT_BENCH_OUTPUT ?? `${packageRoot}/evidence/latest-benchmark.json`);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ output: out, ...report })}\n`);
} finally {
  await Promise.all(envs.map(env => env.close()));
}
