import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GamebotTrainingEnvironment } from '../src/environment.mjs';

const seed = '3JY1 6FLR';
const decisions = Number(process.argv[2] ?? 45);
if (!Number.isInteger(decisions) || decisions < 1) throw new RangeError('decisions must be positive');
const digest = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const sequences = [];
const details = [];

for (let repeat = 0; repeat < 2; repeat += 1) {
  const env = new GamebotTrainingEnvironment({ actionRepeat: 4, maxSteps: decisions * 4 + 1 });
  try {
    let observation = await env.reset({ seed });
    const hashes = [digest(observation)];
    let completed = 0;
    for (let index = 0; index < decisions; index += 1) {
      const result = await env.step({
        move: ['none', 'left', 'up', 'right', 'down'][index % 5],
        fire: ['down', 'left', 'up', 'right'][index % 4],
      });
      observation = result.observation;
      hashes.push(digest(observation));
      completed += result.info.framesAdvanced;
      if (result.terminated || result.truncated) break;
    }
    sequences.push(hashes);
    details.push({ decisions: hashes.length - 1, frames: completed, finalFrame: observation.frame,
      dead: observation.terminal.dead, cleared: observation.terminal.cleared });
  } finally {
    await env.close();
  }
}

const pass = JSON.stringify(sequences[0]) === JSON.stringify(sequences[1]);
assert.equal(pass, true, 'same-seed cold resets must reproduce every public observation');
const report = { schema: 'gamebot-determinism-check/1', pass, seed, actionRepeat: 4, details,
  observationHashes: sequences[0] };
const output = process.env.GAMEBOT_DETERMINISM_OUTPUT;
if (output) writeFileSync(resolve(output), `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({ pass, seed, details })}\n`);
