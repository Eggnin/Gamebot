import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GamebotTrainingEnvironment, ROOM_SCENARIOS } from '../src/environment.mjs';

const env = new GamebotTrainingEnvironment({
  actionRepeat: 4,
  maxSteps: 5000,
  requestTimeoutMs: Number(process.env.GAMEBOT_REQUEST_TIMEOUT_MS ?? 30000),
});
const seeds = ['3JY1 6FLR', '7W3L P1Z9', 'SWGF G7WV'];
const cases = Object.values(ROOM_SCENARIOS).map((scenario, index) => ({
  scene: scenario.key,
  seed: seeds[index % seeds.length],
  variant: scenario.variant,
  targetNPC: scenario.targetNPC,
}));
const autoScenarios = Object.values(ROOM_SCENARIOS).filter(scenario => scenario.type === 5);
const expectedAuto = autoScenarios[cases.length % autoScenarios.length];
cases.push({ scene: 'auto', variant: expectedAuto.variant, targetNPC: expectedAuto.targetNPC });
const report = { schema: 'gamebot-auto-reset-check/1', pass: false, episodes: [] };

try {
  for (const requested of cases) {
    const started = performance.now();
    const observation = await env.reset({ scene: requested.scene, seed: requested.seed });
    const row = {
      requestedScene: requested.scene,
      actualScene: env.scene.key,
      room: observation.room,
      seed: env.seed,
      resetMode: env.resetMode,
      resetMs: performance.now() - started,
      targetObserved: observation.entities.some(entity => entity.type === requested.targetNPC),
    };
    assert.equal(observation.room.variant, requested.variant);
    assert.equal(row.targetObserved, true);
    if (report.episodes.length > 0) assert.equal(row.resetMode, 'warm-console');
    report.episodes.push(row);
  }
  const warm = report.episodes.filter(row => row.resetMode === 'warm-console');
  report.summary = {
    distinctScenarios: new Set(report.episodes.map(row => row.actualScene)).size,
    warmResets: warm.length,
    averageWarmResetMs: warm.reduce((sum, row) => sum + row.resetMs, 0) / Math.max(1, warm.length),
    coldResets: report.episodes.filter(row => row.resetMode === 'cold').length,
  };
  report.pass = report.episodes.length === cases.length &&
    report.episodes.every((row, index) => row.actualScene === (cases[index].scene === 'auto'
      ? expectedAuto.key : cases[index].scene));
} finally {
  await env.close();
}

const output = resolve(process.env.GAMEBOT_AUTO_RESET_REPORT ?? 'evidence/auto-reset-check.json');
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({ pass: report.pass, episodes: report.episodes.map(({ requestedScene, actualScene, room, seed, resetMode, resetMs, targetObserved }) => ({ requestedScene, actualScene, variant: room.variant, seed, resetMode, resetMs, targetObserved })) })}\n`);
