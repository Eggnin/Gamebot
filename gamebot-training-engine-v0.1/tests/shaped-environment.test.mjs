import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ShapedEnvironment, DEFAULT_SHAPING} from '../scripts/lib/shaped_environment.mjs';

function fakeEnvironment(script) {
  let index = 0;
  const observation = (terminal = {}) => ({
    schema: 'gamebot-room-observation/1', frame: index, room: {width: 15, height: 9},
    player: {position: {x: 0, y: 0}, velocity: {x: 0, y: 0}, health: 6}, entities: [], grid: [], terminal,
  });
  return {
    actionRepeat: 1,
    maxSteps: 100,
    observations: [],
    resets: 0,
    async reset() {
      this.resets += 1;
      index = 0;
      const result = observation();
      this.observations.push(result);
      return result;
    },
    async step() {
      const entry = script[Math.min(index, script.length - 1)];
      index += 1;
      const result = {
        observation: observation(entry.terminal ?? {}),
        reward: entry.environmentReward ?? 0,
        terminated: entry.terminal?.dead === true || entry.terminal?.cleared === true,
        truncated: false,
        info: {framesAdvanced: entry.frames ?? 1, enemyDamage: entry.dealt ?? 0, healthUnitLost: entry.lost ?? 0},
      };
      this.observations.push(result.observation);
      return result;
    },
    async close() {},
  };
}

test('shaped reward uses real damage, health loss, clear and death', async () => {
  const env = fakeEnvironment([
    {dealt: 10},
    {lost: 1},
    {terminal: {cleared: true}},
    {terminal: {dead: true}},
  ]);
  const shaped = new ShapedEnvironment(env);
  await shaped.reset();
  const damage = await shaped.step({move: 'up', fire: 'left'});
  assert.ok(Math.abs(damage.reward - (10 * DEFAULT_SHAPING.damage - DEFAULT_SHAPING.frame)) < 1e-9);
  assert.equal(damage.rewardComponents.damage, 10);
  assert.equal(damage.environmentReward, 0);

  const hurt = await shaped.step({move: 'down', fire: 'left'});
  assert.ok(Math.abs(hurt.reward - (-DEFAULT_SHAPING.healthLost - DEFAULT_SHAPING.frame)) < 1e-9);

  const clear = await shaped.step({move: 'none', fire: 'left'});
  assert.equal(clear.rewardComponents.clear, DEFAULT_SHAPING.clear);
  assert.ok(clear.reward > 0);

  const death = await shaped.step({move: 'up', fire: 'none'});
  assert.equal(death.rewardComponents.death, -DEFAULT_SHAPING.death);
  assert.ok(death.reward < 0);
});

test('anti-stall starts only after the configured no-damage window and reset clears it', async () => {
  const env = fakeEnvironment([{frames: 100, dealt: 0}, {frames: 30, dealt: 0}, {frames: 1, dealt: 0}]);
  const shaped = new ShapedEnvironment(env);
  await shaped.reset();
  const first = await shaped.step({move: 'up', fire: 'left'});
  assert.equal(first.rewardComponents.stall, 0, '100 frames without damage is still inside the window');
  const second = await shaped.step({move: 'up', fire: 'left'});
  assert.ok(second.rewardComponents.stall < 0, 'crossing 120 frames starts the stall penalty');
  assert.ok(Math.abs(second.rewardComponents.stall + 30 * DEFAULT_SHAPING.noDamagePerFrame) < 1e-9);
  await shaped.reset();
  const afterReset = await shaped.step({move: 'up', fire: 'left'});
  assert.equal(afterReset.rewardComponents.stall, 0, 'reset must clear the stall counter');
});

test('idle decisions are penalized but not terminal actions', async () => {
  const env = fakeEnvironment([{}, {terminal: {cleared: true}}]);
  const shaped = new ShapedEnvironment(env);
  await shaped.reset();
  const idle = await shaped.step({move: 'none', fire: 'none'});
  assert.equal(idle.rewardComponents.idle, -DEFAULT_SHAPING.idlePenalty);
  const cleared = await shaped.step({move: 'none', fire: 'none'});
  assert.equal(cleared.rewardComponents.idle, 0);
  assert.equal(cleared.rewardComponents.clear, DEFAULT_SHAPING.clear);
});

test('shaped environment forwards actionRepeat/maxSteps and rejects invalid inputs', async () => {
  const env = fakeEnvironment([{}]);
  const shaped = new ShapedEnvironment(env);
  assert.equal(shaped.actionRepeat, 1);
  assert.equal(shaped.maxSteps, 100);
  assert.throws(() => new ShapedEnvironment(null), /reset\(\) and step\(\)/);
});
