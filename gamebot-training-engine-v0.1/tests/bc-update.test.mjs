import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ActorCriticNetwork, encodeObservation} from '../src/actor_critic.mjs';

function observation({x = 0, y = 0, vx = 0, vy = 0} = {}) {
  return {
    schema: 'gamebot-room-observation/1',
    frame: 0,
    room: {width: 15, height: 9},
    player: {position: {x, y}, velocity: {x: vx, y: vy}, health: 6, fireDelay: 10,
      damageCooldown: 0, controls: true},
    entities: [],
    grid: [],
    terminal: {cleared: false, dead: false},
  };
}

// Linearly separable synthetic task: the encoder sees the player position, so
// the two heads can learn "move toward the target side" and "fire along the
// target axis". It proves the supervised gradient path actually learns.
function separableDataset(count = 96) {
  const samples = [];
  for (let index = 0; index < count; index++) {
    const positiveX = index % 2 === 0;
    const positiveY = Math.floor(index / 2) % 2 === 0;
    samples.push({
      observation: observation({x: positiveX ? 260 : -260, y: positiveY ? 140 : -140}),
      action: {move: positiveX ? 'right' : 'left', fire: positiveY ? 'down' : 'up'},
      weight: 1,
    });
  }
  return samples;
}

function accuracyAgainst(policy, samples) {
  let move = 0, fire = 0, joint = 0;
  for (const sample of samples) {
    const decision = policy.act(sample.observation, {deterministic: true});
    move += decision.action.move === sample.action.move ? 1 : 0;
    fire += decision.action.fire === sample.action.fire ? 1 : 0;
    joint += decision.action.move === sample.action.move && decision.action.fire === sample.action.fire ? 1 : 0;
  }
  return {move: move / samples.length, fire: fire / samples.length, joint: joint / samples.length};
}

test('supervised update learns separable expert actions', () => {
  const policy = new ActorCriticNetwork({seed: 7});
  const samples = separableDataset();
  const initial = accuracyAgainst(policy, samples);
  const first = policy.updateSupervised(samples, {epochs: 1, batchSize: 32, learningRate: 5e-3, entropyCoefficient: 0});
  assert.ok(Number.isFinite(first.moveLoss) && first.moveLoss > 0);
  assert.ok(first.moveLoss > 0.5 && first.fireLoss > 0.5, 'untrained losses should be far from zero');
  for (let round = 0; round < 30; round++) {
    policy.updateSupervised(samples, {epochs: 1, batchSize: 32, learningRate: 5e-3, entropyCoefficient: 0});
  }
  const final = accuracyAgainst(policy, samples);
  assert.ok(final.move > 0.9, `move accuracy ${final.move.toFixed(3)}`);
  assert.ok(final.fire > 0.9, `fire accuracy ${final.fire.toFixed(3)}`);
  assert.ok(final.joint > 0.9, `joint accuracy ${final.joint.toFixed(3)}`);
  assert.ok(final.joint > initial.joint + 0.3, 'training should improve over the untrained policy');
});

test('supervised updates reuse the PPO optimizer state and checkpoint format', () => {
  const policy = new ActorCriticNetwork({seed: 11});
  const samples = separableDataset(32);
  const before = policy.optimizer.step;
  policy.updateSupervised(samples, {epochs: 2, batchSize: 16});
  assert.ok(policy.optimizer.step > before, 'Adam step counter must advance');
  const payload = JSON.parse(policy.save());
  assert.equal(payload.format, 'gamebot-actor-critic/1');
  assert.ok(Array.isArray(payload.optimizer.state) && payload.optimizer.state.length > 0);

  const restored = ActorCriticNetwork.fromJSON(payload);
  const observationSample = samples[0].observation;
  const original = policy.forward(observationSample);
  const copy = restored.forward(observationSample);
  for (let index = 0; index < original.moveProbabilities.length; index++) {
    assert.ok(Math.abs(original.moveProbabilities[index] - copy.moveProbabilities[index]) < 1e-9);
  }
  for (let index = 0; index < original.fireProbabilities.length; index++) {
    assert.ok(Math.abs(original.fireProbabilities[index] - copy.fireProbabilities[index]) < 1e-9);
  }
  assert.ok(Math.abs(original.value - copy.value) < 1e-9);

  // The restored checkpoint must continue the same optimizer timeline.
  const beforeStep = restored.optimizer.step;
  restored.updateSupervised(samples, {epochs: 1, batchSize: 16});
  assert.ok(restored.optimizer.step > beforeStep);
});

test('zero-weight rows do not move the weights', () => {
  const policy = new ActorCriticNetwork({seed: 3});
  const weights = policy.layers.map(layer => Array.from(layer.weights));
  policy.updateSupervised([{observation: observation(), action: {move: 'up', fire: 'left'}, weight: 0}],
    {epochs: 1, batchSize: 1});
  policy.layers.forEach((layer, layerIndex) => {
    for (let index = 0; index < layer.weights.length; index++) {
      assert.ok(Math.abs(layer.weights[index] - weights[layerIndex][index]) < 1e-12);
    }
  });
});

test('supervised update rejects unknown labels and observations', () => {
  const policy = new ActorCriticNetwork({seed: 5});
  assert.throws(() => policy.updateSupervised([{observation: observation(), action: {move: 'teleport', fire: 'none'}}]),
    /move must be one of/);
  assert.throws(() => policy.updateSupervised([{observation: observation(), action: {move: 'up', fire: 'bomb'}}]),
    /fire must be one of/);
  assert.throws(() => policy.updateSupervised([], {}), /non-empty array/);
  assert.throws(() => encodeObservation(null), /must be an object/);
});
