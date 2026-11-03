import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ActorCriticNetwork, encodeObservation, MOVE_ACTIONS, FIRE_ACTIONS} from '../src/actor_critic.mjs';

function observation() {
  return {
    schema: 'gamebot-room-observation/1', frame: 0, room: {width: 15, height: 9},
    player: {position: {x: 120, y: 80}, velocity: {x: 2, y: -1}, health: 6, fireDelay: 10,
      damageCooldown: 0, controls: true},
    entities: [{kind: 'enemy', type: 10, variant: 0, subtype: 0, position: {x: 300, y: 200},
      velocity: {x: 0, y: 0}, radius: 14, hp: 40, damage: 1, friendly: false, dead: false}],
    grid: [], terminal: {cleared: false, dead: false},
  };
}

test('context one-hot is appended to the player input', () => {
  const context = {move: 'up', fire: 'left'};
  const encoded = encodeObservation(observation(), context);
  const base = 8;
  assert.equal(encoded.player.length, base + MOVE_ACTIONS.length + FIRE_ACTIONS.length);
  assert.equal(encoded.player[base + MOVE_ACTIONS.indexOf('up')], 1);
  assert.equal(encoded.player[base + MOVE_ACTIONS.length + FIRE_ACTIONS.indexOf('left')], 1);
  const none = encodeObservation(observation());
  assert.equal(none.player[base], 1, 'absent context means previous move none');
  assert.equal(none.player[base + MOVE_ACTIONS.length], 1, 'absent context means previous fire none');
});

test('the network output depends on the previous action context', () => {
  const policy = new ActorCriticNetwork({seed: 3});
  const plain = policy.forward(observation());
  const withContext = policy.forward(observation(), {context: {move: 'right', fire: 'down'}});
  const difference = plain.moveProbabilities.reduce((sum, value, index) =>
    sum + Math.abs(value - withContext.moveProbabilities[index]), 0);
  assert.ok(difference > 1e-6, 'context must reach the network');
});

test('legacy checkpoints without context are zero-padded and stay context-invariant', () => {
  const modern = new ActorCriticNetwork({seed: 9});
  const payload = JSON.parse(modern.save());
  const layer = payload.layers[0];
  const outputSize = layer.outputSize;
  const base = 8;
  const truncate = values => values.slice(0, base * outputSize);
  payload.layers[0] = {
    inputSize: base,
    outputSize,
    weights: truncate(layer.weights),
    bias: layer.bias,
  };
  payload.optimizer.state[0] = {
    weightsM: truncate(payload.optimizer.state[0].weightsM),
    weightsV: truncate(payload.optimizer.state[0].weightsV),
    biasM: payload.optimizer.state[0].biasM,
    biasV: payload.optimizer.state[0].biasV,
  };
  const restored = ActorCriticNetwork.fromJSON(payload);
  const restoredLayer = restored.layers[0];
  assert.equal(restoredLayer.inputSize, base + MOVE_ACTIONS.length + FIRE_ACTIONS.length);
  for (let index = 0; index < base * outputSize; index++) {
    assert.ok(Math.abs(restoredLayer.weights[index] - layer.weights[index]) < 1e-12, 'legacy weights preserved');
  }
  for (let index = base * outputSize; index < restoredLayer.weights.length; index++) {
    assert.equal(restoredLayer.weights[index], 0, 'new context rows start at zero');
  }
  assert.equal(restored.optimizer.state[0].weightsM.length, restoredLayer.weights.length);
  const plain = restored.forward(observation());
  const withContext = restored.forward(observation(), {context: {move: 'up_left', fire: 'right'}});
  assert.deepEqual(plain.moveProbabilities, withContext.moveProbabilities);
  assert.deepEqual(plain.fireProbabilities, withContext.fireProbabilities);
});

test('invalid context is rejected', () => {
  const policy = new ActorCriticNetwork({seed: 5});
  assert.throws(() => policy.act(observation(), {context: {move: 'teleport', fire: 'none'}}),
    /move context must be one of/);
  assert.throws(() => policy.act(observation(), {context: {move: 'up', fire: 'bomb'}}),
    /fire context must be one of/);
});
