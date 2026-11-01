import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeAction, publicObservation, ROOM_SCENARIOS } from '../src/environment.mjs';

test('movement accepts the eight directions and still allows independent cardinal fire', () => {
  const moves = ['up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left'];
  for (const move of moves) {
    assert.equal(normalizeAction({ move, fire: 'left' }).move, move);
    assert.equal(normalizeAction({ move, fire: 'left' }).fire, 'left');
  }
});

test('the action surface rejects bombs, items, pills, cards, and arbitrary controls', () => {
  for (const action of [
    { move: 'none', bomb: true },
    { move: 'none', item: true },
    { move: 'none', card: true },
    { move: 'none', pill: true },
    { move: 'none', fire: 'bomb' },
    { move: 'teleport' },
  ]) assert.throws(() => normalizeAction(action));
});

test('public observations contain only combat entities and remove memory pointers', () => {
  const result = publicObservation({
    ready: true, state: 'ready', frame: 8, cleared: false, dead: false,
    player: { position: { x: 2, y: 3 }, velocity: { x: 0, y: 0 }, health: 6, address: 123 },
    room: { type: 5, variant: 1010, shape: 1, width: 13, height: 7, pointer: 456 },
    entities: [
      { type: 20, kind: 'npc', active: true, exists: true, dead: false, address: 789 },
      { type: 9, kind: 'projectile', active: true, exists: true, dead: false },
      { type: 4, kind: 'bomb', active: true, exists: true, dead: false },
      { type: 5, kind: 'pickup', active: true, exists: true, dead: false },
    ],
    grid: [{ index: 0, x: 0, y: 0, type: 1, address: 999, collision: 0 }],
  });
  assert.deepEqual(result.entities.map(entity => entity.type), [20, 9]);
  assert.equal('address' in result.player, false);
  assert.equal(result.room.shape, 1);
  assert.equal('pointer' in result.room, false);
  assert.equal('address' in result.grid[0], false);
});

test('the automatic reset catalogue has twelve unique verified boss-room identities', () => {
  const scenarios = Object.values(ROOM_SCENARIOS).filter(scenario => scenario.type === 5);
  assert.equal(scenarios.length, 12);
  assert.equal(new Set(scenarios.map(scenario => scenario.room)).size, 12);
  assert.ok(scenarios.every(scenario => scenario.type === 5 && scenario.targetNPC >= 10));
});

test('the natural Frowning Gaper scene specifies five exact enemies', () => {
  const scene = ROOM_SCENARIOS.frowningGaperGroup;
  assert.equal(scene.room, 'd.841');
  assert.deepEqual([scene.type, scene.variant, scene.shape, scene.width, scene.height,
    scene.targetNPC, scene.targetVariant, scene.targetSubtype, scene.initialCount],
    [1, 841, 1, 15, 9, 10, 0, 0, 5]);
});
