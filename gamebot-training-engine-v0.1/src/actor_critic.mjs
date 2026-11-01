/**
 * A dependency-free, observation-aware Actor-Critic policy for the Gamebot
 * room observation schema.  The actor factorizes movement and firing given
 * the shared state embedding; the critic predicts V(s) only.
 */

export const MOVE_ACTIONS = Object.freeze([
  'none', 'up', 'up_right', 'right', 'down_right',
  'down', 'down_left', 'left', 'up_left',
]);
export const FIRE_ACTIONS = Object.freeze(['none', 'up', 'right', 'down', 'left']);

const GRID_WIDTH = 15;
const GRID_HEIGHT = 9;
const ARENA_WIDTH = 640;
const ARENA_HEIGHT = 560;
const MAX_ENEMIES = 8;
const ENEMY_FEATURES = 9;
const MAX_PROJECTILES = 16;
const PROJECTILE_FEATURES = 8;
const PLAYER_INPUTS = 8;
const ENEMY_INPUTS = 1 + MAX_ENEMIES * ENEMY_FEATURES;
const PROJECTILE_INPUTS = 1 + MAX_PROJECTILES * PROJECTILE_FEATURES;
const TERRAIN_INPUTS = GRID_WIDTH * GRID_HEIGHT * 4 + 4;
const MODEL_FORMAT = 'gamebot-actor-critic/1';

const MOVE_INDEX = new Map(MOVE_ACTIONS.map((action, index) => [action, index]));
const FIRE_INDEX = new Map(FIRE_ACTIONS.map((action, index) => [action, index]));

function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function alive(entity) {
  return entity && entity.dead !== true && entity.exists !== false && entity.active !== false;
}

function distanceSquared(entity, player) {
  const dx = finite(entity.position?.x) - player.x;
  const dy = finite(entity.position?.y) - player.y;
  return dx * dx + dy * dy;
}

function normalizeEntities(entities, player, limit, featureCount, encode) {
  const rows = (Array.isArray(entities) ? entities : [])
    .filter(alive)
    .map(entity => ({ entity, distance: distanceSquared(entity, player) }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, limit);
  const values = new Float64Array(1 + limit * featureCount);
  values[0] = Math.log1p(rows.length) / Math.log1p(limit);
  rows.forEach(({ entity }, index) => {
    const offset = 1 + index * featureCount;
    encode(values, offset, entity, player);
  });
  return values;
}

/** Convert the variable-length game observation into four fixed-size inputs. */
export function encodeObservation(observation) {
  if (!observation || typeof observation !== 'object' || Array.isArray(observation)) {
    throw new TypeError('observation must be an object');
  }

  const playerObservation = observation.player ?? {};
  const player = {
    x: finite(playerObservation.position?.x),
    y: finite(playerObservation.position?.y),
  };
  const playerFeatures = new Float64Array([
    clamp(player.x / ARENA_WIDTH, -1, 2),
    clamp(player.y / ARENA_HEIGHT, -1, 2),
    clamp(finite(playerObservation.velocity?.x) / 10, -3, 3),
    clamp(finite(playerObservation.velocity?.y) / 10, -3, 3),
    clamp(finite(playerObservation.health) / 12, 0, 2),
    clamp(finite(playerObservation.fireDelay) / 30, 0, 4),
    clamp(finite(playerObservation.damageCooldown) / 60, 0, 4),
    playerObservation.controls === true ? 1 : 0,
  ]);

  const entities = Array.isArray(observation.entities) ? observation.entities : [];
  const enemies = entities.filter(entity =>
    finite(entity?.type, -1) >= 10 && finite(entity?.type, -1) < 1000 && entity?.friendly !== true);
  const enemyFeatures = normalizeEntities(enemies, player, MAX_ENEMIES, ENEMY_FEATURES,
    (target, offset, entity, origin) => {
      target[offset] = clamp((finite(entity.position?.x) - origin.x) / ARENA_WIDTH, -2, 2);
      target[offset + 1] = clamp((finite(entity.position?.y) - origin.y) / ARENA_HEIGHT, -2, 2);
      target[offset + 2] = clamp(finite(entity.velocity?.x) / 10, -3, 3);
      target[offset + 3] = clamp(finite(entity.velocity?.y) / 10, -3, 3);
      target[offset + 4] = clamp(Math.log1p(Math.max(0, finite(entity.hp))) / Math.log1p(1000), 0, 1.5);
      target[offset + 5] = clamp(finite(entity.radius) / 80, 0, 2);
      target[offset + 6] = clamp(finite(entity.damage) / 10, 0, 2);
      target[offset + 7] = entity.invulnerable === true ? 1 : 0;
      target[offset + 8] = entity.targetable === false ? 0 : 1;
    });

  const projectiles = entities.filter(entity =>
    finite(entity?.type, -1) === 2 || finite(entity?.type, -1) === 9 || entity?.kind === 'projectile');
  const projectileFeatures = normalizeEntities(projectiles, player, MAX_PROJECTILES, PROJECTILE_FEATURES,
    (target, offset, entity, origin) => {
      target[offset] = clamp((finite(entity.position?.x) - origin.x) / ARENA_WIDTH, -2, 2);
      target[offset + 1] = clamp((finite(entity.position?.y) - origin.y) / ARENA_HEIGHT, -2, 2);
      target[offset + 2] = clamp(finite(entity.velocity?.x) / 10, -3, 3);
      target[offset + 3] = clamp(finite(entity.velocity?.y) / 10, -3, 3);
      target[offset + 4] = clamp(finite(entity.radius) / 40, 0, 2);
      target[offset + 5] = clamp(finite(entity.damage) / 10, 0, 2);
      target[offset + 6] = entity.friendly === true ? 1 : 0;
      target[offset + 7] = clamp(finite(entity.collision) / 4, 0, 1);
    });

  const room = observation.room ?? {};
  const width = Math.max(1, finite(room.width, GRID_WIDTH));
  const height = Math.max(1, finite(room.height, GRID_HEIGHT));
  const terrainFeatures = new Float64Array(TERRAIN_INPUTS);
  for (const cell of Array.isArray(observation.grid) ? observation.grid : []) {
    const x = clamp(Math.floor(finite(cell.x) * GRID_WIDTH / width), 0, GRID_WIDTH - 1);
    const y = clamp(Math.floor(finite(cell.y) * GRID_HEIGHT / height), 0, GRID_HEIGHT - 1);
    const offset = (y * GRID_WIDTH + x) * 4;
    terrainFeatures[offset] = cell.type == null ? 0 : 1;
    terrainFeatures[offset + 1] = finite(cell.collision) > 0 ? 1 : 0;
    terrainFeatures[offset + 2] = cell.danger === true ? 1 : 0;
    terrainFeatures[offset + 3] = cell.destructible === true ? 1 : 0;
  }
  terrainFeatures[TERRAIN_INPUTS - 4] = clamp(width / GRID_WIDTH, 0, 2);
  terrainFeatures[TERRAIN_INPUTS - 3] = clamp(height / GRID_HEIGHT, 0, 2);
  terrainFeatures[TERRAIN_INPUTS - 2] = room.cleared === true || observation.terminal?.cleared === true ? 1 : 0;
  terrainFeatures[TERRAIN_INPUTS - 1] = observation.terminal?.dead === true ? 1 : 0;

  return { player: playerFeatures, enemies: enemyFeatures, projectiles: projectileFeatures, terrain: terrainFeatures };
}

function makeRng(seed) {
  let state = (Number(seed) >>> 0) || 0x6d2b79f5;
  return {
    get state() { return state >>> 0; },
    set state(value) { state = (Number(value) >>> 0) || 0x6d2b79f5; },
    next() {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      return (state >>> 0) / 0x100000000;
    },
  };
}

function makeLayer(inputSize, outputSize, rng) {
  const scale = Math.sqrt(6 / (inputSize + outputSize));
  const weights = new Float64Array(inputSize * outputSize);
  for (let index = 0; index < weights.length; index++) weights[index] = (rng.next() * 2 - 1) * scale;
  return { inputSize, outputSize, weights, bias: new Float64Array(outputSize) };
}

function forwardLayer(layer, input, activation = 'tanh') {
  const output = new Float64Array(layer.outputSize);
  for (let column = 0; column < layer.outputSize; column++) {
    let sum = layer.bias[column];
    for (let row = 0; row < layer.inputSize; row++) {
      sum += input[row] * layer.weights[row * layer.outputSize + column];
    }
    output[column] = activation === 'tanh' ? Math.tanh(sum) : sum;
  }
  return output;
}

function zeroGradients(layers) {
  return layers.map(layer => ({
    weights: new Float64Array(layer.weights.length),
    bias: new Float64Array(layer.bias.length),
  }));
}

function backwardLayer(layer, input, output, outputGradient, gradient, activation = 'tanh') {
  const inputGradient = new Float64Array(layer.inputSize);
  for (let column = 0; column < layer.outputSize; column++) {
    const derivative = activation === 'tanh' ? 1 - output[column] * output[column] : 1;
    const delta = outputGradient[column] * derivative;
    gradient.bias[column] += delta;
    for (let row = 0; row < layer.inputSize; row++) {
      const weightIndex = row * layer.outputSize + column;
      gradient.weights[weightIndex] += input[row] * delta;
      inputGradient[row] += layer.weights[weightIndex] * delta;
    }
  }
  return inputGradient;
}

function softmax(logits) {
  let maximum = -Infinity;
  for (const value of logits) maximum = Math.max(maximum, value);
  const probabilities = new Float64Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index++) {
    probabilities[index] = Math.exp(logits[index] - maximum);
    total += probabilities[index];
  }
  for (let index = 0; index < probabilities.length; index++) probabilities[index] /= total;
  return probabilities;
}

function categoricalStats(probabilities) {
  let entropy = 0;
  const logProbabilities = new Float64Array(probabilities.length);
  for (let index = 0; index < probabilities.length; index++) {
    logProbabilities[index] = Math.log(Math.max(probabilities[index], Number.MIN_VALUE));
    entropy -= probabilities[index] * logProbabilities[index];
  }
  return { entropy, logProbabilities };
}

function resolveActionIndex(action, actions, lookup, name) {
  const index = typeof action === 'number' ? action : lookup.get(String(action));
  if (!Number.isInteger(index) || index < 0 || index >= actions.length) {
    throw new TypeError(`${name} must be one of: ${actions.join(', ')}`);
  }
  return index;
}

function makeGradVector(probabilities, chosenIndex, policyScale, entropyScale) {
  const result = new Float64Array(probabilities.length);
  let entropy = 0;
  for (let index = 0; index < probabilities.length; index++) {
    entropy -= probabilities[index] * Math.log(Math.max(probabilities[index], Number.MIN_VALUE));
  }
  for (let index = 0; index < probabilities.length; index++) {
    const logProbability = Math.log(Math.max(probabilities[index], Number.MIN_VALUE));
    result[index] = policyScale * ((index === chosenIndex ? 1 : 0) - probabilities[index])
      + entropyScale * probabilities[index] * (logProbability + entropy);
  }
  return result;
}

function addVectors(target, source) {
  for (let index = 0; index < target.length; index++) target[index] += source[index];
}

function adamState(layers) {
  return layers.map(layer => ({
    weightsM: new Float64Array(layer.weights.length),
    weightsV: new Float64Array(layer.weights.length),
    biasM: new Float64Array(layer.bias.length),
    biasV: new Float64Array(layer.bias.length),
  }));
}

function finiteOption(value, fallback, name, minimum = 0) {
  const number = value === undefined ? fallback : Number(value);
  if (!Number.isFinite(number) || number < minimum) throw new RangeError(`${name} must be a finite number >= ${minimum}`);
  return number;
}

/** Shared-encoder categorical Actor-Critic with a clipped PPO minibatch update. */
export class ActorCriticNetwork {
  constructor({ seed = 1 } = {}) {
    this.rng = makeRng(seed);
    this.layers = [
      makeLayer(PLAYER_INPUTS, 24, this.rng),
      makeLayer(ENEMY_INPUTS, 48, this.rng),
      makeLayer(PROJECTILE_INPUTS, 48, this.rng),
      makeLayer(TERRAIN_INPUTS, 48, this.rng),
      makeLayer(168, 128, this.rng),
      makeLayer(128, 96, this.rng),
      makeLayer(96, MOVE_ACTIONS.length, this.rng),
      makeLayer(96, FIRE_ACTIONS.length, this.rng),
      makeLayer(96, 1, this.rng),
    ];
    this.optimizer = { step: 0, state: adamState(this.layers) };
  }

  static get architecture() {
    return Object.freeze({
      inputs: Object.freeze({ player: PLAYER_INPUTS, enemies: ENEMY_INPUTS, projectiles: PROJECTILE_INPUTS, terrain: TERRAIN_INPUTS }),
      featureEncoders: Object.freeze({ player: 24, enemies: 48, projectiles: 48, terrain: 48 }),
      sharedEncoder: Object.freeze([128, 96]),
      actor: Object.freeze({ move: MOVE_ACTIONS.length, fire: FIRE_ACTIONS.length }),
      critic: 'V(s)',
    });
  }

  #forward(observation) {
    const encoded = encodeObservation(observation);
    const inputs = [encoded.player, encoded.enemies, encoded.projectiles, encoded.terrain];
    const embeddings = inputs.map((input, index) => forwardLayer(this.layers[index], input));
    const merged = new Float64Array(168);
    let offset = 0;
    for (const embedding of embeddings) {
      merged.set(embedding, offset);
      offset += embedding.length;
    }
    const sharedOne = forwardLayer(this.layers[4], merged);
    const sharedTwo = forwardLayer(this.layers[5], sharedOne);
    const moveLogits = forwardLayer(this.layers[6], sharedTwo, 'linear');
    const fireLogits = forwardLayer(this.layers[7], sharedTwo, 'linear');
    const valueOutput = forwardLayer(this.layers[8], sharedTwo, 'linear');
    return {
      inputs, embeddings, merged, sharedOne, sharedTwo,
      moveLogits, fireLogits, value: valueOutput[0],
      moveProbabilities: softmax(moveLogits), fireProbabilities: softmax(fireLogits),
    };
  }

  /** Return both independent action distributions and the state value V(s). */
  forward(observation) {
    const result = this.#forward(observation);
    return {
      moveLogits: Array.from(result.moveLogits),
      moveProbabilities: Array.from(result.moveProbabilities),
      fireLogits: Array.from(result.fireLogits),
      fireProbabilities: Array.from(result.fireProbabilities),
      value: result.value,
    };
  }

  #distributionResult(result, moveIndex, fireIndex) {
    const moveStats = categoricalStats(result.moveProbabilities);
    const fireStats = categoricalStats(result.fireProbabilities);
    const moveLogProb = moveStats.logProbabilities[moveIndex];
    const fireLogProb = fireStats.logProbabilities[fireIndex];
    return {
      action: { move: MOVE_ACTIONS[moveIndex], fire: FIRE_ACTIONS[fireIndex] },
      moveIndex,
      fireIndex,
      moveLogProb,
      fireLogProb,
      logProb: moveLogProb + fireLogProb,
      entropy: moveStats.entropy + fireStats.entropy,
      moveEntropy: moveStats.entropy,
      fireEntropy: fireStats.entropy,
      value: result.value,
      moveProbabilities: Array.from(result.moveProbabilities),
      fireProbabilities: Array.from(result.fireProbabilities),
    };
  }

  /** Sample each head independently; both selected controls are executed together. */
  act(observation, { deterministic = false } = {}) {
    const result = this.#forward(observation);
    const select = probabilities => {
      if (deterministic) {
        let best = 0;
        for (let index = 1; index < probabilities.length; index++) {
          if (probabilities[index] > probabilities[best]) best = index;
        }
        return best;
      }
      const draw = this.rng.next();
      let cumulative = 0;
      for (let index = 0; index < probabilities.length; index++) {
        cumulative += probabilities[index];
        if (draw < cumulative) return index;
      }
      return probabilities.length - 1;
    };
    return this.#distributionResult(result, select(result.moveProbabilities), select(result.fireProbabilities));
  }

  /** Score a supplied pair of actions under the factorized joint policy. */
  evaluateActions(observation, action) {
    if (!action || typeof action !== 'object') throw new TypeError('action must contain move and fire');
    const moveIndex = resolveActionIndex(action.move, MOVE_ACTIONS, MOVE_INDEX, 'move');
    const fireIndex = resolveActionIndex(action.fire, FIRE_ACTIONS, FIRE_INDEX, 'fire');
    return this.#distributionResult(this.#forward(observation), moveIndex, fireIndex);
  }

  #accumulateSampleGradients(cache, moveGradient, fireGradient, valueGradient, gradients) {
    const moveSharedGradient = backwardLayer(
      this.layers[6], cache.sharedTwo, cache.moveLogits, moveGradient, gradients[6], 'linear');
    const fireSharedGradient = backwardLayer(
      this.layers[7], cache.sharedTwo, cache.fireLogits, fireGradient, gradients[7], 'linear');
    const valueOutput = new Float64Array([cache.value]);
    const valueSharedGradient = backwardLayer(
      this.layers[8], cache.sharedTwo, valueOutput, new Float64Array([valueGradient]), gradients[8], 'linear');
    const sharedGradient = new Float64Array(96);
    addVectors(sharedGradient, moveSharedGradient);
    addVectors(sharedGradient, fireSharedGradient);
    addVectors(sharedGradient, valueSharedGradient);

    const sharedOneGradient = backwardLayer(
      this.layers[5], cache.sharedOne, cache.sharedTwo, sharedGradient, gradients[5]);
    const mergedGradient = backwardLayer(
      this.layers[4], cache.merged, cache.sharedOne, sharedOneGradient, gradients[4]);

    let embeddingOffset = 0;
    for (let index = 0; index < 4; index++) {
      const embeddingGradient = mergedGradient.slice(embeddingOffset, embeddingOffset + cache.embeddings[index].length);
      backwardLayer(this.layers[index], cache.inputs[index], cache.embeddings[index], embeddingGradient, gradients[index]);
      embeddingOffset += cache.embeddings[index].length;
    }
  }

  #applyAdam(gradients, { learningRate, beta1, beta2, epsilon, maxGradNorm }) {
    let normSquared = 0;
    for (const gradient of gradients) {
      for (const value of gradient.weights) normSquared += value * value;
      for (const value of gradient.bias) normSquared += value * value;
    }
    const norm = Math.sqrt(normSquared);
    const scale = maxGradNorm > 0 && norm > maxGradNorm ? maxGradNorm / norm : 1;
    this.optimizer.step++;
    const step = this.optimizer.step;
    const correction1 = 1 - beta1 ** step;
    const correction2 = 1 - beta2 ** step;

    for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex++) {
      const layer = this.layers[layerIndex];
      const gradient = gradients[layerIndex];
      const state = this.optimizer.state[layerIndex];
      for (let index = 0; index < layer.weights.length; index++) {
        const g = gradient.weights[index] * scale;
        state.weightsM[index] = beta1 * state.weightsM[index] + (1 - beta1) * g;
        state.weightsV[index] = beta2 * state.weightsV[index] + (1 - beta2) * g * g;
        const m = state.weightsM[index] / correction1;
        const v = state.weightsV[index] / correction2;
        layer.weights[index] -= learningRate * m / (Math.sqrt(v) + epsilon);
      }
      for (let index = 0; index < layer.bias.length; index++) {
        const g = gradient.bias[index] * scale;
        state.biasM[index] = beta1 * state.biasM[index] + (1 - beta1) * g;
        state.biasV[index] = beta2 * state.biasV[index] + (1 - beta2) * g * g;
        const m = state.biasM[index] / correction1;
        const v = state.biasV[index] / correction2;
        layer.bias[index] -= learningRate * m / (Math.sqrt(v) + epsilon);
      }
    }
    return norm;
  }

  /**
   * Apply PPO updates to rollout rows of the form:
   * { observation, action: { move, fire }, oldLogProb, advantage, return }.
   */
  updatePPO(transitions, options = {}) {
    if (!Array.isArray(transitions) || transitions.length === 0) {
      throw new TypeError('transitions must be a non-empty array');
    }
    const epochs = options.epochs ?? 4;
    const batchSize = options.batchSize ?? 256;
    if (!Number.isInteger(epochs) || epochs < 1) throw new RangeError('epochs must be a positive integer');
    if (!Number.isInteger(batchSize) || batchSize < 1) throw new RangeError('batchSize must be a positive integer');
    const clipRange = finiteOption(options.clipRange, 0.2, 'clipRange');
    const valueCoefficient = finiteOption(options.valueCoefficient, 0.5, 'valueCoefficient');
    const entropyCoefficient = finiteOption(options.entropyCoefficient, 0.005, 'entropyCoefficient');
    const learningRate = finiteOption(options.learningRate, 3e-4, 'learningRate');
    const maxGradNorm = finiteOption(options.maxGradNorm, 0.5, 'maxGradNorm');
    const beta1 = finiteOption(options.beta1, 0.9, 'beta1');
    const beta2 = finiteOption(options.beta2, 0.999, 'beta2');
    const epsilon = finiteOption(options.epsilon, 1e-8, 'epsilon');
    if (beta1 >= 1 || beta2 >= 1) throw new RangeError('beta1 and beta2 must be less than 1');
    if (clipRange >= 1) throw new RangeError('clipRange must be less than 1');

    const rows = transitions.map((transition, index) => {
      const action = transition?.action ?? transition;
      const oldLogProb = finite(transition?.oldLogProb ?? transition?.oldLogProbability, NaN);
      const advantage = finite(transition?.advantage, NaN);
      const targetValue = finite(transition?.return ?? transition?.targetValue, NaN);
      if (!Number.isFinite(oldLogProb) || !Number.isFinite(advantage) || !Number.isFinite(targetValue)) {
        throw new TypeError(`transition ${index} needs finite oldLogProb, advantage, and return`);
      }
      return {
        observation: transition.observation,
        moveIndex: resolveActionIndex(action?.move, MOVE_ACTIONS, MOVE_INDEX, `transition ${index} move`),
        fireIndex: resolveActionIndex(action?.fire, FIRE_ACTIONS, FIRE_INDEX, `transition ${index} fire`),
        oldLogProb,
        advantage,
        targetValue,
      };
    });

    if (options.normalizeAdvantages !== false && rows.length > 1) {
      const mean = rows.reduce((sum, row) => sum + row.advantage, 0) / rows.length;
      const variance = rows.reduce((sum, row) => sum + (row.advantage - mean) ** 2, 0) / rows.length;
      const standardDeviation = Math.sqrt(variance);
      if (standardDeviation > 1e-8) {
        for (const row of rows) row.advantage = (row.advantage - mean) / standardDeviation;
      }
    }

    const metrics = { policyLoss: 0, valueLoss: 0, entropy: 0, approximateKl: 0, clipFraction: 0, gradientNorm: 0, updates: 0 };
    for (let epoch = 0; epoch < epochs; epoch++) {
      const order = Array.from({ length: rows.length }, (_, index) => index);
      for (let index = order.length - 1; index > 0; index--) {
        const swap = Math.floor(this.rng.next() * (index + 1));
        [order[index], order[swap]] = [order[swap], order[index]];
      }

      for (let start = 0; start < order.length; start += batchSize) {
        const batch = order.slice(start, start + batchSize).map(index => rows[index]);
        const gradients = zeroGradients(this.layers);
        const inverseBatchSize = 1 / batch.length;
        for (const row of batch) {
          const cache = this.#forward(row.observation);
          const moveStats = categoricalStats(cache.moveProbabilities);
          const fireStats = categoricalStats(cache.fireProbabilities);
          const newLogProb = moveStats.logProbabilities[row.moveIndex] + fireStats.logProbabilities[row.fireIndex];
          const logRatio = clamp(newLogProb - row.oldLogProb, -20, 20);
          const ratio = Math.exp(logRatio);
          const clippedRatio = clamp(ratio, 1 - clipRange, 1 + clipRange);
          const unclippedObjective = ratio * row.advantage;
          const clippedObjective = clippedRatio * row.advantage;
          const policyLoss = -Math.min(unclippedObjective, clippedObjective);
          const clipped = clippedObjective < unclippedObjective;
          const policyScale = clipped ? 0 : -row.advantage * ratio * inverseBatchSize;
          const entropyScale = entropyCoefficient * inverseBatchSize;
          const moveGradient = makeGradVector(
            cache.moveProbabilities, row.moveIndex, policyScale, entropyScale);
          const fireGradient = makeGradVector(
            cache.fireProbabilities, row.fireIndex, policyScale, entropyScale);
          const valueDifference = cache.value - row.targetValue;
          const valueLoss = 0.5 * valueCoefficient * valueDifference * valueDifference;
          const valueGradient = valueCoefficient * valueDifference * inverseBatchSize;
          this.#accumulateSampleGradients(
            cache, moveGradient, fireGradient, valueGradient, gradients);

          metrics.policyLoss += policyLoss / (rows.length * epochs);
          metrics.valueLoss += valueLoss / (rows.length * epochs);
          metrics.entropy += (moveStats.entropy + fireStats.entropy) / (rows.length * epochs);
          metrics.approximateKl += (row.oldLogProb - newLogProb) / (rows.length * epochs);
          metrics.clipFraction += (clipped ? 1 : 0) / (rows.length * epochs);
        }
        metrics.gradientNorm += this.#applyAdam(gradients, {
          learningRate, beta1, beta2, epsilon, maxGradNorm,
        });
        metrics.updates++;
      }
    }
    metrics.gradientNorm /= metrics.updates;
    return metrics;
  }

  toJSON() {
    return {
      format: MODEL_FORMAT,
      rngState: this.rng.state,
      layers: this.layers.map(layer => ({
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        weights: Array.from(layer.weights),
        bias: Array.from(layer.bias),
      })),
      optimizer: {
        step: this.optimizer.step,
        state: this.optimizer.state.map(state => ({
          weightsM: Array.from(state.weightsM),
          weightsV: Array.from(state.weightsV),
          biasM: Array.from(state.biasM),
          biasV: Array.from(state.biasV),
        })),
      },
    };
  }

  save() {
    return JSON.stringify(this.toJSON());
  }

  static fromJSON(serialized) {
    const payload = typeof serialized === 'string' ? JSON.parse(serialized) : serialized;
    if (!payload || payload.format !== MODEL_FORMAT || !Array.isArray(payload.layers)) {
      throw new TypeError(`model checkpoint must use format ${MODEL_FORMAT}`);
    }
    const network = new ActorCriticNetwork({ seed: payload.rngState });
    if (payload.layers.length !== network.layers.length) throw new TypeError('checkpoint layer count does not match architecture');
    for (let index = 0; index < network.layers.length; index++) {
      const saved = payload.layers[index];
      const layer = network.layers[index];
      if (saved.inputSize !== layer.inputSize || saved.outputSize !== layer.outputSize
        || saved.weights?.length !== layer.weights.length || saved.bias?.length !== layer.bias.length) {
        throw new TypeError(`checkpoint layer ${index} shape does not match architecture`);
      }
      layer.weights.set(saved.weights);
      layer.bias.set(saved.bias);
    }
    network.rng.state = payload.rngState;
    if (Number.isInteger(payload.optimizer?.step) && Array.isArray(payload.optimizer?.state)
      && payload.optimizer.state.length === network.layers.length) {
      network.optimizer.step = payload.optimizer.step;
      for (let index = 0; index < network.layers.length; index++) {
        const saved = payload.optimizer.state[index];
        const state = network.optimizer.state[index];
        for (const key of ['weightsM', 'weightsV', 'biasM', 'biasV']) {
          if (saved[key]?.length !== state[key].length) throw new TypeError(`checkpoint optimizer state ${index}.${key} has an invalid shape`);
          state[key].set(saved[key]);
        }
      }
    }
    return network;
  }
}

