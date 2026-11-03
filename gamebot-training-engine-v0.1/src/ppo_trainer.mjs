import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { ActorCriticNetwork } from './actor_critic.mjs';

const CHECKPOINT_FORMAT = 'gamebot-ppo-checkpoint/1';

function finite(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new TypeError(`${name} must be finite`);
  return number;
}

function validateProbability(value, name) {
  const number = finite(value, name);
  if (number < 0 || number > 1) throw new RangeError(`${name} must be between 0 and 1`);
  return number;
}

function cloneObservation(observation) {
  if (!observation || typeof observation !== 'object' || Array.isArray(observation)) {
    throw new TypeError('rollout observation must be an object');
  }
  return structuredClone(observation);
}

/** A bounded on-policy buffer for state/action/reward/value rollout rows. */
export class RolloutBuffer {
  constructor({ capacity = Infinity } = {}) {
    if (capacity !== Infinity && (!Number.isInteger(capacity) || capacity < 1)) {
      throw new RangeError('capacity must be a positive integer or Infinity');
    }
    this.capacity = capacity;
    this.rows = [];
  }

  get size() {
    return this.rows.length;
  }

  add(transition) {
    if (this.rows.length >= this.capacity) throw new RangeError('rollout buffer is full');
    const action = transition?.action;
    if (!action || typeof action !== 'object' || typeof action.move !== 'string' || typeof action.fire !== 'string') {
      throw new TypeError('transition action must contain move and fire strings');
    }
    const row = {
      observation: cloneObservation(transition.observation),
      action: { move: action.move, fire: action.fire },
      reward: finite(transition.reward, 'reward'),
      oldLogProb: finite(transition.oldLogProb, 'oldLogProb'),
      value: finite(transition.value, 'value'),
      nextValue: finite(transition.nextValue, 'nextValue'),
      terminated: transition.terminated === true,
      truncated: transition.truncated === true,
    };
    this.rows.push(row);
    return this.rows.length;
  }

  /** Compute GAE without crossing episode ends; truncations still bootstrap V(s'). */
  computeGAE({ gamma = 0.999, gaeLambda = 0.95 } = {}) {
    return computeGAE(this.rows, { gamma, gaeLambda });
  }

  clear() {
    this.rows.length = 0;
  }

  toJSON() {
    return { capacity: this.capacity, rows: structuredClone(this.rows) };
  }
}

/**
 * GAE over a sequential rollout. nextValue is always stored explicitly so the
 * last row of a rollout and time-limit truncations can bootstrap correctly.
 */
export function computeGAE(transitions, { gamma = 0.999, gaeLambda = 0.95 } = {}) {
  if (!Array.isArray(transitions)) throw new TypeError('transitions must be an array');
  gamma = validateProbability(gamma, 'gamma');
  gaeLambda = validateProbability(gaeLambda, 'gaeLambda');

  const result = new Array(transitions.length);
  let nextAdvantage = 0;
  for (let index = transitions.length - 1; index >= 0; index--) {
    const row = transitions[index];
    const reward = finite(row.reward, `transition ${index} reward`);
    const value = finite(row.value, `transition ${index} value`);
    const nextValue = finite(row.nextValue, `transition ${index} nextValue`);
    const terminated = row.terminated === true;
    const truncated = row.truncated === true;

    // A true terminal state has no bootstrap value. A time limit does.
    const bootstrap = terminated ? 0 : nextValue;
    const delta = reward + gamma * bootstrap - value;
    const traceContinuation = terminated || truncated ? 0 : 1;
    const advantage = delta + gamma * gaeLambda * traceContinuation * nextAdvantage;
    result[index] = {
      observation: row.observation,
      action: row.action,
      oldLogProb: finite(row.oldLogProb, `transition ${index} oldLogProb`),
      value,
      nextValue,
      reward,
      terminated,
      truncated,
      advantage,
      return: advantage + value,
    };
    nextAdvantage = advantage;
  }
  return result;
}

function newEpisodeStats() {
  return {
    reward: 0,
    damageDealt: 0,
    damageTaken: 0,
    decisions: 0,
    frames: 0,
    firingDecisions: 0,
    damageBearingFireDecisions: 0,
  };
}

async function atomicWriteJson(path, value) {
  const target = resolve(path);
  await mkdir(dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, `${JSON.stringify(value)}\n`, 'utf8');
    await rename(temporary, target);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

const DEFAULT_CONFIG = Object.freeze({
  learningRate: 3e-4,
  gamma: 0.999,
  gaeLambda: 0.95,
  clipRange: 0.2,
  updateEpochs: 4,
  minibatchSize: 256,
  entropyCoefficient: 0.005,
  valueCoefficient: 0.5,
  maxGradNorm: 0.5,
  rolloutSteps: 1024,
});

/** Single-environment PPO loop around GamebotTrainingEnvironment. */
export class PPOTrainer {
  constructor({
    environment,
    policy = new ActorCriticNetwork(),
    checkpointDirectory = 'runs/ppo-checkpoints',
    resetOptions = { scene: 'auto' },
    seedSchedule = [],
    config = {},
  } = {}) {
    if (!environment || typeof environment.reset !== 'function' || typeof environment.step !== 'function') {
      throw new TypeError('environment must implement reset() and step()');
    }
    if (!(policy instanceof ActorCriticNetwork)) throw new TypeError('policy must be an ActorCriticNetwork');
    this.environment = environment;
    this.policy = policy;
    this.checkpointDirectory = resolve(checkpointDirectory);
    this.latestCheckpointPath = join(this.checkpointDirectory, 'latest-training.json');
    this.bestPolicyPath = join(this.checkpointDirectory, 'best-policy.json');
    this.resetOptions = { ...resetOptions };
    if (!Array.isArray(seedSchedule) || seedSchedule.some(seed => typeof seed !== 'string')) {
      throw new TypeError('seedSchedule must be an array of seed strings');
    }
    this.seedSchedule = [...seedSchedule];
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.#validateConfig();
    this.buffer = new RolloutBuffer({ capacity: this.config.rolloutSteps });
    this.observation = null;
    this.iteration = 0;
    this.globalSteps = 0;
    this.globalFrames = 0;
    this.completedEpisodes = 0;
    this.roomsCleared = 0;
    this.deaths = 0;
    this.bestMeanEpisodeReward = null;
    this.episodeReturn = 0;
    this.episodeStats = null;
    this.lastAction = null;
  }

  #validateConfig() {
    const positive = ['learningRate', 'clipRange', 'entropyCoefficient', 'valueCoefficient', 'maxGradNorm'];
    for (const key of positive) {
      const value = finite(this.config[key], key);
      if (value < 0) throw new RangeError(`${key} must be non-negative`);
      this.config[key] = value;
    }
    if (this.config.clipRange >= 1) throw new RangeError('clipRange must be less than 1');
    this.config.gamma = validateProbability(this.config.gamma, 'gamma');
    this.config.gaeLambda = validateProbability(this.config.gaeLambda, 'gaeLambda');
    for (const key of ['updateEpochs', 'minibatchSize', 'rolloutSteps']) {
      if (!Number.isInteger(this.config[key]) || this.config[key] < 1) {
        throw new RangeError(`${key} must be a positive integer`);
      }
    }
  }

  async #ensureObservation() {
    if (!this.observation) {
      this.observation = await this.environment.reset(this.#episodeResetOptions());
      this.episodeStats = newEpisodeStats();
      this.lastAction = null;
    }
  }

  #episodeResetOptions() {
    const options = { ...this.resetOptions };
    if (this.seedSchedule.length) {
      options.seed = this.seedSchedule[this.completedEpisodes % this.seedSchedule.length];
    }
    return options;
  }

  /** Gather exactly rolloutSteps decisions, resetting after terminations/time limits. */
  async collectRollout() {
    await this.#ensureObservation();
    const episodeRewards = [];
    const episodeSummaries = [];
    let rewardSum = 0;
    let terminatedCount = 0;
    let truncatedCount = 0;
    let framesAdvanced = 0;
    let damageDealt = 0;
    let damageTaken = 0;
    let firingDecisions = 0;
    let damageBearingFireDecisions = 0;

    for (let index = 0; index < this.config.rolloutSteps; index++) {
      const observation = this.observation;
      const context = this.lastAction;
      const decision = this.policy.act(observation, { context });
      const result = await this.environment.step(decision.action);
      const terminated = result.terminated === true;
      const truncated = result.truncated === true;
      // Bootstrapping is disabled only for a real terminal state. Time-limit
      // truncation still uses V(last observation) in the GAE TD residual.
      const nextValue = terminated ? 0
        : this.policy.forward(result.observation, { context: decision.action }).value;
      const reward = finite(result.reward, 'environment reward');
      const dealt = finite(result.info?.enemyDamage ?? 0, 'enemyDamage');
      const taken = finite(result.info?.healthUnitLost ?? 0, 'healthUnitLost');
      const frames = finite(result.info?.framesAdvanced ?? 0, 'framesAdvanced');
      const fired = decision.action.fire !== 'none';

      this.buffer.add({
        observation,
        action: decision.action,
        reward,
        oldLogProb: decision.logProb,
        value: decision.value,
        nextValue,
        terminated,
        truncated,
      });
      this.observation = result.observation;
      this.lastAction = decision.action;
      this.episodeReturn += reward;
      this.episodeStats.reward += reward;
      this.episodeStats.damageDealt += dealt;
      this.episodeStats.damageTaken += taken;
      this.episodeStats.decisions++;
      this.episodeStats.frames += frames;
      if (fired) this.episodeStats.firingDecisions++;
      if (fired && dealt > 0) this.episodeStats.damageBearingFireDecisions++;
      rewardSum += reward;
      damageDealt += dealt;
      damageTaken += taken;
      if (fired) firingDecisions++;
      if (fired && dealt > 0) damageBearingFireDecisions++;
      framesAdvanced += frames;
      this.globalSteps++;
      this.globalFrames += frames;

      if (terminated || truncated) {
        if (terminated) terminatedCount++;
        if (truncated) truncatedCount++;
        episodeRewards.push(this.episodeReturn);
        this.completedEpisodes++;
        const cleared = result.observation.terminal?.cleared === true;
        const dead = result.observation.terminal?.dead === true;
        if (cleared) this.roomsCleared++;
        if (dead) this.deaths++;
        const episodeSummary = {
          reward: this.episodeStats.reward,
          cleared,
          dead,
          terminated,
          truncated,
          decisions: this.episodeStats.decisions,
          frames: this.episodeStats.frames,
          clearTimeFrames: cleared ? this.episodeStats.frames : null,
          damageDealt: this.episodeStats.damageDealt,
          damageTaken: this.episodeStats.damageTaken,
          firingDecisions: this.episodeStats.firingDecisions,
          damageBearingFireDecisions: this.episodeStats.damageBearingFireDecisions,
          damageBearingFireDecisionRate: this.episodeStats.firingDecisions
            ? this.episodeStats.damageBearingFireDecisions / this.episodeStats.firingDecisions : null,
        };
        episodeSummaries.push(episodeSummary);
        this.episodeReturn = 0;
        this.episodeStats = null;
        this.observation = await this.environment.reset(this.#episodeResetOptions());
        this.episodeStats = newEpisodeStats();
        this.lastAction = null;
      }
    }

    return {
      decisions: this.buffer.size,
      rewardSum,
      meanRewardPerDecision: rewardSum / this.buffer.size,
      damageDealt,
      damageTaken,
      firingDecisions,
      damageBearingFireDecisions,
      damageBearingFireDecisionRate: firingDecisions ? damageBearingFireDecisions / firingDecisions : null,
      framesAdvanced,
      terminatedCount,
      truncatedCount,
      episodeRewards,
      episodeSummaries,
      meanEpisodeReward: episodeRewards.length
        ? episodeRewards.reduce((sum, value) => sum + value, 0) / episodeRewards.length
        : null,
      roomClearRate: episodeSummaries.length
        ? episodeSummaries.filter(episode => episode.cleared).length / episodeSummaries.length
        : null,
      cumulativeRoomClearRate: this.completedEpisodes ? this.roomsCleared / this.completedEpisodes : null,
    };
  }

  async trainIteration() {
    const rollout = await this.collectRollout();
    // Rewards in this rollout were generated by these weights; keep that exact
    // policy as the candidate for best-policy selection before applying PPO.
    const rolloutPolicy = rollout.meanEpisodeReward === null ? null : this.policy.toJSON();
    const transitions = this.buffer.computeGAE({
      gamma: this.config.gamma,
      gaeLambda: this.config.gaeLambda,
    });
    const update = this.policy.updatePPO(transitions, {
      learningRate: this.config.learningRate,
      clipRange: this.config.clipRange,
      epochs: this.config.updateEpochs,
      batchSize: this.config.minibatchSize,
      entropyCoefficient: this.config.entropyCoefficient,
      valueCoefficient: this.config.valueCoefficient,
      maxGradNorm: this.config.maxGradNorm,
    });
    const returnMean = transitions.reduce((sum, row) => sum + row.return, 0) / transitions.length;
    const valueMean = transitions.reduce((sum, row) => sum + row.value, 0) / transitions.length;
    const returnVariance = transitions.reduce((sum, row) => sum + (row.return - returnMean) ** 2, 0) / transitions.length;
    const residualVariance = transitions.reduce((sum, row) =>
      sum + ((row.return - row.value) - (returnMean - valueMean)) ** 2, 0)
      / transitions.length;
    update.explainedVariance = returnVariance > 1e-12 ? 1 - residualVariance / returnVariance : null;
    this.buffer.clear();
    this.iteration++;

    const isBest = rollout.meanEpisodeReward !== null
      && (this.bestMeanEpisodeReward === null || rollout.meanEpisodeReward > this.bestMeanEpisodeReward);
    if (isBest) this.bestMeanEpisodeReward = rollout.meanEpisodeReward;
    await this.saveCheckpoint({
      saveBestPolicy: isBest,
      bestScore: rollout.meanEpisodeReward,
      bestPolicyModel: isBest ? rolloutPolicy : null,
    });

    return {
      iteration: this.iteration,
      globalSteps: this.globalSteps,
      globalFrames: this.globalFrames,
      completedEpisodes: this.completedEpisodes,
      roomsCleared: this.roomsCleared,
      deaths: this.deaths,
      bestMeanEpisodeReward: this.bestMeanEpisodeReward,
      rollout,
      update,
      checkpoint: this.latestCheckpointPath,
      bestPolicySaved: isBest,
    };
  }

  async train(iterations, { onIteration } = {}) {
    if (!Number.isInteger(iterations) || iterations < 1) throw new RangeError('iterations must be a positive integer');
    const summaries = [];
    for (let index = 0; index < iterations; index++) {
      const summary = await this.trainIteration();
      summaries.push(summary);
      if (onIteration) await onIteration(summary);
    }
    return summaries;
  }

  async saveCheckpoint({ saveBestPolicy = false, bestScore = null, bestPolicyModel = null } = {}) {
    const payload = {
      format: CHECKPOINT_FORMAT,
      savedAt: new Date().toISOString(),
      model: this.policy.toJSON(),
      trainer: {
        iteration: this.iteration,
        globalSteps: this.globalSteps,
        globalFrames: this.globalFrames,
        completedEpisodes: this.completedEpisodes,
        bestMeanEpisodeReward: this.bestMeanEpisodeReward,
        config: this.config,
        environment: {
          actionRepeat: Number.isInteger(this.environment.actionRepeat) ? this.environment.actionRepeat : null,
          maxSteps: Number.isInteger(this.environment.maxSteps) ? this.environment.maxSteps : null,
        },
        resetOptions: this.resetOptions,
        seedSchedule: this.seedSchedule,
        curriculumStage: this.curriculumStage ?? null,
        resumeStartsNewEpisode: true,
      },
    };
    await atomicWriteJson(this.latestCheckpointPath, payload);
    if (saveBestPolicy) {
      await atomicWriteJson(this.bestPolicyPath, {
        ...(bestPolicyModel ?? this.policy.toJSON()),
        best: {
          iteration: this.iteration,
          meanEpisodeReward: bestScore,
          completedEpisodes: this.completedEpisodes,
        },
      });
    }
    return { latest: this.latestCheckpointPath, bestPolicy: saveBestPolicy ? this.bestPolicyPath : null };
  }

  /** Restore model, Adam state and counters; the engine starts a fresh episode. */
  async restoreCheckpoint(path = this.latestCheckpointPath, { configOverrides = {}, resetOptionsOverride, seedScheduleOverride } = {}) {
    const payload = JSON.parse(await readFile(resolve(path), 'utf8'));
    if (!payload || payload.format !== CHECKPOINT_FORMAT || !payload.model || !payload.trainer) {
      throw new TypeError(`checkpoint must use format ${CHECKPOINT_FORMAT}`);
    }
    this.policy = ActorCriticNetwork.fromJSON(payload.model);
    this.config = { ...DEFAULT_CONFIG, ...(payload.trainer.config ?? {}), ...configOverrides };
    this.#validateConfig();
    this.buffer = new RolloutBuffer({ capacity: this.config.rolloutSteps });
    this.resetOptions = resetOptionsOverride
      ? { ...resetOptionsOverride }
      : { ...(payload.trainer.resetOptions ?? this.resetOptions) };
    const restoredSeedSchedule = seedScheduleOverride ?? payload.trainer.seedSchedule ?? this.seedSchedule;
    if (!Array.isArray(restoredSeedSchedule) || restoredSeedSchedule.some(seed => typeof seed !== 'string')) {
      throw new TypeError('checkpoint seedSchedule must be an array of seed strings');
    }
    this.seedSchedule = [...restoredSeedSchedule];
    this.iteration = Number(payload.trainer.iteration) || 0;
    this.globalSteps = Number(payload.trainer.globalSteps) || 0;
    this.globalFrames = Number(payload.trainer.globalFrames) || 0;
    this.completedEpisodes = Number(payload.trainer.completedEpisodes) || 0;
    this.roomsCleared = Number(payload.trainer.roomsCleared) || 0;
    this.deaths = Number(payload.trainer.deaths) || 0;
    this.curriculumStage = payload.trainer.curriculumStage ?? null;
    this.bestMeanEpisodeReward = Number.isFinite(payload.trainer.bestMeanEpisodeReward)
      ? payload.trainer.bestMeanEpisodeReward : null;
    this.episodeReturn = 0;
    this.episodeStats = null;
    this.observation = null;
    this.lastAction = null;
    return {
      iteration: this.iteration,
      globalSteps: this.globalSteps,
      globalFrames: this.globalFrames,
      completedEpisodes: this.completedEpisodes,
      roomsCleared: this.roomsCleared,
      deaths: this.deaths,
      bestMeanEpisodeReward: this.bestMeanEpisodeReward,
      resumedFrom: resolve(path),
    };
  }
}

export { DEFAULT_CONFIG as PPO_DEFAULT_CONFIG };

