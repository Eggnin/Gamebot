// Reward-shaping adapter for PPO.
//
// The trainer keeps using the unchanged GamebotTrainingEnvironment; this wrapper
// replaces its default scalar reward with a combat-shaped one built from the
// real transition (actual enemy HP removed, actual health lost, clear/death)
// plus an anti-stall term. It never invents damage: every positive component is
// derived from `info.enemyDamage`, `info.healthUnitLost` or the terminal state.
export const DEFAULT_SHAPING = Object.freeze({
  damage: 1.0,            // per HP actually removed from a hostile entity
  healthLost: 8.0,        // per half-heart actually lost
  clear: 40.0,
  death: 40.0,
  frame: 0.002,           // per native logic frame
  noDamageFrames: 120,    // 4 s at 30 Hz before the anti-stall term starts
  noDamagePerFrame: 0.02, // per frame while no damage has been dealt
  idlePenalty: 0.01,      // per decision with move=none and fire=none
});

export class ShapedEnvironment {
  constructor(environment, options = {}) {
    if (!environment || typeof environment.reset !== 'function' || typeof environment.step !== 'function') {
      throw new TypeError('ShapedEnvironment needs an environment with reset() and step()');
    }
    this.environment = environment;
    this.weights = {...DEFAULT_SHAPING, ...(options.weights ?? {})};
    this.lastObservation = null;
    this.framesSinceDamage = 0;
  }

  get actionRepeat() { return this.environment.actionRepeat; }
  get maxSteps() { return this.environment.maxSteps; }

  async reset(options = {}) {
    this.lastObservation = await this.environment.reset(options);
    this.framesSinceDamage = 0;
    return this.lastObservation;
  }

  async step(action) {
    const before = this.lastObservation;
    const result = await this.environment.step(action);
    const after = result.observation;
    const weights = this.weights;
    const dealt = Math.max(0, Number(result.info?.enemyDamage) || 0);
    const lost = Math.max(0, Number(result.info?.healthUnitLost) || 0);
    const frames = Math.max(1, Number(result.info?.framesAdvanced) || 1);
    const cleared = before?.terminal?.cleared !== true && after?.terminal?.cleared === true;
    const dead = before?.terminal?.dead !== true && after?.terminal?.dead === true;

    let reward = 0;
    const components = {
      damage: dealt * weights.damage,
      health: -lost * weights.healthLost,
      clear: cleared ? weights.clear : 0,
      death: dead ? -weights.death : 0,
      time: -frames * weights.frame,
      stall: 0,
      idle: 0,
    };
    if (dealt > 0) this.framesSinceDamage = 0;
    else this.framesSinceDamage += frames;
    if (this.framesSinceDamage > weights.noDamageFrames) {
      components.stall = -frames * weights.noDamagePerFrame;
    }
    if (action?.move === 'none' && action?.fire === 'none' && !cleared && !dead) {
      components.idle = -weights.idlePenalty;
    }
    for (const value of Object.values(components)) reward += value;

    this.lastObservation = after;
    return {
      ...result,
      environmentReward: result.reward,
      reward,
      rewardComponents: components,
      info: {...(result.info ?? {}), shapedReward: reward, shapedComponents: components},
    };
  }

  async close() {
    return this.environment.close?.();
  }
}
