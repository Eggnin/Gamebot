import { createHash } from 'node:crypto';
import { fork, execFileSync } from 'node:child_process';
import {
  appendFileSync, copyFileSync, existsSync, linkSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync,
} from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const EXPECTED_EXE_SHA256 = '7122AC28779925B24E23E2416F231322B1470388BD25E2C08665AD8D53B3EA4F';
const DEFAULT_RESEARCH_ROOT = 'D:\\Isaac\\research';
const DEFAULT_SEED = '3JY1 6FLR';
const AUTO_SEEDS = Object.freeze(['3JY1 6FLR', '7W3L P1Z9', 'SWGF G7WV']);
export const ROOM_SCENARIOS = Object.freeze({
  monstro: Object.freeze({ key: 'monstro', room: 's.boss.1010', type: 5, variant: 1010, targetNPC: 20, label: 'Monstro' }),
  larryJr: Object.freeze({ key: 'larryJr', room: 's.boss.1020', type: 5, variant: 1020, targetNPC: 19, label: 'Larry Jr.' }),
  chub: Object.freeze({ key: 'chub', room: 's.boss.1030', type: 5, variant: 1030, targetNPC: 28, label: 'Chub' }),
  dukeOfFlies: Object.freeze({ key: 'dukeOfFlies', room: 's.boss.2010', type: 5, variant: 2010, targetNPC: 67, label: 'Duke of Flies' }),
  blastocyst: Object.freeze({ key: 'blastocyst', room: 's.boss.2040', type: 5, variant: 2040, targetNPC: 74, label: 'Blastocyst' }),
  loki: Object.freeze({ key: 'loki', room: 's.boss.2030', type: 5, variant: 2030, targetNPC: 69, label: 'Loki' }),
  gurdyJr: Object.freeze({ key: 'gurdyJr', room: 's.boss.3280', type: 5, variant: 3280, targetNPC: 99, label: 'Gurdy Jr.' }),
  theBloat: Object.freeze({ key: 'theBloat', room: 's.boss.3300', type: 5, variant: 3300, targetNPC: 68, label: 'The Bloat' }),
  widow: Object.freeze({ key: 'widow', room: 's.boss.3340', type: 5, variant: 3340, targetNPC: 100, label: 'Widow' }),
  famine: Object.freeze({ key: 'famine', room: 's.boss.4010', type: 5, variant: 4010, targetNPC: 63, label: 'Famine' }),
  pestilence: Object.freeze({ key: 'pestilence', room: 's.boss.4020', type: 5, variant: 4020, targetNPC: 64, label: 'Pestilence' }),
  theCage: Object.freeze({ key: 'theCage', room: 's.boss.5060', type: 5, variant: 5060, targetNPC: 265, label: 'The Cage' }),
  frowningGaperGroup: Object.freeze({ key: 'frowningGaperGroup', room: 'd.841', type: 1, variant: 841,
    shape: 1, width: 15, height: 9, targetNPC: 10, targetVariant: 0, targetSubtype: 0,
    initialCount: 5, label: 'Five Frowning Gapers' }),
});
const ALL_SCENARIOS = Object.freeze(Object.values(ROOM_SCENARIOS));
const AUTO_SCENARIOS = Object.freeze(ALL_SCENARIOS.filter(scenario => scenario.type === 5));
const DEFAULT_REWARD = Object.freeze({
  enemyDamage: 1,
  healthUnitLost: 5,
  clear: 25,
  death: 25,
  frameCost: 0.001,
});

const MOVES = Object.freeze({
  none: [0, 0],
  up: [0, -1],
  up_right: [1, -1],
  right: [1, 0],
  down_right: [1, 1],
  down: [0, 1],
  down_left: [-1, 1],
  left: [-1, 0],
  up_left: [-1, -1],
});
const FIRES = new Set(['none', 'up', 'right', 'down', 'left']);
const OBSERVED_EFFECT_VARIANTS = new Set([4, 19, 22, 23, 24, 25, 26, 31, 56, 62]);

function resolveScenario({ scene, room } = {}, episodeIndex = 0) {
  const requested = room ?? scene ?? 'monstro';
  if (requested === 'auto') return AUTO_SCENARIOS[episodeIndex % AUTO_SCENARIOS.length];
  const scenario = ROOM_SCENARIOS[requested] ?? ALL_SCENARIOS.find(candidate => candidate.room === requested);
  if (!scenario) throw new TypeError(`unknown room scenario: ${requested}`);
  return scenario;
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex').toUpperCase();
}

function verifyRuntime(researchRoot) {
  const manifestPath = join(researchRoot, 'runtime-inputs', 'manifest.json');
  if (!existsSync(manifestPath)) throw new Error(`runtime manifest missing: ${manifestPath}`);
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (manifest.sourceExecutableSha256 !== EXPECTED_EXE_SHA256) {
    throw new Error(`unsupported game build in runtime manifest: ${manifest.sourceExecutableSha256}`);
  }
  const exePath = manifest.sourceExecutable;
  if (!existsSync(exePath)) throw new Error(`source game executable missing: ${exePath}`);
  const actualExeSha = sha256(readFileSync(exePath));
  if (actualExeSha !== EXPECTED_EXE_SHA256) {
    throw new Error(`game executable SHA-256 mismatch: expected ${EXPECTED_EXE_SHA256}, got ${actualExeSha}`);
  }

  const segmentPath = join(researchRoot, 'runtime-inputs', 'host', 'isaac.segs.bin');
  const segmentManifest = manifest.outputs?.['host\\isaac.segs.bin'];
  if (!segmentManifest || !existsSync(segmentPath)) throw new Error(`verified memory image missing: ${segmentPath}`);
  const segmentBytes = readFileSync(segmentPath);
  if (segmentBytes.length !== segmentManifest.bytes || sha256(segmentBytes) !== segmentManifest.sha256.toUpperCase()) {
    throw new Error('memory image does not match the runtime manifest');
  }
  const imageCheckPath = join(researchRoot, 'runtime-inputs', 'host', 'memimage-verify.json');
  const imageCheck = JSON.parse(readFileSync(imageCheckPath, 'utf8'));
  if (imageCheck.pass !== true || imageCheck.sha256?.toUpperCase() !== EXPECTED_EXE_SHA256) {
    throw new Error('memory image verification record does not match the supported executable');
  }

  const required = [
    join(PACKAGE_ROOT, 'src', 'engine_worker.mjs'),
    join(researchRoot, 'offline', 'training_observer.mjs'),
    join(researchRoot, 'offline', 'core_policy.mjs'),
    join(researchRoot, 'offline', 'canonical_state.mjs'),
    join(researchRoot, 'offline', 'planner_frame.mjs'),
    join(researchRoot, 'offline', 'damage_tracking.mjs'),
    join(researchRoot, 'offline', 'planner_policy.wasm'),
    join(researchRoot, 'the-browsing-of-isaac', 'scripts', 'recomp', 'lift', 'explore.mjs'),
    join(researchRoot, 'the-browsing-of-isaac', 'output', 'recomp', 'lift', 'boot-training-fast', 'boot.mjs'),
    join(researchRoot, 'the-browsing-of-isaac', 'output', 'recomp', 'lift', 'boot-training-fast', 'boot.wasm'),
    join(researchRoot, 'runtime-instance', 'resources', 'packed'),
    join(researchRoot, '..', 'data', 'npc', 'engine_monster_static_mechanism_candidates.json'),
  ];
  const missing = required.filter(path => !existsSync(path));
  if (missing.length) throw new Error(`training runtime files missing:\n${missing.join('\n')}`);
  const bootWasm = join(researchRoot, 'the-browsing-of-isaac', 'output', 'recomp', 'lift', 'boot-training-fast', 'boot.wasm');
  return {
    manifest,
    segmentPath,
    exeSha256: actualExeSha,
    segmentSha256: sha256(segmentBytes),
    bootWasmSha256: sha256(readFileSync(bootWasm)),
  };
}

function normalizeSeed(seed) {
  const value = String(seed ?? DEFAULT_SEED).trim().toUpperCase().replace(/\s+/g, ' ');
  if (!/^[A-Z0-9]{4} [A-Z0-9]{4}$/.test(value)) {
    throw new TypeError('seed must be two four-character groups, for example "3JY1 6FLR"');
  }
  return value;
}

function hashObservation(observation) {
  return createHash('sha256').update(JSON.stringify(observation)).digest('hex');
}

export function normalizeAction(action) {
  if (!action || typeof action !== 'object' || Array.isArray(action)) {
    throw new TypeError('action must be an object with only move and fire');
  }
  const extra = Object.keys(action).filter(key => key !== 'move' && key !== 'fire');
  if (extra.length) throw new TypeError(`unsupported action field: ${extra.join(', ')}`);
  const moveName = String(action.move ?? 'none').trim().toLowerCase();
  const fire = String(action.fire ?? 'none').trim().toLowerCase();
  const vector = MOVES[moveName];
  if (!vector) throw new TypeError(`move must be one of: ${Object.keys(MOVES).join(', ')}`);
  if (!FIRES.has(fire)) throw new TypeError('fire must be up/right/down/left/none');
  return { move: moveName, fire, moveX: vector[0], moveY: vector[1] };
}

function alive(entity) {
  return entity?.exists !== false && entity?.dead !== true && entity?.active !== false;
}

function enemyHpMap(observation) {
  const result = new Map();
  for (const entity of observation?.entities ?? []) {
    if (entity.type < 10 || entity.type >= 1000 || entity.friendly || !alive(entity)) continue;
    const id = `${entity.seed || entity.address}:${entity.type}.${entity.variant}.${entity.subtype}`;
    result.set(id, Math.max(0, Number(entity.hp) || 0));
  }
  return result;
}

function rewardDelta(before, after, frames, weights) {
  const beforeHealth = Number(before?.player?.health);
  const afterHealth = Number(after?.player?.health);
  const healthUnitLost = Number.isFinite(beforeHealth) && Number.isFinite(afterHealth)
    ? Math.max(0, beforeHealth - afterHealth) : 0;
  const beforeHp = enemyHpMap(before);
  const afterHp = enemyHpMap(after);
  let enemyDamage = 0;
  for (const [id, hp] of beforeHp) enemyDamage += Math.max(0, hp - (afterHp.get(id) ?? 0));

  const cleared = before?.cleared !== true && after?.cleared === true;
  const died = before?.dead !== true && after?.dead === true;
  const components = {
    enemyDamage: enemyDamage * weights.enemyDamage,
    health: -healthUnitLost * weights.healthUnitLost,
    clear: cleared ? weights.clear : 0,
    death: died ? -weights.death : 0,
    time: -frames * weights.frameCost,
  };
  return {
    reward: Object.values(components).reduce((sum, value) => sum + value, 0),
    enemyDamage,
    healthUnitLost,
    cleared,
    died,
    components,
  };
}

function publicEntity(entity) {
  return {
    kind: entity.kind,
    type: entity.type,
    variant: entity.variant,
    subtype: entity.subtype,
    position: entity.position,
    velocity: entity.velocity,
    radius: entity.radius,
    hp: entity.hp,
    damage: entity.damage,
    collision: entity.collision,
    friendly: entity.friendly,
    invulnerable: entity.invulnerable,
    targetable: entity.targetable,
    dead: entity.dead,
    spawnerType: entity.spawnerType,
  };
}

export function publicObservation(observation) {
  const terminalState = observation?.dead === true || observation?.cleared === true;
  if ((!observation?.ready || observation.state !== 'ready') && !terminalState) {
    throw new Error(`engine observation is not ready: ${observation?.state ?? 'missing'}${observation?.error ? ` (${observation.error})` : ''}`);
  }
  const player = observation.player;
  const entities = (observation.entities ?? []).filter(entity => {
    if (!alive(entity)) return false;
    if (entity.type >= 10 && entity.type < 1000) return true;
    if (entity.type === 2 || entity.type === 7 || entity.type === 9) return true;
    return entity.type === 1000 && OBSERVED_EFFECT_VARIANTS.has(entity.variant);
  }).map(publicEntity);
  const grid = (observation.grid ?? []).map(cell => ({
    index: cell.index,
    x: cell.x,
    y: cell.y,
    type: cell.type,
    variant: cell.variant,
    state: cell.state,
    collision: cell.collision,
    destructible: cell.destructible,
    danger: cell.danger,
  }));
  return {
    schema: 'gamebot-room-observation/1',
    frame: observation.frame,
    room: {
      type: observation.room?.type,
      variant: observation.room?.variant,
      shape: observation.room?.shape,
      width: observation.room?.width,
      height: observation.room?.height,
      cleared: observation.cleared,
    },
    player: {
      position: player.position,
      velocity: player.velocity,
      health: player.health,
      fireDelay: player.fireDelay,
      damageCooldown: player.damageCooldown,
      controls: player.controls,
    },
    entities,
    grid,
    terminal: { cleared: observation.cleared === true, dead: observation.dead === true },
  };
}

export class GamebotTrainingEnvironment {
  constructor(options = {}) {
    const researchRoot = resolve(options.researchRoot ?? process.env.GAMEBOT_RESEARCH_ROOT ?? DEFAULT_RESEARCH_ROOT);
    const runRoot = resolve(options.runRoot ?? process.env.GAMEBOT_RUN_ROOT ?? join(PACKAGE_ROOT, 'runs'));
    const actionRepeat = Number(options.actionRepeat ?? process.env.GAMEBOT_ACTION_REPEAT ?? 4);
    const maxSteps = Number(options.maxSteps ?? 5000);
    const requestTimeoutMs = Number(options.requestTimeoutMs ?? 30000);
    if (!Number.isInteger(actionRepeat) || actionRepeat < 1 || actionRepeat > 16) {
      throw new RangeError('actionRepeat must be an integer from 1 to 16');
    }
    if (!Number.isInteger(maxSteps) || maxSteps < 1) throw new RangeError('maxSteps must be a positive integer');
    if (!Number.isInteger(requestTimeoutMs) || requestTimeoutMs < 1000) {
      throw new RangeError('requestTimeoutMs must be an integer of at least 1000');
    }
    this.researchRoot = researchRoot;
    this.runRoot = runRoot;
    this.actionRepeat = actionRepeat;
    this.maxSteps = maxSteps;
    this.requestTimeoutMs = requestTimeoutMs;
    this.deferInitialSnapshot = options.deferInitialSnapshot === true;
    // Opt-in external validation keeps the native Reader path available so an
    // expert policy process can read the same engine memory the trainer sees.
    // It only enables the worker's `externalTruth` command; training behaviour
    // is unchanged when the flag is absent.
    this.externalValidation = options.externalValidation === true;
    this.rewardWeights = { ...DEFAULT_REWARD, ...(options.rewardWeights ?? {}) };
    for (const [name, weight] of Object.entries(this.rewardWeights)) {
      if (!Number.isFinite(weight) || weight < 0) throw new RangeError(`reward weight ${name} must be a non-negative number`);
    }
    this.pending = new Map();
    this.nextId = 1;
    this.child = null;
    this.rawObservation = null;
    this.ended = false;
    this.snapshots = new Map();
    this.forks = new Map();
    this.anchorSnapshotId = null;
    this.episodeIndex = 0;
    this.episodeNumber = 0;
    this.resetMode = null;
  }

  async reset(options = {}) {
    const requestedScene = options.room ?? options.scene ?? (options.seed === undefined ? 'auto' : 'monstro');
    const scene = resolveScenario({ scene: requestedScene, room: options.room }, this.episodeIndex);
    const defaultSeed = requestedScene === 'auto'
      ? AUTO_SEEDS[this.episodeIndex % AUTO_SEEDS.length]
      : DEFAULT_SEED;
    const seed = normalizeSeed(options.seed ?? defaultSeed);

    if (this.child?.connected) {
      if (this.anchorSnapshotId != null && this.seed === seed && this.scene?.key === scene.key) {
        try {
          const result = await this.#request('restore', { snapshotId: this.anchorSnapshotId });
          const observation = publicObservation(result.observation);
          const stored = this.snapshots.get(this.anchorSnapshotId);
          if (!stored || hashObservation(observation) !== stored.publicObservationHash) {
            throw new Error('automatic reset snapshot failed public observation verification');
          }
          this.rawObservation = result.observation;
          this.frames = result.steps;
          this.ended = result.observation.dead === true || result.observation.cleared === true;
          this.resetMode = 'snapshot-restore';
          this.resetFallbackReason = null;
          this.#writeEpisodeStart(observation);
          this.episodeIndex++;
          return observation;
        } catch (error) {
          this.resetFallbackReason = error instanceof Error ? error.message : String(error);
          await this.close();
        }
      } else {
        try {
          this.snapshots.clear();
          this.forks.clear();
          this.anchorSnapshotId = null;
          const result = await this.#request('reset', {
            seed,
            room: scene.room,
            targetNPC: scene.targetNPC,
            roomType: scene.type,
            stage: 1,
          });
          const logText = readFileSync(join(this.runDirectory, 'engine.log'), 'utf8');
          const seedMatches = [...logText.matchAll(/RNG Start Seed:\s*([A-Z0-9]{4}) ([A-Z0-9]{4})/g)];
          const confirmedSeed = seedMatches.at(-1)?.slice(1).join(' ');
          if (confirmedSeed !== seed || seedMatches.length <= (this.seedLogCount ?? 0)) {
            throw new Error(`warm reset seed was not newly confirmed: ${confirmedSeed ?? 'none'}`);
          }
          this.seedLogCount = seedMatches.length;
          this.rawObservation = result.observation;
          this.#assertScenarioObservation(result.observation, scene, true);
          this.seed = seed;
          this.scene = scene;
          this.frames = 0;
          this.ended = result.observation.dead === true || result.observation.cleared === true;
          this.resetMode = 'warm-console';
          this.resetFallbackReason = null;
          await this.#captureSnapshot();
          const observation = publicObservation(result.observation);
          this.#writeEpisodeStart(observation);
          this.episodeIndex++;
          return observation;
        } catch (error) {
          this.resetFallbackReason = error instanceof Error ? error.message : String(error);
          await this.close();
        }
      }
    }

    return this.#coldReset(seed, scene);
  }

  #assertScenarioObservation(raw, scene, initial = false) {
    if (raw?.room?.type !== scene.type || raw?.room?.variant !== scene.variant) {
      throw new Error(`unexpected room identity: ${raw?.room?.type}.${raw?.room?.variant}, expected ${scene.type}.${scene.variant}`);
    }
    if (scene.shape !== undefined && (raw.room.shape !== scene.shape ||
        raw.room.width !== scene.width || raw.room.height !== scene.height)) {
      throw new Error(`unexpected room geometry in ${scene.room}`);
    }
    if (!raw?.entities?.some(entity => entity.type === scene.targetNPC && alive(entity))) {
      throw new Error(`target NPC ${scene.targetNPC} was not observed in ${scene.room}`);
    }
    if (initial && scene.initialCount !== undefined) {
      const enemies = raw.entities.filter(entity => entity.type >= 10 && entity.type < 1000 &&
        !entity.friendly && alive(entity));
      if (raw.player?.health !== 6 || enemies.length !== scene.initialCount ||
          enemies.some(entity => entity.type !== scene.targetNPC || entity.variant !== scene.targetVariant ||
            entity.subtype !== scene.targetSubtype)) {
        throw new Error(`unexpected initial enemies or health in ${scene.room}`);
      }
    }
  }

  #writeEpisodeStart(observation) {
    this.episodeNumber++;
    const fileName = this.episodeNumber === 1
      ? 'trajectory.jsonl'
      : `trajectory-${String(this.episodeNumber).padStart(4, '0')}.jsonl`;
    this.trajectoryPath = join(this.runDirectory, fileName);
    writeFileSync(this.trajectoryPath, '');
    appendFileSync(this.trajectoryPath, `${JSON.stringify({
      kind: 'reset', seed: this.seed, scene: this.scene.key, room: this.scene.room,
      resetMode: this.resetMode,
      sourceExecutableSha256: this.runtimeMetadata.exeSha256,
      segmentImageSha256: this.runtimeMetadata.segmentSha256,
      bootWasmSha256: this.runtimeMetadata.bootWasmSha256,
      frame: observation.frame, observation,
    })}\n`);
  }

  async #captureSnapshot() {
    const saved = await this.#request('snapshot');
    const observation = publicObservation(this.rawObservation);
    const metadata = {
      ...saved,
      seed: this.seed,
      scene: this.scene.key,
      room: this.scene.room,
      publicObservationHash: hashObservation(observation),
      episodeFrames: this.frames,
    };
    this.snapshots.set(saved.id, metadata);
    this.anchorSnapshotId = saved.id;
    return metadata;
  }

  async #coldReset(seed, scene) {
    await this.close();
    this.seed = seed;
    this.scene = scene;
    const runtime = verifyRuntime(this.researchRoot);
    mkdirSync(this.runRoot, { recursive: true });
    const runId = `${Date.now()}-${process.pid}-${Math.random().toString(36).slice(2, 8)}`;
    const run = join(this.runRoot, runId);
    const instance = join(run, 'instance');
    const resourceSource = join(this.researchRoot, 'runtime-instance', 'resources');
    mkdirSync(instance, { recursive: true });
    try {
      symlinkSync(resourceSource, join(instance, 'resources'), 'junction');
    } catch (error) {
      throw new Error(`could not create isolated resource junction (${error.message}); source data was left untouched`);
    }
    mkdirSync(join(run, 'saves'), { recursive: true });
    const config = {
      seed: this.seed,
      room: scene.room,
      targetNPC: scene.targetNPC,
      maxSteps: this.maxSteps,
      snapshotRoot: this.runRoot,
      sourceExecutableSha256: runtime.exeSha256,
      bootWasmSha256: runtime.bootWasmSha256,
      setupTimeoutMs: 120000,
      noRender: true,
      setupTrace: process.env.GAMEBOT_RESET_TRACE === '1',
      debugFlags: [],
      drive: 'explore',
      research: this.researchRoot,
      run,
      instance,
      externalValidation: this.externalValidation,
    };
    this.runDirectory = run;
    this.episodeNumber = 0;
    this.snapshots.clear();
    this.forks.clear();
    this.anchorSnapshotId = null;
    this.rawObservation = null;
    this.ended = false;
    const logPath = join(run, 'engine.log');
    const logFd = (await import('node:fs')).openSync(logPath, 'w');
    const workerEntry = join(PACKAGE_ROOT, 'src', 'engine_worker.mjs');
    const child = fork(workerEntry, [], {
      cwd: instance,
      windowsHide: true,
      execArgv: ['--no-wasm-tier-up'],
      env: { ...process.env, ISAAC_TRAINING_CONFIG: JSON.stringify(config) },
      stdio: ['ignore', logFd, logFd, 'ipc'],
      serialization: 'advanced',
    });
    (await import('node:fs')).closeSync(logFd);
    this.child = child;
    this.workerPid = child.pid;

    const raw = await new Promise((resolveReady, rejectReady) => {
      let settled = false;
      const fail = error => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        rejectReady(error);
      };
      const timer = setTimeout(() => {
        fail(new Error(`engine setup timed out; see ${logPath}`));
        child.kill();
      }, 120000);
      child.on('message', message => {
        if (message.event === 'ready') {
          const log = readFileSync(logPath, 'utf8');
          const seeds = [...log.matchAll(/RNG Start Seed:\s*([A-Z0-9]{4}) ([A-Z0-9]{4})/g)];
          const actualSeed = seeds.at(-1)?.slice(1).join(' ');
          if (actualSeed !== this.seed) {
            fail(new Error(`native seed mismatch: requested ${this.seed}, engine confirmed ${actualSeed ?? 'none'}; see ${logPath}`));
            child.kill();
            return;
          }
          if (message.observation?.room?.type !== scene.type || message.observation?.room?.variant !== scene.variant) {
            fail(new Error(`unexpected room identity: ${message.observation?.room?.type}.${message.observation?.room?.variant}, expected ${scene.type}.${scene.variant}`));
            child.kill();
            return;
          }
          if (!settled) {
            settled = true;
            clearTimeout(timer);
            resolveReady(message.observation);
          }
        } else if (message.event === 'fatal') {
          const error = new Error(`${message.error}; see ${logPath}`);
          fail(error);
          for (const request of this.pending.values()) {
            clearTimeout(request.timer);
            request.reject(error);
          }
          this.pending.clear();
          child.kill();
        } else if (message.id) {
          const request = this.pending.get(message.id);
          if (!request) return;
          this.pending.delete(message.id);
          clearTimeout(request.timer);
          if (message.error) request.reject(new Error(message.error));
          else request.resolve(message.result);
        }
      });
      child.on('error', fail);
      child.on('exit', (code, signal) => {
        if (this.child === child) this.child = null;
        const error = new Error(`engine worker exited (${code ?? signal}); see ${logPath}`);
        fail(error);
        for (const request of this.pending.values()) {
          clearTimeout(request.timer);
          request.reject(error);
        }
        this.pending.clear();
      });
    });
    this.rawObservation = raw;
    this.frames = 0;
    this.ended = raw.dead === true || raw.cleared === true;
    this.resetMode = this.resetFallbackReason ? 'cold-fallback' : 'cold';
    this.resetFallbackReason = null;
    this.seedLogCount = [...readFileSync(logPath, 'utf8').matchAll(/RNG Start Seed:\s*([A-Z0-9]{4}) ([A-Z0-9]{4})/g)].length;
    this.runtimeMetadata = {
      exeSha256: runtime.exeSha256,
      segmentSha256: runtime.segmentSha256,
      bootWasmSha256: runtime.bootWasmSha256,
      sourceExecutableSha256: runtime.exeSha256,
      segmentImageSha256: runtime.segmentSha256,
      bootWasmSha256: runtime.bootWasmSha256,
    };
    const observation = publicObservation(raw);
    try { this.#assertScenarioObservation(raw, scene, true); }
    catch (error) { await this.close(); throw error; }
    this.#writeEpisodeStart(observation);
    if (this.deferInitialSnapshot) this.deferInitialSnapshot = false;
    else await this.#captureSnapshot();
    this.episodeIndex++;
    return observation;
  }

  #request(command, action) {
    if (!this.child?.connected) return Promise.reject(new Error('call reset() before using the environment'));
    const id = this.nextId++;
    return new Promise((resolveRequest, rejectRequest) => {
      const timer = setTimeout(() => {
        if (!this.pending.delete(id)) return;
        rejectRequest(new Error(`engine did not complete '${command}' within ${this.requestTimeoutMs} ms`));
        this.child?.kill();
      }, this.requestTimeoutMs);
      this.pending.set(id, { resolve: resolveRequest, reject: rejectRequest, timer });
      this.child.send({ id, command, action }, error => {
        if (!error) return;
        this.pending.delete(id);
        clearTimeout(timer);
        rejectRequest(error);
      });
    });
  }

  async observe() {
    if (!this.child?.connected || !this.rawObservation) throw new Error('call reset() before observe()');
    const raw = await this.#request('peek');
    this.rawObservation = raw;
    return publicObservation(raw);
  }

  // Additive transport for validation tooling. With `externalValidation: true`
  // the worker answers `externalTruth`, which returns the live WASM heap
  // address and an independently captured damage oracle for a native Reader.
  // It never changes the observation or action contract used for training.
  request(command, action) {
    return this.#request(command, action);
  }

  async snapshot(label = '') {
    if (!this.child?.connected || !this.rawObservation) throw new Error('call reset() before snapshot()');
    const saved = await this.#request('snapshot');
    const observation = publicObservation(this.rawObservation);
    const metadata = {
      ...saved,
      label: String(label).slice(0, 120),
      seed: this.seed,
      scene: this.scene.key,
      room: this.scene.room,
      publicObservationHash: hashObservation(observation),
      episodeFrames: this.frames,
    };
    this.snapshots.set(saved.id, metadata);
    return metadata;
  }

  async restore(snapshotOrFork) {
    if (!this.child?.connected || !this.rawObservation) throw new Error('call reset() before restore()');
    const id = typeof snapshotOrFork === 'object'
      ? (snapshotOrFork.branchId ?? snapshotOrFork.id)
      : snapshotOrFork;
    const snapshotId = typeof id === 'string'
      ? this.forks.get(id)?.snapshotId
      : id;
    const metadata = this.snapshots.get(snapshotId);
    if (!metadata) throw new Error(`unknown snapshot/fork handle: ${id}`);
    const result = await this.#request('restore', { id });
    const observation = publicObservation(result.observation);
    const actualHash = hashObservation(observation);
    if (actualHash !== metadata.publicObservationHash) {
      throw new Error(`restored public observation mismatch: expected ${metadata.publicObservationHash}, got ${actualHash}`);
    }
    this.rawObservation = result.observation;
    this.frames = result.steps;
    this.ended = result.observation.dead === true || result.observation.cleared === true;
    this.seed = metadata.seed;
    this.scene = ROOM_SCENARIOS[metadata.scene];
    appendFileSync(this.trajectoryPath, `${JSON.stringify({
      kind: 'restore', snapshotId: metadata.id, frame: observation.frame,
      scene: metadata.scene, seed: metadata.seed, observationHash: actualHash,
    })}\n`);
    return observation;
  }

  async fork(snapshotOrFork, label = '') {
    if (!this.child?.connected || !this.rawObservation) throw new Error('call reset() before fork()');
    const id = typeof snapshotOrFork === 'object'
      ? (snapshotOrFork.branchId ?? snapshotOrFork.id)
      : snapshotOrFork;
    const snapshotId = typeof id === 'string' ? this.forks.get(id)?.snapshotId : id;
    if (!this.snapshots.has(snapshotId)) throw new Error(`unknown snapshot/fork handle: ${id}`);
    const branch = await this.#request('fork', { snapshotId, label });
    this.forks.set(branch.id, branch);
    return branch;
  }

  async dropSnapshot(snapshotOrFork) {
    if (!this.child?.connected) throw new Error('call reset() before dropSnapshot()');
    const id = typeof snapshotOrFork === 'object'
      ? (snapshotOrFork.branchId ?? snapshotOrFork.id)
      : snapshotOrFork;
    const snapshotId = typeof id === 'string' ? this.forks.get(id)?.snapshotId : id;
    const metadata = this.snapshots.get(snapshotId);
    const dropped = await this.#request('dropSnapshot', { id });
    if (typeof id === 'string') {
      this.forks.delete(id);
      return dropped;
    }
    this.snapshots.delete(id);
    if (this.anchorSnapshotId === id) this.anchorSnapshotId = null;
    if (metadata?.ownsExportedFile && metadata.exportedPath) {
      const exported = resolve(metadata.exportedPath);
      const rel = relative(this.runRoot, exported);
      if (rel && rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel)) {
        rmSync(exported, { force: true });
        rmSync(`${exported}.json`, { force: true });
      }
    }
    for (const [branchId, branch] of this.forks) {
      if (branch.snapshotId === id) this.forks.delete(branchId);
    }
    return dropped;
  }

  async exportSnapshot(snapshotOrFork) {
    if (!this.child?.connected) throw new Error('call reset() before exportSnapshot()');
    const id = typeof snapshotOrFork === 'object'
      ? (snapshotOrFork.branchId ?? snapshotOrFork.id)
      : snapshotOrFork;
    const snapshotId = typeof id === 'string' ? this.forks.get(id)?.snapshotId : id;
    const metadata = this.snapshots.get(snapshotId);
    if (!metadata) throw new Error(`unknown snapshot/fork handle: ${id}`);
    if (metadata.exportedPath && existsSync(metadata.exportedPath) && existsSync(`${metadata.exportedPath}.json`)) {
      return { snapshotId, path: metadata.exportedPath, bytes: metadata.bytes, memorySha256: metadata.memorySha256 };
    }
    const path = join(this.runDirectory, 'snapshots', `snapshot-${snapshotId}.bin`);
    const exported = await this.#request('snapshot_export', { id: snapshotId, path });
    metadata.exportedPath = exported.path;
    metadata.ownsExportedFile = true;
    return exported;
  }

  async importSnapshot(path, { ownFile = false } = {}) {
    if (!this.child?.connected || !this.rawObservation) throw new Error('call reset() before importSnapshot()');
    const imported = await this.#request('snapshot_import', { path: resolve(path) });
    const scene = resolveScenario({ room: imported.room });
    const observation = publicObservation(imported.observation);
    this.#assertScenarioObservation(imported.observation, scene);
    const metadata = {
      id: imported.snapshotId,
      filePath: imported.filePath,
      bytes: imported.memoryBytes,
      memorySha256: imported.memorySha256,
      frame: imported.frame,
      steps: imported.steps,
      seed: imported.seed,
      scene: scene.key,
      room: scene.room,
      publicObservationHash: hashObservation(observation),
      episodeFrames: imported.steps,
      exportedPath: imported.filePath,
      ownsExportedFile: ownFile,
    };
    this.snapshots.set(imported.snapshotId, metadata);
    this.anchorSnapshotId = imported.snapshotId;
    this.rawObservation = imported.observation;
    this.frames = imported.steps;
    this.ended = imported.observation.dead === true || imported.observation.cleared === true;
    this.seed = imported.seed;
    this.scene = scene;
    this.resetMode = 'fork-import';
    appendFileSync(this.trajectoryPath, `${JSON.stringify({
      kind: 'fork-import', snapshotId: imported.snapshotId, sourcePath: imported.filePath,
      seed: imported.seed, scene: scene.key, frame: observation.frame,
      memorySha256: imported.memorySha256, observation,
    })}\n`);
    return observation;
  }

  async forkParallel(snapshotOrFork, label = '') {
    const exported = await this.exportSnapshot(snapshotOrFork);
    const id = typeof snapshotOrFork === 'object'
      ? (snapshotOrFork.branchId ?? snapshotOrFork.id)
      : snapshotOrFork;
    const snapshotId = typeof id === 'string' ? this.forks.get(id)?.snapshotId : id;
    const metadata = this.snapshots.get(snapshotId);
    if (!metadata) throw new Error(`unknown snapshot/fork handle: ${id}`);
    const branch = new GamebotTrainingEnvironment({
      researchRoot: this.researchRoot,
      runRoot: this.runRoot,
      actionRepeat: this.actionRepeat,
      maxSteps: this.maxSteps,
      requestTimeoutMs: this.requestTimeoutMs,
      rewardWeights: this.rewardWeights,
      deferInitialSnapshot: true,
    });
    let branchSnapshotPath = null;
    try {
      await branch.reset({ seed: metadata.seed, scene: metadata.scene });
      const snapshotDirectory = join(branch.runDirectory, 'snapshots');
      mkdirSync(snapshotDirectory, { recursive: true });
      branchSnapshotPath = join(snapshotDirectory, `fork-${snapshotId}.bin`);
      for (const sourcePath of [exported.path, `${exported.path}.json`]) {
        const destinationPath = sourcePath.endsWith('.json')
          ? `${branchSnapshotPath}.json`
          : branchSnapshotPath;
        try { linkSync(sourcePath, destinationPath); }
        catch { copyFileSync(sourcePath, destinationPath); }
      }
      await branch.importSnapshot(branchSnapshotPath, { ownFile: true });
      branch.branchLabel = String(label).slice(0, 120);
      return branch;
    } catch (error) {
      await branch.close();
      if (branchSnapshotPath) {
        rmSync(branchSnapshotPath, { force: true });
        rmSync(`${branchSnapshotPath}.json`, { force: true });
      }
      throw error;
    }
  }

  async step(action) {
    if (!this.child?.connected || !this.rawObservation) throw new Error('call reset() before step()');
    if (this.ended) throw new Error('episode ended; call reset()');
    const normalized = normalizeAction(action);
    const before = this.rawObservation;
    const publicBefore = publicObservation(before);
    let after = before;
    let reward = 0;
    let framesAdvanced = 0;
    let terminated = false;
    let truncated = false;
    const components = { enemyDamage: 0, health: 0, clear: 0, death: 0, time: 0 };
    let enemyDamage = 0;
    let healthUnitLost = 0;
    let previous = before;
    for (let index = 0; index < this.actionRepeat; index += 1) {
      const result = await this.#request('step', {
        moveX: normalized.moveX,
        moveY: normalized.moveY,
        fire: normalized.fire,
      });
      after = result.observation;
      framesAdvanced += 1;
      this.frames += 1;
      const delta = rewardDelta(previous, after, 1, this.rewardWeights);
      reward += delta.reward;
      enemyDamage += delta.enemyDamage;
      healthUnitLost += delta.healthUnitLost;
      for (const key of Object.keys(components)) components[key] += delta.components[key];
      terminated = result.terminated === true;
      truncated = result.truncated === true;
      previous = after;
      if (terminated || truncated) break;
    }
    this.rawObservation = after;
    this.ended = terminated || truncated;
    const observation = publicObservation(after);
    const info = {
      framesAdvanced,
      totalFrames: this.frames,
      enemyDamage,
      healthUnitLost,
      rewardComponents: components,
    };
    appendFileSync(this.trajectoryPath, `${JSON.stringify({
      kind: 'step', seed: this.seed, action: { move: normalized.move, fire: normalized.fire },
      framesAdvanced, reward, terminated, truncated, info,
      observationBefore: publicBefore,
      observation,
    })}\n`);
    return { observation, reward, terminated, truncated, info };
  }

  async close() {
    const child = this.child;
    this.child = null;
    for (const request of this.pending.values()) {
      clearTimeout(request.timer);
      request.reject(new Error('environment closed'));
    }
    this.pending.clear();
    this.snapshots.clear();
    this.forks.clear();
    this.anchorSnapshotId = null;
    if (!child) return;
    await new Promise(resolveExit => {
      if (child.exitCode !== null || child.signalCode !== null) return resolveExit();
      child.once('exit', resolveExit);
      child.kill();
    });
  }
}

export function expectedGameBuildSha256() {
  return EXPECTED_EXE_SHA256;
}

export function sampleWorkerMemory(pid) {
  if (process.platform !== 'win32' || !Number.isInteger(pid) || pid <= 0) return null;
  try {
    const script = `$p=Get-Process -Id ${pid} -ErrorAction Stop; @{workingSetBytes=$p.WorkingSet64;privateBytes=$p.PrivateMemorySize64} | ConvertTo-Json -Compress`;
    return JSON.parse(execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true,
    }));
  } catch {
    return null;
  }
}
