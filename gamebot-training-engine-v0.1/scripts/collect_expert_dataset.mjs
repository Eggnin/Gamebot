// Phase 1 -- expert policy data collection.
//
// Runs the existing C++ Gamebot policy (MPC by default) inside the offline
// J460 training engine and records one aligned (observation, expert action,
// transition outcome) sample per native logic frame.
//
// The expert is not reimplemented here. `external-validation/reader_main.cpp`
// exposes the production `MpcCombatPlanner` / `Planner` over the existing
// native Reader bridge; this script only drives it, applies the action through
// the training environment, and writes the dataset.
//
// Usage:
//   node scripts/collect_expert_dataset.mjs --scenes=monstro,theBloat --seeds=3JY1\ 6FLR --frames=900
//   node scripts/collect_expert_dataset.mjs --policy=quick --output=runs/expert-quick-control
import {appendFileSync, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {GamebotTrainingEnvironment, ROOM_SCENARIOS} from '../src/environment.mjs';

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RESEARCH_ROOT = resolve(process.env.GAMEBOT_RESEARCH_ROOT ?? 'D:\\Isaac\\research');
const MOVES = ['none', 'up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left'];
const FIRES = ['none', 'up', 'right', 'down', 'left'];
const DEFAULT_SOURCE_EXE = 'D:/Steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng.exe';

function option(name, fallback) {
  const prefix = `--${name}=`;
  const value = process.argv.find(argument => argument.startsWith(prefix));
  return value === undefined ? fallback : value.slice(prefix.length);
}
function flag(name) { return process.argv.includes(`--${name}`); }
function sha256File(path) { return createHash('sha256').update(readFileSync(path)).digest('hex'); }
function sha256Text(text) { return createHash('sha256').update(text).digest('hex'); }
function list(value) {
  return String(value ?? '').split(',').map(entry => entry.trim()).filter(Boolean);
}

const scenes = list(option('scenes', 'monstro'));
const seeds = list(option('seeds', '3JY1 6FLR'));
const framesLimit = Number(option('frames', '900'));
const policy = option('policy', 'mpc');
const repetitions = Number(option('repeat', '1'));
const attackEnabled = !flag('no-fire');
const diagnostics = flag('diagnostics');
const sourceExe = option('source-exe', DEFAULT_SOURCE_EXE);
const readerPath = resolve(option('reader', join(RESEARCH_ROOT, 'external-validation', 'ExternalReader.exe')));
const semanticsPath = resolve(option('semantics', 'D:/Isaac/data/mechanisms/gamebot_runtime_semantics.json'));
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const output = resolve(option('output', join(PACKAGE_ROOT, 'runs', `expert-dataset-${stamp}`)));

for (const scene of scenes) {
  if (!ROOM_SCENARIOS[scene]) throw new Error(`unknown scene ${scene}; known: ${Object.keys(ROOM_SCENARIOS).join(', ')}`);
}
if (!['mpc', 'quick'].includes(policy)) throw new Error('--policy must be mpc or quick');
if (!Number.isInteger(framesLimit) || framesLimit < 1) throw new Error('--frames must be a positive integer');
if (!Number.isInteger(repetitions) || repetitions < 1) throw new Error('--repeat must be a positive integer');
if (!existsSync(readerPath)) throw new Error(`expert reader missing: ${readerPath}`);
if (!existsSync(semanticsPath)) throw new Error(`runtime semantics missing: ${semanticsPath}`);
import {ExpertReader} from './lib/expert_reader.mjs';


function decodeDecision(decision) {
  const usable = decision?.valid === true && decision?.stop !== true;
  const moveIndex = usable ? Number(decision.move ?? 0) : 0;
  const fireIndex = usable ? Number(decision.fire ?? 0) : 0;
  if (!Number.isInteger(moveIndex) || moveIndex < 0 || moveIndex >= MOVES.length) throw new Error(`expert move index out of range: ${decision?.move}`);
  if (!Number.isInteger(fireIndex) || fireIndex < 0 || fireIndex >= FIRES.length) throw new Error(`expert fire index out of range: ${decision?.fire}`);
  return {
    moveIndex,
    fireIndex,
    action: {move: MOVES[moveIndex], fire: attackEnabled ? FIRES[fireIndex] : 'none'},
  };
}

function enemyHpTotal(observation) {
  let total = 0;
  for (const entity of observation?.entities ?? []) {
    if (entity.type < 10 || entity.type >= 1000 || entity.friendly) continue;
    total += Math.max(0, Number(entity.hp) || 0);
  }
  return total;
}

function observationDigest(observation) {
  return sha256Text(JSON.stringify(observation));
}

function formatSeed(seed) { return seed.replace(/\s+/g, ''); }

const provenance = {
  package: 'gamebot-training-engine-v0.1',
  policy,
  reader_path: readerPath,
  reader_sha256: sha256File(readerPath),
  semantics_path: semanticsPath,
  semantics_sha256: sha256File(semanticsPath),
  worker_sha256: sha256File(join(PACKAGE_ROOT, 'src', 'engine_worker.mjs')),
  worker_path: join(PACKAGE_ROOT, 'src', 'engine_worker.mjs'),
  environment_sha256: sha256File(join(PACKAGE_ROOT, 'src', 'environment.mjs')),
  boot_wasm_sha256: sha256File(join(RESEARCH_ROOT, 'the-browsing-of-isaac', 'output', 'recomp', 'lift', 'boot-training-fast', 'boot.wasm')),
  boot_wasm_path: join(RESEARCH_ROOT, 'the-browsing-of-isaac', 'output', 'recomp', 'lift', 'boot-training-fast', 'boot.wasm'),
  source_game_exe: sourceExe,
  source_game_exe_sha256: sha256File(sourceExe),
  research_root: RESEARCH_ROOT,
};

const manifest = {
  format: 'gamebot-expert-dataset/1',
  created_at: new Date().toISOString(),
  status: 'running',
  policy,
  expert: {
    implementation: policy === 'mpc' ? 'gamebot::MpcCombatPlanner via RealtimeMpcController' : 'gamebot::Planner (production defaults)',
    bridge: 'external-validation/reader_main.cpp native Reader over ReadProcessMemory into the training WASM heap',
    // The offline logic clock is 30 Hz, so the MPC clock and horizon are stated
    // in native frames. The wall-clock deadline is disabled for reproducibility;
    // beam width and the simulation cap keep their production defaults.
    mpc_clock: policy === 'mpc' ? {dt: 1 / 30, actual_hz: 30, horizon_frames: 48, min_horizon_frames: 24,
      segment_frames: 6, time_budget_ms: 0, max_total_ms: 0, beam_width: 'default', max_simulations: 'default'} : null,
    attack_enabled: attackEnabled,
    action_space: {move: MOVES, fire: FIRES},
  },
  engine: {logic_hz: 30, action_repeat: 1, frames_per_episode_limit: framesLimit, observations: 'gamebot-room-observation/1'},
  provenance,
  plan: [],
  episodes: [],
};

function writeManifest() {
  writeFileSync(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
}

// A dataset outlives the binaries that produced it, and rebuilding the expert
// changes its hash. Archive the small provenance files inside the dataset so
// the recorded hashes stay verifiable after any later rebuild or source edit.
function archiveProvenance() {
  const directory = join(output, 'provenance');
  mkdirSync(directory, {recursive: true});
  const candidates = [
    ['ExternalReader.exe', readerPath],
    ['engine_worker.mjs', join(PACKAGE_ROOT, 'src', 'engine_worker.mjs')],
    ['environment.mjs', join(PACKAGE_ROOT, 'src', 'environment.mjs')],
    ['collect_expert_dataset.mjs', fileURLToPath(import.meta.url)],
  ];
  const addressPath = join(RESEARCH_ROOT, 'external-validation', 'address.mjs');
  if (existsSync(addressPath)) candidates.push(['address.mjs', addressPath]);
  const archived = [];
  for (const [name, source] of candidates) {
    if (!existsSync(source)) continue;
    const target = join(directory, name);
    copyFileSync(source, target);
    archived.push({name, source, archived: target, sha256: sha256File(target)});
  }
  manifest.provenance.archive_dir = directory;
  manifest.provenance.archived = archived;
}

function planEpisodes() {
  let serial = 0;
  for (const scene of scenes) {
    for (const seed of seeds) {
      for (let repeat = 0; repeat < repetitions; repeat += 1) {
        manifest.plan.push({episode: `${scene}-${formatSeed(seed)}-${repeat}`, scene, seed, repeat, serial: serial++});
      }
    }
  }
}

async function runEpisode(env, reader, plan) {
  const episode = {
    format: 'gamebot-expert-episode/1',
    episode: plan.episode,
    scene: plan.scene,
    scene_label: ROOM_SCENARIOS[plan.scene].label,
    seed: plan.seed,
    policy,
    attack_enabled: attackEnabled,
    started_at: new Date().toISOString(),
    status: 'running',
    frames: 0,
    frames_limit: framesLimit,
  };
  const directory = join(output, 'episodes', plan.episode);
  mkdirSync(directory, {recursive: true});
  const samplesPath = join(directory, 'samples.jsonl');
  const diagnosticsPath = join(directory, 'diagnostics.jsonl');
  writeFileSync(samplesPath, '');
  if (diagnostics) writeFileSync(diagnosticsPath, '');
  episode.files = diagnostics ? {samples: 'samples.jsonl', diagnostics: 'diagnostics.jsonl'} : {samples: 'samples.jsonl'};
  let observation;
  try {
    observation = await env.reset({scene: plan.scene, seed: plan.seed});
    const initialDigest = observationDigest(observation);
    let address = await env.request('externalAddress');
    episode.process = {pid: address.pid, host_base: address.host_base, length: address.length};
    const initialHp = enemyHpTotal(observation);
    const metrics = {
      health_start: observation.player.health,
      health_final: observation.player.health,
      damage_taken: 0,
      hurt_events: 0,
      enemy_hp_start: initialHp,
      enemy_hp_final: initialHp,
      enemy_damage: 0,
      shots: 0,
      stop_actions: 0,
      hit_frames: 0,
      clear: false,
      dead: false,
      seconds: 0,
      dps: 0,
      shots_per_second: 0,
      damage_per_shot: 0,
    };
    const expert = {
      fallbacks: 0, overrides: 0, reused: 0, timed_out: 0, tier_b: 0, emergency: 0,
      safety_stop: 0, invalid: 0, mpc_ms_total: 0, mpc_ms_max: 0,
      projectile_spawns: 0, projectile_hits: 0, projectile_misses: 0, attributed_damage: 0,
      action_histogram: {move: Object.fromEntries(MOVES.map(name => [name, 0])),
        fire: Object.fromEntries(FIRES.map(name => [name, 0]))},
    };
    let actionHash = createHash('sha256');
    let observationHash = createHash('sha256');
    let lastClock = null;
    const started = performance.now();
    for (let step = 0; step < framesLimit; step += 1) {
      if (step % 30 === 0) {
        // The Wasm heap can move when it grows; refresh the read address. If a
        // refresh fails, the previous address stays valid for the same heap.
        try { address = await env.request('externalAddress'); }
        catch (error) { if (step === 0) throw error; }
      }
      const frame = await reader.capture({
        pid: address.pid, host_base: address.host_base, length: address.length,
        source_exe: sourceExe, policy, diagnostics,
        attack_enabled: attackEnabled, reset: step === 0,
      });
      if (!frame.valid) throw new Error(`expert frame invalid at step ${step}: ${frame.error ?? 'unknown'}`);
      if (frame.clock !== observation.frame) {
        throw new Error(`time alignment failed at step ${step}: expert clock ${frame.clock}, observation frame ${observation.frame}`);
      }
      if (frame.pid !== address.pid) throw new Error(`expert read process ${frame.pid}, expected ${address.pid}`);
      const player = observation.player.position;
      if (Math.abs(frame.player_pos.x - player.x) > 1e-3 || Math.abs(frame.player_pos.y - player.y) > 1e-3) {
        throw new Error(`state alignment failed at step ${step}: reader player ${JSON.stringify(frame.player_pos)}, observation ${JSON.stringify(player)}`);
      }
      const {moveIndex, fireIndex, action} = decodeDecision(frame.decision);
      const healthBefore = observation.player.health;
      const hpBefore = enemyHpTotal(observation);
      const result = await env.step(action);
      const next = result.observation;
      const healthAfter = next.player.health;
      const hpAfter = enemyHpTotal(next);
      const outcome = {
        health_before: healthBefore,
        health_after: healthAfter,
        health_delta: healthBefore - healthAfter,
        enemy_hp_before: hpBefore,
        enemy_hp_after: hpAfter,
        enemy_damage: Math.max(0, hpBefore - hpAfter),
        reward: result.reward,
        reward_components: result.info?.rewardComponents ?? null,
        terminated: result.terminated === true,
        truncated: result.truncated === true,
      };
      const mpc = frame.mpc ?? null;
      const progress = frame.closed_loop?.progress ?? null;
      const obsDigest = observationDigest(observation);
      const nextDigest = observationDigest(next);
      const sample = {
        v: 1,
        episode: plan.episode,
        step,
        frame: observation.frame,
        next_frame: next.frame,
        obs: observation,
        obs_sha256: obsDigest,
        next_obs_sha256: nextDigest,
        action,
        action_index: {move: moveIndex, fire: fireIndex},
        expert: {
          policy,
          clock: frame.clock,
          valid: frame.decision.valid === true,
          stop: frame.decision.stop === true,
          move: frame.decision.move,
          fire: frame.decision.fire,
          target: frame.decision.target ?? 0,
          reason: frame.decision.reason ?? '',
          planning_ms: frame.decision.planning_ms ?? null,
          available: mpc?.available ?? null,
          timed_out: mpc?.timed_out ?? null,
          used_fallback: mpc?.used_fallback ?? null,
          overridden: mpc?.overridden ?? null,
          reused: mpc?.reused ?? null,
          emergency: mpc?.emergency ?? null,
          value: mpc?.value ?? null,
          baseline_value: mpc?.baseline_value ?? null,
          stats: mpc?.stats ?? null,
          safety: mpc?.safety ?? null,
          combat: mpc?.combat ?? null,
          shot: mpc?.shot ?? null,
          sequence: mpc?.sequence ?? null,
          arbitration: mpc?.arbitration ?? null,
          progress,
          closed_loop: frame.closed_loop ?? null,
        },
        outcome,
      };
      appendFileSync(samplesPath, `${JSON.stringify(sample)}\n`);
      if (diagnostics) appendFileSync(diagnosticsPath, `${JSON.stringify(frame)}\n`);
      actionHash.update(`${moveIndex}:${fireIndex}|`);
      observationHash.update(`${nextDigest}|`);
      metrics.shots += action.fire === 'none' ? 0 : 1;
      metrics.stop_actions += action.move === 'none' && action.fire === 'none' ? 1 : 0;
      metrics.hit_frames += outcome.enemy_damage > 0 ? 1 : 0;
      metrics.damage_taken += outcome.health_delta;
      metrics.hurt_events += outcome.health_delta > 0 ? 1 : 0;
      expert.action_histogram.move[action.move] += 1;
      expert.action_histogram.fire[action.fire] += 1;
      if (mpc) {
        expert.fallbacks += mpc.used_fallback ? 1 : 0;
        expert.overrides += mpc.overridden ? 1 : 0;
        expert.reused += mpc.reused ? 1 : 0;
        expert.timed_out += mpc.timed_out ? 1 : 0;
        expert.tier_b += mpc.stats?.tier_b_used ? 1 : 0;
        expert.emergency += mpc.emergency ? 1 : 0;
        const mpcMs = frame.closed_loop?.timings?.mpc_ms ?? 0;
        expert.mpc_ms_total += mpcMs;
        expert.mpc_ms_max = Math.max(expert.mpc_ms_max, mpcMs);
      }
      expert.safety_stop += frame.closed_loop?.safety_stop ? 1 : 0;
      expert.invalid += frame.decision.valid === true ? 0 : 1;
      if (progress) {
        expert.projectile_spawns = Math.max(expert.projectile_spawns, Number(progress.projectile_spawns) || 0);
        expert.projectile_hits = Math.max(expert.projectile_hits, Number(progress.projectile_hits) || 0);
        expert.projectile_misses = Math.max(expert.projectile_misses, Number(progress.projectile_misses) || 0);
        expert.attributed_damage = Math.max(expert.attributed_damage, Number(progress.attributed_damage) || 0);
      }
      lastClock = next.frame;
      // Enemies can spawn during a fight (Peep's eyes, Bloat's flies), so the
      // boss's total hp is not a damage ledger. Damage dealt is the sum of the
      // per-frame positive hp drops, exactly like the environment's reward.
      metrics.enemy_damage += outcome.enemy_damage;
      metrics.enemy_hp_final = hpAfter;
      metrics.health_final = healthAfter;
      observation = next;
      episode.frames = step + 1;
      if (result.terminated || result.truncated) {
        metrics.dead = next.terminal?.dead === true;
        metrics.clear = next.terminal?.cleared === true;
        break;
      }
    }
    metrics.seconds = Number((episode.frames / 30).toFixed(3));
    metrics.dps = metrics.seconds > 0 ? Number((metrics.enemy_damage / metrics.seconds).toFixed(4)) : 0;
    metrics.shots_per_second = metrics.seconds > 0 ? Number((metrics.shots / metrics.seconds).toFixed(4)) : 0;
    metrics.damage_per_shot = metrics.shots > 0 ? Number((metrics.enemy_damage / metrics.shots).toFixed(4)) : 0;
    expert.mpc_ms_mean = episode.frames > 0 ? Number((expert.mpc_ms_total / episode.frames).toFixed(4)) : 0;
    expert.mpc_ms_total = Number(expert.mpc_ms_total.toFixed(3));
    expert.mpc_ms_max = Number(expert.mpc_ms_max.toFixed(3));
    episode.status = 'complete';
    episode.end_reason = metrics.clear ? 'clear' : metrics.dead ? 'death' : 'frame_limit';
    episode.quality = metrics.clear ? 'success' : metrics.dead ? 'death' : 'timeout';
    episode.bc_weight = metrics.clear ? 1 : metrics.dead ? 0.25 : 0.6;
    episode.metrics = metrics;
    episode.expert_stats = expert;
    episode.final_frame = lastClock;
    episode.wall_seconds = Number(((performance.now() - started) / 1000).toFixed(2));
    episode.integrity = {
      initial_observation_sha256: initialDigest,
      observation_chain_sha256: observationHash.digest('hex'),
      action_sequence_sha256: actionHash.digest('hex'),
      final_observation_sha256: observationDigest(observation),
    };
    episode.checks = {
      frames_written: episode.frames,
      clock_alignment_failures: 0,
      state_alignment_failures: 0,
      health_accounting_ok: metrics.damage_taken === Math.max(0, metrics.health_start - metrics.health_final),
      enemy_hp_delta: Number((metrics.enemy_hp_start - metrics.enemy_hp_final).toFixed(4)),
      enemy_spawned_hp: Number(Math.max(0, metrics.enemy_hp_final - metrics.enemy_hp_start).toFixed(4)),
    };
  } catch (error) {
    episode.status = 'error';
    episode.error = error.stack ?? String(error);
    episode.error_message = String(error.message ?? error).split('\n')[0];
    episode.frames = episode.frames ?? 0;
  }
  episode.finished_at = new Date().toISOString();
  writeFileSync(join(directory, 'episode.json'), `${JSON.stringify(episode, null, 2)}\n`);
  appendFileSync(join(output, 'index.jsonl'), `${JSON.stringify({
    episode: episode.episode, scene: episode.scene, seed: episode.seed, status: episode.status,
    quality: episode.quality ?? null, frames: episode.frames, end_reason: episode.end_reason ?? null,
    error: episode.error_message ?? null, metrics: episode.metrics ?? null,
    integrity: episode.integrity ?? null,
  })}\n`);
  return episode;
}

async function main() {
  mkdirSync(join(output, 'episodes'), {recursive: true});
  mkdirSync(join(output, 'engine-runs'), {recursive: true});
  planEpisodes();
  manifest.inputs = {scenes, seeds, repetitions, frames: framesLimit, policy, attack_enabled: attackEnabled};
  archiveProvenance();
  writeManifest();
  let env = new GamebotTrainingEnvironment({
    actionRepeat: 1,
    maxSteps: framesLimit + 64,
    externalValidation: true,
    researchRoot: RESEARCH_ROOT,
    runRoot: join(output, 'engine-runs'),
    requestTimeoutMs: 180000,
  });
  let reader = new ExpertReader({exe: readerPath, semantics: semanticsPath, logPath: join(output, 'reader.log')});
  let failures = 0;
  try {
    for (const plan of manifest.plan) {
      const episode = await runEpisode(env, reader, plan);
      manifest.episodes.push({
        episode: episode.episode, scene: episode.scene, seed: episode.seed,
        status: episode.status, quality: episode.quality ?? null,
        frames: episode.frames, end_reason: episode.end_reason ?? null,
        error: episode.error_message ?? null, integrity: episode.integrity ?? null,
        metrics: episode.metrics ?? null,
      });
      writeManifest();
      console.log(JSON.stringify({
        episode: episode.episode, scene: episode.scene, seed: episode.seed, status: episode.status,
        quality: episode.quality ?? null, frames: episode.frames, end: episode.end_reason ?? null,
        damage_taken: episode.metrics?.damage_taken ?? null,
        enemy_damage: episode.metrics?.enemy_damage ?? null,
        dps: episode.metrics?.dps ?? null,
        wall_s: episode.wall_seconds ?? null,
        error: episode.error_message ?? null,
      }));
      if (episode.status === 'error') {
        failures += 1;
        // A failed episode can leave the engine or reader in an unknown state.
        // Recreate both so later episodes are not contaminated by the failure.
        await reader.close();
        await env.close();
        env = new GamebotTrainingEnvironment({
          actionRepeat: 1, maxSteps: framesLimit + 64, externalValidation: true,
          researchRoot: RESEARCH_ROOT, runRoot: join(output, 'engine-runs'), requestTimeoutMs: 180000,
        });
        reader = new ExpertReader({exe: readerPath, semantics: semanticsPath, logPath: join(output, 'reader.log')});
      }
    }
  } finally {
    await reader.close();
    await env.close();
  }
  const complete = manifest.episodes.filter(entry => entry.status === 'complete');
  manifest.status = 'complete';
  manifest.finished_at = new Date().toISOString();
  manifest.summary = {
    episodes: manifest.episodes.length,
    complete: complete.length,
    errors: failures,
    frames: complete.reduce((total, entry) => total + entry.frames, 0),
    clears: manifest.episodes.filter(entry => entry.quality === 'success').length,
    deaths: manifest.episodes.filter(entry => entry.quality === 'death').length,
    timeouts: manifest.episodes.filter(entry => entry.quality === 'timeout').length,
    damage_taken: manifest.episodes.reduce((total, entry) => total + (entry.metrics?.damage_taken ?? 0), 0),
    enemy_damage: manifest.episodes.reduce((total, entry) => total + (entry.metrics?.enemy_damage ?? 0), 0),
  };
  writeManifest();
  console.log(JSON.stringify({output, summary: manifest.summary}));
  if (failures) process.exitCode = 1;
}

await main();
