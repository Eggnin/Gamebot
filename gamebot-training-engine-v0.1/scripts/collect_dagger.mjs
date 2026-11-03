// Phase 2b — DAgger-style aggregation.
//
// Runs the current learned policy in the offline engine and asks the MPC expert
// for the correct action at every state the policy actually visits. The result
// is a normal expert dataset (same schema as Phase 1) that can be merged into
// BC training, which addresses the distribution shift of pure behavior cloning.
//
// Usage:
//   node scripts/collect_dagger.mjs --policy=.../best-policy.json \
//     --scenes=monstro,theBloat --seeds='3JY1 6FLR' --frames=900 --output=...
import {createHash} from 'node:crypto';
import {appendFileSync, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {GamebotTrainingEnvironment, ROOM_SCENARIOS} from '../src/environment.mjs';
import {ActorCriticNetwork} from '../src/actor_critic.mjs';
import {ExpertReader} from './lib/expert_reader.mjs';

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RESEARCH_ROOT = resolve(process.env.GAMEBOT_RESEARCH_ROOT ?? 'D:\\Isaac\\research');
const MOVES = ['none', 'up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left'];
const FIRES = ['none', 'up', 'right', 'down', 'left'];

function option(name, fallback) {
  const prefix = `--${name}=`;
  const value = process.argv.find(argument => argument.startsWith(prefix));
  return value === undefined ? fallback : value.slice(prefix.length);
}
function list(value) {
  return String(value ?? '').split(',').map(entry => entry.trim()).filter(Boolean);
}
const sha256File = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const sha256Text = text => createHash('sha256').update(text).digest('hex');

const policyPath = resolve(option('policy', ''));
if (!option('policy', '')) throw new Error('--policy=<policy.json or checkpoint.json> is required');
if (!existsSync(policyPath)) throw new Error(`policy file missing: ${policyPath}`);
const raw = JSON.parse(readFileSync(policyPath, 'utf8'));
const policy = ActorCriticNetwork.fromJSON(raw.model ?? raw);
const scenes = list(option('scenes', 'monstro'));
const seeds = list(option('seeds', '3JY1 6FLR'));
const framesLimit = Number(option('frames', '900'));
const sourceExe = option('source-exe', 'D:/Steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng.exe');
const readerPath = resolve(option('reader', join(RESEARCH_ROOT, 'external-validation', 'ExternalReader.exe')));
const semanticsPath = resolve(option('semantics', 'D:/Isaac/data/mechanisms/gamebot_runtime_semantics.json'));
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const output = resolve(option('output', join(PACKAGE_ROOT, 'runs', `dagger-${stamp}`)));
if (!Number.isInteger(framesLimit) || framesLimit < 1) throw new Error('--frames must be a positive integer');
for (const scene of scenes) if (!ROOM_SCENARIOS[scene]) throw new Error(`unknown scene ${scene}`);
if (!existsSync(readerPath)) throw new Error(`expert reader missing: ${readerPath}`);
mkdirSync(join(output, 'episodes'), {recursive: true});
mkdirSync(join(output, 'engine-runs'), {recursive: true});

function decodeExpert(decision) {
  const usable = decision?.valid === true && decision?.stop !== true;
  const moveIndex = usable ? Number(decision.move ?? 0) : 0;
  const fireIndex = usable ? Number(decision.fire ?? 0) : 0;
  if (!Number.isInteger(moveIndex) || moveIndex < 0 || moveIndex >= MOVES.length) throw new Error(`expert move index out of range: ${decision?.move}`);
  if (!Number.isInteger(fireIndex) || fireIndex < 0 || fireIndex >= FIRES.length) throw new Error(`expert fire index out of range: ${decision?.fire}`);
  return {moveIndex, fireIndex, action: {move: MOVES[moveIndex], fire: FIRES[fireIndex]}};
}
const alive = entity => entity.exists !== false && entity.dead !== true && entity.active !== false;
function enemyHpTotal(observation) {
  let total = 0;
  for (const entity of observation?.entities ?? []) {
    if (entity.type < 10 || entity.type >= 1000 || entity.friendly || !alive(entity)) continue;
    total += Math.max(0, Number(entity.hp) || 0);
  }
  return total;
}
const digest = observation => sha256Text(JSON.stringify(observation));

const provenance = {
  policy_path: policyPath,
  policy_sha256: sha256File(policyPath),
  reader_path: readerPath,
  reader_sha256: sha256File(readerPath),
  semantics_path: semanticsPath,
  semantics_sha256: sha256File(semanticsPath),
  worker_path: join(PACKAGE_ROOT, 'src', 'engine_worker.mjs'),
  worker_sha256: sha256File(join(PACKAGE_ROOT, 'src', 'engine_worker.mjs')),
  environment_sha256: sha256File(join(PACKAGE_ROOT, 'src', 'environment.mjs')),
  boot_wasm_path: join(RESEARCH_ROOT, 'the-browsing-of-isaac', 'output', 'recomp', 'lift', 'boot-training-fast', 'boot.wasm'),
  boot_wasm_sha256: sha256File(join(RESEARCH_ROOT, 'the-browsing-of-isaac', 'output', 'recomp', 'lift', 'boot-training-fast', 'boot.wasm')),
  source_game_exe: sourceExe,
  source_game_exe_sha256: sha256File(sourceExe),
  research_root: RESEARCH_ROOT,
};
const manifest = {
  format: 'gamebot-expert-dataset/1',
  kind: 'dagger',
  created_at: new Date().toISOString(),
  status: 'running',
  policy: 'mpc',
  expert: {
    implementation: 'gamebot::MpcCombatPlanner via RealtimeMpcController, queried on DAgger states',
    bridge: 'external-validation/reader_main.cpp native Reader over ReadProcessMemory',
    attack_enabled: true,
    action_space: {move: MOVES, fire: FIRES},
  },
  engine: {logic_hz: 30, action_repeat: 1, frames_per_episode_limit: framesLimit,
    observations: 'gamebot-room-observation/1'},
  provenance,
  plan: [],
  episodes: [],
};
for (const scene of scenes) for (const seed of seeds) {
  manifest.plan.push({episode: `${scene}-${seed.replace(/\s+/g, '')}-dagger`, scene, seed, repeat: 0});
}
// Archive the small provenance files so the dataset stays verifiable.
const archive = join(output, 'provenance');
mkdirSync(archive, {recursive: true});
const archived = [];
for (const [name, source] of [['ExternalReader.exe', readerPath],
  ['engine_worker.mjs', join(PACKAGE_ROOT, 'src', 'engine_worker.mjs')],
  ['environment.mjs', join(PACKAGE_ROOT, 'src', 'environment.mjs')],
  ['collect_dagger.mjs', fileURLToPath(import.meta.url)],
  ['policy.json', policyPath],
  ['address.mjs', join(RESEARCH_ROOT, 'external-validation', 'address.mjs')]]) {
  if (!existsSync(source)) continue;
  const target = join(archive, name);
  copyFileSync(source, target);
  archived.push({name, source, archived: target, sha256: sha256File(target)});
}
manifest.provenance.archived = archived;
manifest.provenance.archive_dir = archive;
const writeManifest = () => writeFileSync(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
writeManifest();

const env = new GamebotTrainingEnvironment({
  actionRepeat: 1, maxSteps: framesLimit + 64, externalValidation: true,
  researchRoot: RESEARCH_ROOT, runRoot: join(output, 'engine-runs'), requestTimeoutMs: 180000,
});
const reader = new ExpertReader({exe: readerPath, semantics: semanticsPath, logPath: join(output, 'reader.log')});
const totals = {episodes: 0, frames: 0, samples: 0, agreements: 0, expert_labels: 0};
try {
  for (const plan of manifest.plan) {
    const directory = join(output, 'episodes', plan.episode);
    mkdirSync(directory, {recursive: true});
    const samplesPath = join(directory, 'samples.jsonl');
    writeFileSync(samplesPath, '');
    const episode = {
      format: 'gamebot-expert-episode/1', episode: plan.episode, scene: plan.scene,
      scene_label: ROOM_SCENARIOS[plan.scene].label, seed: plan.seed, policy: 'mpc',
      source: 'dagger', started_at: new Date().toISOString(), status: 'running', frames: 0,
    };
    let observation;
    try {
      observation = await env.reset({scene: plan.scene, seed: plan.seed});
      let address = await env.request('externalAddress');
      episode.process = {pid: address.pid, host_base: address.host_base, length: address.length};
      const metrics = {health_start: observation.player.health, health_final: observation.player.health,
        damage_taken: 0, enemy_hp_start: enemyHpTotal(observation), enemy_damage: 0, shots: 0, hit_frames: 0,
        clear: false, dead: false};
      const agreement = {move: 0, fire: 0, joint: 0};
      let samples = 0;
      const initialDigest = digest(observation);
      const observationHash = createHash('sha256');
      const actionHash = createHash('sha256');
      for (let step = 0; step < framesLimit; step += 1) {
        if (step % 30 === 0) {
          try { address = await env.request('externalAddress'); } catch { if (step === 0) throw new Error('address unavailable'); }
        }
        const frame = await reader.capture({
          pid: address.pid, host_base: address.host_base, length: address.length,
          source_exe: sourceExe, policy: 'mpc', attack_enabled: true, diagnostics: false, reset: step === 0,
        });
        if (!frame.valid) throw new Error(`expert frame invalid at step ${step}: ${frame.error ?? 'unknown'}`);
        if (frame.clock !== observation.frame) {
          throw new Error(`time alignment failed at step ${step}: expert ${frame.clock}, observation ${observation.frame}`);
        }
        const {moveIndex, fireIndex, action: expertAction} = decodeExpert(frame.decision);
        const bc = policy.act(observation, {deterministic: true});
        const hpBefore = enemyHpTotal(observation);
        const result = await env.step(bc.action);
        const next = result.observation;
        const hpAfter = enemyHpTotal(next);
        const healthDelta = Math.max(0, observation.player.health - next.player.health);
        const expertMove = moveIndex;
        const expertFire = fireIndex;
        const jointMatch = bc.action.move === expertAction.move && bc.action.fire === expertAction.fire;
        agreement.move += bc.action.move === expertAction.move ? 1 : 0;
        agreement.fire += bc.action.fire === expertAction.fire ? 1 : 0;
        agreement.joint += jointMatch ? 1 : 0;
        const sample = {
          v: 1, episode: plan.episode, step, frame: observation.frame, next_frame: next.frame,
          obs: observation, obs_sha256: digest(observation), next_obs_sha256: digest(next),
          action: expertAction, action_index: {move: expertMove, fire: expertFire},
          bc_action: bc.action,
          agreement: {move: bc.action.move === expertAction.move, fire: bc.action.fire === expertAction.fire, joint: jointMatch},
          expert: {policy: 'mpc-dagger', clock: frame.clock, valid: frame.decision.valid === true,
            stop: frame.decision.stop === true, move: expertMove, fire: expertFire,
            target: frame.decision.target ?? 0, reason: frame.decision.reason ?? '',
            planning_ms: frame.decision.planning_ms ?? null},
          outcome: {health_before: observation.player.health, health_after: next.player.health,
            health_delta: healthDelta, enemy_hp_before: hpBefore, enemy_hp_after: hpAfter,
            enemy_damage: Math.max(0, hpBefore - hpAfter), reward: result.reward,
            terminated: result.terminated === true, truncated: result.truncated === true},
        };
        appendFileSync(samplesPath, `${JSON.stringify(sample)}\n`);
        observationHash.update(`${sample.next_obs_sha256}|`);
        actionHash.update(`${moveIndex}:${fireIndex}|`);
        samples += 1;
        metrics.damage_taken += healthDelta;
        metrics.enemy_damage += Math.max(0, hpBefore - hpAfter);
        metrics.shots += bc.action.fire === 'none' ? 0 : 1;
        metrics.hit_frames += hpAfter < hpBefore ? 1 : 0;
        observation = next;
        episode.frames = step + 1;
        if (result.terminated || result.truncated) {
          metrics.clear = next.terminal?.cleared === true;
          metrics.dead = next.terminal?.dead === true;
          break;
        }
      }
      metrics.health_final = observation.player.health;
      metrics.seconds = Number((episode.frames / 30).toFixed(3));
      metrics.dps = metrics.seconds > 0 ? Number((metrics.enemy_damage / metrics.seconds).toFixed(4)) : 0;
      episode.status = 'complete';
      episode.quality = metrics.clear ? 'success' : metrics.dead ? 'death' : 'timeout';
      episode.bc_weight = metrics.clear ? 1 : metrics.dead ? 0.25 : 0.6;
      episode.end_reason = metrics.clear ? 'clear' : metrics.dead ? 'death' : 'frame_limit';
      episode.metrics = metrics;
      const frames = Math.max(episode.frames, 1);
      episode.expert_stats = {samples, agreement_rate: Number((agreement.joint / frames).toFixed(4)),
        move_agreement_rate: Number((agreement.move / frames).toFixed(4)),
        fire_agreement_rate: Number((agreement.fire / frames).toFixed(4))};
      episode.integrity = {
        initial_observation_sha256: initialDigest,
        observation_chain_sha256: observationHash.digest('hex'),
        action_sequence_sha256: actionHash.digest('hex'),
        final_observation_sha256: digest(observation),
        action_label: 'mpc expert at DAgger state',
      };
      totals.episodes += 1;
      totals.frames += episode.frames;
      totals.samples += samples;
      totals.agreements += agreement.joint;
      totals.expert_labels += samples;
    } catch (error) {
      episode.status = 'error';
      episode.error_message = String(error.message ?? error).split('\n')[0];
      episode.error = error.stack ?? String(error);
    }
    episode.finished_at = new Date().toISOString();
    writeFileSync(join(directory, 'episode.json'), `${JSON.stringify(episode, null, 2)}\n`);
    manifest.episodes.push({episode: episode.episode, scene: episode.scene, seed: episode.seed,
      status: episode.status, quality: episode.quality ?? null, frames: episode.frames,
      expert_stats: episode.expert_stats ?? null, error: episode.error_message ?? null});
    writeManifest();
    console.log(JSON.stringify({episode: episode.episode, status: episode.status,
      quality: episode.quality ?? null, frames: episode.frames,
      agreement: episode.expert_stats?.agreement_rate ?? null,
      damage_taken: episode.metrics?.damage_taken ?? null,
      enemy_damage: episode.metrics?.enemy_damage ?? null, error: episode.error_message ?? null}));
  }
} finally {
  await reader.close();
  await env.close();
}
manifest.status = 'complete';
manifest.finished_at = new Date().toISOString();
manifest.summary = {
  episodes: manifest.episodes.length,
  errors: manifest.episodes.filter(entry => entry.status === 'error').length,
  frames: totals.frames,
  samples: totals.samples,
  expert_labels: totals.expert_labels,
  bc_joint_agreement: totals.frames ? Number((totals.agreements / totals.frames).toFixed(4)) : null,
};
writeManifest();
console.log(JSON.stringify({output, summary: manifest.summary}));
