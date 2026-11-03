// Phase 2 — closed-loop evaluation of a trained policy (BC or PPO checkpoint)
// inside the offline J460 engine, with optional comparison against recorded
// expert episodes on the same scene/seed.
//
// Usage:
//   node scripts/evaluate_policy.mjs --policy=.../best-policy.json \
//     --scenes=monstro,theBloat,theCage,gurdyJr --seeds='SWGF G7WV' --frames=900 \
//     --compare-dataset=D:/Isaac/validation/expert-dataset-heldout-20261103 \
//     --output=...
import {existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {GamebotTrainingEnvironment, ROOM_SCENARIOS} from '../src/environment.mjs';
import {ActorCriticNetwork} from '../src/actor_critic.mjs';

function option(name, fallback) {
  const prefix = `--${name}=`;
  const value = process.argv.find(argument => argument.startsWith(prefix));
  return value === undefined ? fallback : value.slice(prefix.length);
}
function list(value) {
  return String(value ?? '').split(',').map(entry => entry.trim()).filter(Boolean);
}
function flag(name) { return process.argv.includes(`--${name}`); }
const sha256File = path => createHash('sha256').update(readFileSync(path)).digest('hex');

const policyPath = resolve(option('policy', ''));
const randomPolicy = flag('random');
if (!randomPolicy && !option('policy', '')) throw new Error('--policy=<policy.json or checkpoint.json> is required (or --random)');
if (!randomPolicy && !existsSync(policyPath)) throw new Error(`policy file missing: ${policyPath}`);
let policy = null;
if (!randomPolicy) {
  const raw = JSON.parse(readFileSync(policyPath, 'utf8'));
  policy = ActorCriticNetwork.fromJSON(raw.model ?? raw);
}
// Seeded uniform-random baseline, used to show that BC is not just noise.
let randomState = Number(option('random-seed', '12345')) >>> 0;
function randomUnit() {
  randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
  return randomState / 4294967296;
}
const policyLabel = option('label', randomPolicy ? 'random' : policyPath);
const scenes = list(option('scenes', 'monstro'));
const seeds = list(option('seeds', 'SWGF G7WV'));
const framesLimit = Number(option('frames', '900'));
const actionRepeat = Number(option('action-repeat', '1'));
if (!Number.isInteger(actionRepeat) || actionRepeat < 1 || actionRepeat > 16) {
  throw new Error('--action-repeat must be an integer from 1 to 16');
}
// Optional deployment wrapper: keep the previous movement direction for at
// least N native frames before accepting a different one. It is not part of the
// network; it only suppresses single-frame direction flapping.
const moveHold = Number(option('move-hold', '0'));
if (!Number.isInteger(moveHold) || moveHold < 0) throw new Error('--move-hold must be a non-negative integer');
const deterministic = !flag('sample');
const compareDataset = option('compare-dataset', '');
const researchRoot = resolve(option('research-root', process.env.GAMEBOT_RESEARCH_ROOT ?? 'D:\\Isaac\\research'));
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const output = resolve(option('output', join('runs', `policy-eval-${stamp}`)));
if (!Number.isInteger(framesLimit) || framesLimit < 1) throw new Error('--frames must be a positive integer');
for (const scene of scenes) if (!ROOM_SCENARIOS[scene]) throw new Error(`unknown scene ${scene}`);
mkdirSync(output, {recursive: true});

const MOVES = ['none', 'up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left'];
const FIRES = ['none', 'up', 'right', 'down', 'left'];
const alive = entity => entity.exists !== false && entity.dead !== true && entity.active !== false;
function enemyHpTotal(observation) {
  let total = 0;
  for (const entity of observation?.entities ?? []) {
    if (entity.type < 10 || entity.type >= 1000 || entity.friendly || !alive(entity)) continue;
    total += Math.max(0, Number(entity.hp) || 0);
  }
  return total;
}

// Optional expert reference: episode metrics recorded by collect_expert_dataset.
const reference = new Map();
if (compareDataset) {
  const manifest = JSON.parse(readFileSync(join(compareDataset, 'manifest.json'), 'utf8'));
  for (const plan of manifest.plan) {
    const path = join(compareDataset, 'episodes', plan.episode, 'episode.json');
    if (!existsSync(path)) continue;
    const episode = JSON.parse(readFileSync(path, 'utf8'));
    reference.set(`${plan.scene}|${plan.seed}`, episode);
  }
}

const report = {
  format: 'gamebot-policy-eval/1',
  started_at: new Date().toISOString(),
  label: policyLabel,
  policy_path: randomPolicy ? null : policyPath,
  policy_sha256: randomPolicy ? null : sha256File(policyPath),
  deterministic,
  action_repeat: actionRepeat,
  move_hold: moveHold,
  frames_limit: framesLimit,
  compare_dataset: compareDataset || null,
  provenance: {environment_sha256: sha256File(join(resolve('.'), 'src', 'environment.mjs')),
    worker_sha256: sha256File(join(resolve('.'), 'src', 'engine_worker.mjs'))},
  episodes: [],
};

const env = new GamebotTrainingEnvironment({
  actionRepeat,
  maxSteps: framesLimit + 64,
  researchRoot,
  runRoot: join(output, 'engine-runs'),
  requestTimeoutMs: 180000,
});

try {
  for (const scene of scenes) {
    for (const seed of seeds) {
      const row = {scene, seed, frames: 0, status: 'running'};
      const started = performance.now();
      try {
        let observation = await env.reset({scene, seed});
        const metrics = {
          health_start: observation.player.health, health_final: observation.player.health,
          damage_taken: 0, hurt_events: 0, enemy_hp_start: enemyHpTotal(observation),
          enemy_hp_final: enemyHpTotal(observation), enemy_damage: 0, shots: 0, hit_frames: 0,
          clear: false, dead: false, seconds: 0, dps: 0,
        };
        const histogram = {move: Object.fromEntries(MOVES.map(name => [name, 0])),
          fire: Object.fromEntries(FIRES.map(name => [name, 0]))};
        let advancedFrames = 0;
        let heldMove = null;
        let framesOnMove = 0;
        let previousAction = {move: 'none', fire: 'none'};
        for (let step = 0; step < framesLimit; step += 1) {
          const decision = randomPolicy
            ? {action: {move: MOVES[Math.floor(randomUnit() * MOVES.length)],
              fire: FIRES[Math.floor(randomUnit() * FIRES.length)]}}
            : policy.act(observation, {deterministic, context: previousAction});
          let move = decision.action.move;
          if (moveHold > 0 && heldMove !== null && move !== heldMove && framesOnMove < moveHold) {
            move = heldMove;
          }
          if (move !== heldMove) { heldMove = move; framesOnMove = 0; }
          const appliedAction = {move, fire: decision.action.fire};
          const hpBefore = enemyHpTotal(observation);
          const result = await env.step(appliedAction);
          const next = result.observation;
          const advanced = result.info?.framesAdvanced ?? actionRepeat;
          advancedFrames += advanced;
          framesOnMove += advanced;
          const hpAfter = enemyHpTotal(next);
          const loss = Math.max(0, observation.player.health - next.player.health);
          metrics.damage_taken += loss;
          metrics.hurt_events += loss > 0 ? 1 : 0;
          metrics.enemy_damage += Math.max(0, hpBefore - hpAfter);
          metrics.shots += appliedAction.fire === 'none' ? 0 : 1;
          metrics.hit_frames += hpAfter < hpBefore ? 1 : 0;
          histogram.move[move] += 1;
          histogram.fire[appliedAction.fire] += 1;
          previousAction = {move, fire: appliedAction.fire};
          observation = next;
          row.frames = advancedFrames;
          if (result.terminated || result.truncated) {
            metrics.clear = next.terminal?.cleared === true;
            metrics.dead = next.terminal?.dead === true;
            break;
          }
          if (advancedFrames >= framesLimit) break;
        }
        metrics.health_final = observation.player.health;
        metrics.enemy_hp_final = enemyHpTotal(observation);
        metrics.seconds = Number((row.frames / 30).toFixed(3));
        metrics.dps = metrics.seconds > 0 ? Number((metrics.enemy_damage / metrics.seconds).toFixed(4)) : 0;
        row.status = 'complete';
        row.quality = metrics.clear ? 'success' : metrics.dead ? 'death' : 'timeout';
        row.metrics = metrics;
        row.action_histogram = histogram;
        row.end_reason = metrics.clear ? 'clear' : metrics.dead ? 'death' : 'frame_limit';
        row.final_health = metrics.health_final;
      } catch (error) {
        row.status = 'error';
        row.error = String(error.message ?? error).split('\n')[0];
      }
      row.wall_seconds = Number(((performance.now() - started) / 1000).toFixed(2));
      const expert = reference.get(`${scene}|${seed}`);
      if (expert) {
        row.reference = {
          episode: expert.episode, quality: expert.quality, frames: expert.frames,
          damage_taken: expert.metrics.damage_taken, enemy_damage: expert.metrics.enemy_damage,
          dps: expert.metrics.dps, damage_per_shot: expert.metrics.damage_per_shot,
          shots: expert.metrics.shots,
        };
      }
      report.episodes.push(row);
      console.log(JSON.stringify({scene, seed, status: row.status, frames: row.frames,
        quality: row.quality ?? null, damage_taken: row.metrics?.damage_taken ?? null,
        enemy_damage: row.metrics?.enemy_damage ?? null, dps: row.metrics?.dps ?? null,
        expert_damage: row.reference?.damage_taken ?? null, expert_enemy_damage: row.reference?.enemy_damage ?? null,
        error: row.error ?? null}));
      writeFileSync(join(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
    }
  }
} finally {
  await env.close();
}

function totals(episodes) {
  const complete = episodes.filter(episode => episode.status === 'complete');
  const frames = complete.reduce((total, episode) => total + episode.frames, 0);
  const seconds = frames / 30;
  const damageTaken = complete.reduce((total, episode) => total + episode.metrics.damage_taken, 0);
  const enemyDamage = complete.reduce((total, episode) => total + episode.metrics.enemy_damage, 0);
  return {
    episodes: complete.length,
    errors: episodes.filter(episode => episode.status === 'error').length,
    frames,
    clears: complete.filter(episode => episode.quality === 'success').length,
    deaths: complete.filter(episode => episode.quality === 'death').length,
    timeouts: complete.filter(episode => episode.quality === 'timeout').length,
    damage_taken: damageTaken,
    enemy_damage: Number(enemyDamage.toFixed(1)),
    dps: seconds > 0 ? Number((enemyDamage / seconds).toFixed(4)) : 0,
    hit_frames: complete.reduce((total, episode) => total + episode.metrics.hit_frames, 0),
    shots: complete.reduce((total, episode) => total + episode.metrics.shots, 0),
    mean_seconds: complete.length ? Number((seconds / complete.length).toFixed(2)) : 0,
  };
}

report.policy_summary = totals(report.episodes);
report.finished_at = new Date().toISOString();
report.reference_summary = compareDataset ? (() => {
  const experts = report.episodes.map(episode => episode.reference).filter(Boolean);
  const frames = experts.reduce((total, expert) => total + expert.frames, 0);
  const seconds = frames / 30;
  const enemyDamage = experts.reduce((total, expert) => total + expert.enemy_damage, 0);
  return {
    episodes: experts.length,
    frames,
    clears: experts.filter(expert => expert.quality === 'success').length,
    deaths: experts.filter(expert => expert.quality === 'death').length,
    damage_taken: experts.reduce((total, expert) => total + expert.damage_taken, 0),
    enemy_damage: Number(enemyDamage.toFixed(1)),
    dps: seconds > 0 ? Number((enemyDamage / seconds).toFixed(4)) : 0,
  };
})() : null;
writeFileSync(join(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({output, policy: report.policy_summary, expert: report.reference_summary}));
