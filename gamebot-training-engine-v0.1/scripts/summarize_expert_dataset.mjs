// Phase 1 dataset summary. Reads manifest + per-episode reports and prints the
// tables used in the Phase 1 evidence document. No engine or expert needed.
//
// Usage: node scripts/summarize_expert_dataset.mjs --dataset=<run dir> [--json=out.json]
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';

function option(name, fallback) {
  const prefix = `--${name}=`;
  const value = process.argv.find(argument => argument.startsWith(prefix));
  return value === undefined ? fallback : value.slice(prefix.length);
}
const dataset = resolve(option('dataset', ''));
if (!option('dataset', '')) throw new Error('--dataset=<directory> is required');
const manifest = JSON.parse(readFileSync(join(dataset, 'manifest.json'), 'utf8'));
const qualityPath = join(dataset, 'quality-report.json');
const replayPath = join(dataset, 'replay-report.json');

const episodes = [];
for (const plan of manifest.plan ?? []) {
  const path = join(dataset, 'episodes', plan.episode, 'episode.json');
  if (!existsSync(path)) continue;
  const episode = JSON.parse(readFileSync(path, 'utf8'));
  const seconds = episode.frames / (manifest.engine?.logic_hz ?? 30);
  const estimatedHits = episode.metrics.enemy_damage / 3.5;
  const spawns = episode.expert_stats.projectile_spawns || 0;
  episodes.push({
    episode: episode.episode, scene: episode.scene_label, seed: episode.seed,
    frames: episode.frames, quality: episode.quality, end_reason: episode.end_reason,
    damage_taken: episode.metrics.damage_taken, enemy_damage: episode.metrics.enemy_damage,
    dps: episode.metrics.dps, shots: episode.metrics.shots, spawns,
    estimated_hits: Number(estimatedHits.toFixed(1)),
    estimated_hit_rate: spawns ? Number((estimatedHits / spawns).toFixed(3)) : null,
    fallback_rate: episode.frames ? Number((episode.expert_stats.fallbacks / episode.frames).toFixed(3)) : null,
    override_rate: episode.frames ? Number((episode.expert_stats.overrides / episode.frames).toFixed(3)) : null,
    tier_b_rate: episode.frames ? Number((episode.expert_stats.tier_b / episode.frames).toFixed(3)) : null,
    emergency_rate: episode.frames ? Number((episode.expert_stats.emergency / episode.frames).toFixed(3)) : null,
    mpc_ms_mean: episode.expert_stats.mpc_ms_mean, mpc_ms_max: episode.expert_stats.mpc_ms_max,
    wall_seconds: episode.wall_seconds, seconds: Number(seconds.toFixed(2)),
  });
}

const sum = selector => episodes.reduce((total, episode) => total + (selector(episode) || 0), 0);
const totalFrames = sum(episode => episode.frames);
const totalSeconds = totalFrames / (manifest.engine?.logic_hz ?? 30);
const totalSpawns = sum(episode => episode.spawns);
const totalEnemyDamage = sum(episode => episode.enemy_damage);
const summary = {
  dataset,
  policy: manifest.policy,
  format: manifest.format,
  inputs: manifest.inputs,
  episodes: episodes.length,
  frames: totalFrames,
  logic_seconds: Number(totalSeconds.toFixed(1)),
  clears: episodes.filter(episode => episode.quality === 'success').length,
  deaths: episodes.filter(episode => episode.quality === 'death').length,
  timeouts: episodes.filter(episode => episode.quality === 'timeout').length,
  damage_taken: sum(episode => episode.damage_taken),
  enemy_damage: Number(totalEnemyDamage.toFixed(1)),
  dps: Number((totalEnemyDamage / totalSeconds).toFixed(3)),
  spawns: totalSpawns,
  estimated_hit_rate: totalSpawns ? Number(((totalEnemyDamage / 3.5) / totalSpawns).toFixed(4)) : null,
  wall_seconds: Number(sum(episode => episode.wall_seconds).toFixed(1)),
  fallback_rate: Number((sum(episode => episode.fallback_rate * episode.frames) / totalFrames).toFixed(4)),
  override_rate: Number((sum(episode => episode.override_rate * episode.frames) / totalFrames).toFixed(4)),
  tier_b_rate: Number((sum(episode => episode.tier_b_rate * episode.frames) / totalFrames).toFixed(4)),
  emergency_rate: Number((sum(episode => episode.emergency_rate * episode.frames) / totalFrames).toFixed(4)),
  quality_report: existsSync(qualityPath) ? JSON.parse(readFileSync(qualityPath, 'utf8')).summary : null,
  replay_report: existsSync(replayPath) ? JSON.parse(readFileSync(replayPath, 'utf8')).summary : null,
  episodes_detail: episodes,
};

console.log('| 场景 | 种子 | 帧 | 结果 | 掉血 | 输出 | DPS | 弹幕 | 估算命中率 | 回退% | 覆盖% | MPC 均/最 (ms) | wall s |');
console.log('| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |');
for (const episode of episodes) {
  console.log(`| ${episode.scene} | ${episode.seed} | ${episode.frames} | ${episode.quality}/${episode.end_reason} | ${episode.damage_taken} | ${episode.enemy_damage} | ${episode.dps} | ${episode.spawns} | ${episode.estimated_hit_rate ?? '-'} | ${(episode.fallback_rate * 100).toFixed(1)} | ${(episode.override_rate * 100).toFixed(1)} | ${episode.mpc_ms_mean}/${episode.mpc_ms_max} | ${episode.wall_seconds} |`);
}
console.log(JSON.stringify({
  dataset: summary.dataset, policy: summary.policy, episodes: summary.episodes, frames: summary.frames,
  clears: summary.clears, deaths: summary.deaths, timeouts: summary.timeouts,
  damage_taken: summary.damage_taken, enemy_damage: summary.enemy_damage, dps: summary.dps,
  spawns: summary.spawns, estimated_hit_rate: summary.estimated_hit_rate,
  wall_seconds: summary.wall_seconds, fallback_rate: summary.fallback_rate,
  override_rate: summary.override_rate, tier_b_rate: summary.tier_b_rate,
  emergency_rate: summary.emergency_rate,
  quality: summary.quality_report, replay: summary.replay_report,
}));
const jsonPath = option('json', '');
if (jsonPath) writeFileSync(resolve(jsonPath), `${JSON.stringify(summary, null, 2)}\n`);
