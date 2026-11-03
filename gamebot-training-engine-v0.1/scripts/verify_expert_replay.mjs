// Phase 1 fixed-seed replay verification.
//
// Rebuilds each recorded episode from its seed, replays the recorded action
// sequence one native frame at a time, and compares the resulting observation
// hashes with the ones captured during collection. A mismatch is reported at
// the first divergent frame; that is a hard failure for the dataset.
//
// Usage: node scripts/verify_expert_replay.mjs --dataset=<run dir> [--episodes=all]
import {createHash} from 'node:crypto';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {GamebotTrainingEnvironment, ROOM_SCENARIOS} from '../src/environment.mjs';

function option(name, fallback) {
  const prefix = `--${name}=`;
  const value = process.argv.find(argument => argument.startsWith(prefix));
  return value === undefined ? fallback : value.slice(prefix.length);
}
const dataset = resolve(option('dataset', ''));
if (!option('dataset', '')) throw new Error('--dataset=<directory> is required');
const manifestPath = join(dataset, 'manifest.json');
if (!existsSync(manifestPath)) throw new Error(`dataset manifest missing: ${manifestPath}`);
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const selection = option('episodes', 'all');
const wanted = selection === 'all' ? null : new Set(selection.split(',').map(entry => entry.trim()).filter(Boolean));
const researchRoot = resolve(manifest.provenance?.research_root ?? process.env.GAMEBOT_RESEARCH_ROOT ?? 'D:\\Isaac\\research');

const digest = observation => createHash('sha256').update(JSON.stringify(observation)).digest('hex');
const report = {
  dataset,
  format: 'gamebot-expert-replay-check/1',
  checked_at: new Date().toISOString(),
  policy: manifest.policy,
  episodes: [],
  summary: {},
};

const env = new GamebotTrainingEnvironment({
  actionRepeat: 1,
  maxSteps: (manifest.engine?.frames_per_episode_limit ?? 900) + 64,
  researchRoot,
  runRoot: join(dataset, 'replay-engine-runs'),
  requestTimeoutMs: 180000,
});

let failures = 0;
try {
  for (const planned of manifest.plan ?? []) {
    if (wanted && !wanted.has(planned.episode)) continue;
    const entryPath = join(dataset, 'episodes', planned.episode, 'episode.json');
    const samplesPath = join(dataset, 'episodes', planned.episode, 'samples.jsonl');
    const result = {episode: planned.episode, scene: planned.scene, seed: planned.seed, status: 'running'};
    report.episodes.push(result);
    if (!existsSync(entryPath) || !existsSync(samplesPath)) {
      result.status = 'missing';
      failures += 1;
      continue;
    }
    const episode = JSON.parse(readFileSync(entryPath, 'utf8'));
    if (episode.status !== 'complete') {
      result.status = `skipped-${episode.status}`;
      continue;
    }
    const samples = readFileSync(samplesPath, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
    if (!ROOM_SCENARIOS[planned.scene]) throw new Error(`unknown scene ${planned.scene}`);
    try {
      const initial = await env.reset({scene: planned.scene, seed: planned.seed});
      const initialDigest = digest(initial);
      result.initial_match = initialDigest === samples[0]?.obs_sha256;
      result.frames = samples.length;
      let matched = 0;
      let divergence = null;
      for (const sample of samples) {
        const step = await env.step(sample.action);
        const actual = digest(step.observation);
        if (actual !== sample.next_obs_sha256) {
          divergence = {
            step: sample.step,
            frame: sample.frame,
            expected_hash: sample.next_obs_sha256,
            actual_hash: actual,
            expected: {health: sample.outcome?.health_after, enemy_hp: sample.outcome?.enemy_hp_after},
            actual: {
              health: step.observation?.player?.health,
              enemy_hp: (step.observation?.entities ?? []).filter(entity =>
                entity.type >= 10 && entity.type < 1000 && !entity.friendly)
                .reduce((total, entity) => total + Math.max(0, Number(entity.hp) || 0), 0),
              frame: step.observation?.frame,
            },
          };
          break;
        }
        matched += 1;
      }
      result.matched_frames = matched;
      result.divergence = divergence;
      result.status = divergence || !result.initial_match ? 'diverged' : 'match';
      if (result.status !== 'match') failures += 1;
    } catch (error) {
      result.status = 'error';
      result.error = String(error.message ?? error).split('\n')[0];
      failures += 1;
    }
  }
} finally {
  await env.close();
}

report.summary = {
  episodes: report.episodes.length,
  matched: report.episodes.filter(entry => entry.status === 'match').length,
  diverged: report.episodes.filter(entry => entry.status === 'diverged').length,
  errors: report.episodes.filter(entry => entry.status === 'error').length,
  skipped: report.episodes.filter(entry => entry.status?.startsWith('skipped')).length,
  frames_replayed: report.episodes.reduce((total, entry) => total + (entry.matched_frames ?? 0), 0),
};
report.passed = failures === 0;
const reportPath = join(dataset, 'replay-report.json');
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({dataset, passed: report.passed, summary: report.summary,
  episodes: report.episodes.map(entry => ({episode: entry.episode, status: entry.status,
    frames: entry.frames ?? null, matched: entry.matched_frames ?? null,
    divergence: entry.divergence?.step ?? null, error: entry.error ?? null})), report: reportPath}));
if (!report.passed) process.exitCode = 1;
