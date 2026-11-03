// Phase 1 dataset quality gate.
//
// Validates the expert dataset produced by collect_expert_dataset.mjs:
// schema, time alignment, action domain, transition accounting, observation
// continuity, integrity hashes and provenance. Exits non-zero when any error
// is found so it can guard Phase 2 training.
//
// Usage: node scripts/check_expert_dataset.mjs --dataset=<run dir>
import {createHash} from 'node:crypto';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';

const MOVES = ['none', 'up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left'];
const FIRES = ['none', 'up', 'right', 'down', 'left'];

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
if (manifest.format !== 'gamebot-expert-dataset/1') throw new Error(`unexpected dataset format ${manifest.format}`);

const sha256 = text => createHash('sha256').update(text).digest('hex');
const sha256File = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const isFiniteNumber = value => typeof value === 'number' && Number.isFinite(value);
const close = (a, b, tolerance = 1e-6) => Math.abs(a - b) <= tolerance;
const alive = entity => entity.exists !== false && entity.dead !== true && entity.active !== false;

const report = {
  dataset,
  format: 'gamebot-expert-dataset-check/1',
  checked_at: new Date().toISOString(),
  manifest: {
    policy: manifest.policy,
    inputs: manifest.inputs,
    provenance: manifest.provenance,
  },
  provenance_checks: [],
  episodes: [],
  summary: {},
};

// Provenance files must still hash to the recorded values; otherwise the
// dataset can no longer be reproduced from the stated code. Archived copies
// inside the dataset are preferred: rebuilding the expert changes the live
// binary hash but must not invalidate data that was collected before.
const archivedEntries = manifest.provenance?.archived ?? [];
for (const entry of archivedEntries) {
  if (!entry.archived || !entry.sha256) {
    report.provenance_checks.push({name: entry.name ?? 'archived', status: 'missing-record'});
    continue;
  }
  if (!existsSync(entry.archived)) {
    report.provenance_checks.push({name: entry.name, path: entry.archived, status: 'missing-file'});
    continue;
  }
  const actual = sha256File(entry.archived);
  report.provenance_checks.push({name: `${entry.name} (archived)`, path: entry.archived,
    status: actual === entry.sha256 ? 'match' : 'mismatch', expected: entry.sha256, actual});
}
for (const [name, path, expected] of [
  ...(archivedEntries.length ? [] : [
    ['reader', manifest.provenance?.reader_path, manifest.provenance?.reader_sha256],
    ['worker', manifest.provenance?.worker_path, manifest.provenance?.worker_sha256],
  ]),
  ['boot_wasm', manifest.provenance?.boot_wasm_path, manifest.provenance?.boot_wasm_sha256],
  ['source_game_exe', manifest.provenance?.source_game_exe, manifest.provenance?.source_game_exe_sha256],
]) {
  if (!path || !expected) {
    report.provenance_checks.push({name, status: 'missing-record'});
    continue;
  }
  if (!existsSync(path)) {
    report.provenance_checks.push({name, path, status: 'missing-file'});
    continue;
  }
  const actual = sha256File(path);
  report.provenance_checks.push({name, path, status: actual === expected ? 'match' : 'mismatch', expected, actual});
}

let totalSamples = 0;
let totalErrors = 0;
let totalWarnings = 0;

for (const planned of manifest.plan ?? []) {
  const directory = join(dataset, 'episodes', planned.episode);
  const episodePath = join(directory, 'episode.json');
  const samplesPath = join(directory, 'samples.jsonl');
  const result = {
    episode: planned.episode, scene: planned.scene, seed: planned.seed,
    status: 'checked', samples: 0, errors: [], warnings: [],
  };
  const error = message => { result.errors.push(message); totalErrors += 1; };
  const warn = message => { result.warnings.push(message); totalWarnings += 1; };
  if (!existsSync(episodePath)) { error('episode.json missing'); report.episodes.push(result); continue; }
  if (!existsSync(samplesPath)) { error('samples.jsonl missing'); report.episodes.push(result); continue; }
  const episode = JSON.parse(readFileSync(episodePath, 'utf8'));
  result.status = episode.status;
  if (episode.status !== 'complete') error(`episode status ${episode.status}: ${episode.error_message ?? episode.error ?? ''}`);
  if (episode.scene !== planned.scene || episode.seed !== planned.seed) error('episode scene/seed differs from manifest plan');
  const text = readFileSync(samplesPath, 'utf8');
  const lines = text.split('\n').filter(line => line.length > 0);
  result.samples = lines.length;
  totalSamples += lines.length;
  if (lines.length === 0) error('samples.jsonl is empty');

  const samples = [];
  for (let index = 0; index < lines.length; index += 1) {
    let sample;
    try { sample = JSON.parse(lines[index]); }
    catch (parseError) { error(`line ${index + 1}: invalid JSON (${parseError.message})`); continue; }
    samples.push(sample);
    const at = `step ${index}`;
    if (sample.v !== 1) error(`${at}: unsupported sample version ${sample.v}`);
    if (sample.episode !== planned.episode) error(`${at}: episode id mismatch`);
    if (sample.step !== index) error(`${at}: step index ${sample.step} is not ${index}`);
    if (!Number.isInteger(sample.frame) || !Number.isInteger(sample.next_frame)) error(`${at}: frame/next_frame must be integers`);
    else if (sample.next_frame !== sample.frame + 1) error(`${at}: next_frame ${sample.next_frame} does not follow frame ${sample.frame}`);
    if (sample.obs?.frame !== sample.frame) error(`${at}: observation frame ${sample.obs?.frame} differs from sample frame ${sample.frame}`);
    if (sample.expert?.clock !== sample.frame) error(`${at}: expert clock ${sample.expert?.clock} differs from sample frame ${sample.frame} (time alignment)`);
    if (sample.obs_sha256 !== sha256(JSON.stringify(sample.obs))) error(`${at}: observation hash mismatch`);
    if (!MOVES.includes(sample.action?.move)) error(`${at}: invalid move ${sample.action?.move}`);
    if (!FIRES.includes(sample.action?.fire)) error(`${at}: invalid fire ${sample.action?.fire}`);
    const usable = sample.expert?.valid === true && sample.expert?.stop !== true;
    const expectedAction = {
      move: usable ? MOVES[sample.expert.move] : 'none',
      fire: usable ? FIRES[sample.expert.fire] : 'none',
    };
    if (sample.action?.move !== expectedAction.move || sample.action?.fire !== expectedAction.fire) {
      error(`${at}: action ${JSON.stringify(sample.action)} is not the decoded expert decision ${JSON.stringify(expectedAction)}`);
    }
    if (sample.action_index?.move !== MOVES.indexOf(sample.action?.move) ||
        sample.action_index?.fire !== FIRES.indexOf(sample.action?.fire)) {
      error(`${at}: action_index does not match action names`);
    }
    if (sample.obs?.schema !== 'gamebot-room-observation/1') error(`${at}: unexpected observation schema ${sample.obs?.schema}`);
    if (JSON.stringify(sample.obs).includes('"address"')) error(`${at}: observation leaks a memory address`);
    const player = sample.obs?.player;
    if (!isFiniteNumber(player?.position?.x) || !isFiniteNumber(player?.position?.y)) error(`${at}: player position missing or non-finite`);
    if (!isFiniteNumber(player?.velocity?.x) || !isFiniteNumber(player?.velocity?.y)) error(`${at}: player velocity missing or non-finite`);
    if (!Number.isInteger(player?.health)) error(`${at}: player health missing`);
    const width = sample.obs?.room?.width;
    const height = sample.obs?.room?.height;
    if (!Number.isInteger(width) || !Number.isInteger(height)) error(`${at}: room width/height missing`);
    else if (!Array.isArray(sample.obs.grid) || sample.obs.grid.length !== width * height) {
      error(`${at}: grid has ${sample.obs?.grid?.length} cells, expected ${width * height}`);
    }
    for (const entity of sample.obs?.entities ?? []) {
      if (!isFiniteNumber(entity.position?.x) || !isFiniteNumber(entity.position?.y)) error(`${at}: entity position non-finite`);
      if (entity.type >= 10 && entity.type < 1000 && !entity.friendly && alive(entity) && !isFiniteNumber(entity.hp)) {
        error(`${at}: living enemy has no hp`);
      }
    }
    const outcome = sample.outcome;
    if (!outcome) { error(`${at}: outcome missing`); continue; }
    if (outcome.health_before !== player?.health) error(`${at}: outcome health_before disagrees with observation health`);
    if (!Number.isInteger(outcome.health_after)) error(`${at}: outcome health_after missing`);
    if (outcome.health_delta !== outcome.health_before - outcome.health_after) error(`${at}: health delta accounting error`);
    if (!close(outcome.enemy_damage, Math.max(0, outcome.enemy_hp_before - outcome.enemy_hp_after))) {
      error(`${at}: enemy damage accounting error`);
    }
    if (!isFiniteNumber(outcome.reward)) error(`${at}: reward is not finite`);
    if (outcome.reward_components) {
      const components = outcome.reward_components;
      if (!isFiniteNumber(components.enemyDamage) || !isFiniteNumber(components.health) ||
          !isFiniteNumber(components.clear) || !isFiniteNumber(components.death) || !isFiniteNumber(components.time)) {
        error(`${at}: reward components non-finite`);
      }
    }
    if (index < lines.length - 1 && (outcome.terminated || outcome.truncated)) {
      error(`${at}: terminal transition in the middle of the episode`);
    }
  }

  if (samples.length > 1) {
    for (let index = 0; index < samples.length - 1; index += 1) {
      const current = samples[index];
      const next = samples[index + 1];
      const at = `step ${index}`;
      if (current.next_frame !== next.frame) error(`${at}: frame gap ${current.frame} -> ${next.frame}`);
      if (current.next_obs_sha256 !== next.obs_sha256) error(`${at}: observation hash chain break`);
      if (current.outcome?.health_after !== next.outcome?.health_before) error(`${at}: health continuity break`);
      if (!close(current.outcome?.enemy_hp_after ?? 0, next.outcome?.enemy_hp_before ?? 0)) error(`${at}: enemy hp continuity break`);
    }
  }

  const actionHash = createHash('sha256');
  const observationHash = createHash('sha256');
  let damageTaken = 0;
  let enemyDamage = 0;
  for (const sample of samples) {
    actionHash.update(`${sample.action_index?.move}:${sample.action_index?.fire}|`);
    observationHash.update(`${sample.next_obs_sha256}|`);
    damageTaken += sample.outcome?.health_delta ?? 0;
    enemyDamage += sample.outcome?.enemy_damage ?? 0;
  }
  if (episode.frames !== samples.length) error(`episode.frames ${episode.frames} differs from ${samples.length} samples`);
  if (episode.metrics?.damage_taken !== damageTaken) error(`metrics.damage_taken ${episode.metrics?.damage_taken} differs from samples ${damageTaken}`);
  if (!close(episode.metrics?.enemy_damage ?? 0, enemyDamage)) error(`metrics.enemy_damage ${episode.metrics?.enemy_damage} differs from samples ${enemyDamage}`);
  if (episode.metrics?.health_final !== samples.at(-1)?.outcome?.health_after) error('metrics.health_final differs from last transition');
  if (samples[0]?.obs_sha256 !== episode.integrity?.initial_observation_sha256) error('initial observation hash mismatch');
  if (samples.at(-1)?.next_obs_sha256 !== episode.integrity?.final_observation_sha256) error('final observation hash mismatch');
  if (actionHash.digest('hex') !== episode.integrity?.action_sequence_sha256) error('action sequence hash mismatch');
  if (observationHash.digest('hex') !== episode.integrity?.observation_chain_sha256) error('observation chain hash mismatch');
  const hist = episode.expert_stats?.action_histogram;
  if (hist) {
    for (const name of MOVES) {
      const count = samples.filter(sample => sample.action.move === name).length;
      if (hist.move?.[name] !== count) warn(`move histogram ${name}=${hist.move?.[name]} differs from samples ${count}`);
    }
    for (const name of FIRES) {
      const count = samples.filter(sample => sample.action.fire === name).length;
      if (hist.fire?.[name] !== count) warn(`fire histogram ${name}=${hist.fire?.[name]} differs from samples ${count}`);
    }
    const shots = samples.filter(sample => sample.action.fire !== 'none').length;
    if (hist.fire) {
      const histogramShots = FIRES.filter(name => name !== 'none').reduce((sum, name) => sum + (hist.fire[name] ?? 0), 0);
      if (histogramShots !== shots) warn(`expert shot count ${histogramShots} differs from samples ${shots}`);
    }
  }
  result.end_reason = episode.end_reason;
  result.quality = episode.quality;
  result.metrics = episode.metrics;
  report.episodes.push(result);
}

report.summary = {
  episodes: report.episodes.length,
  complete_episodes: report.episodes.filter(entry => entry.status === 'complete').length,
  samples: totalSamples,
  errors: totalErrors,
  warnings: totalWarnings,
  provenance_ok: report.provenance_checks.every(entry => entry.status === 'match'),
  qualities: Object.fromEntries(['success', 'death', 'timeout'].map(quality =>
    [quality, report.episodes.filter(entry => entry.quality === quality).length])),
  frames_per_quality: Object.fromEntries(['success', 'death', 'timeout'].map(quality =>
    [quality, report.episodes.filter(entry => entry.quality === quality).reduce((total, entry) => total + entry.samples, 0)])),
};
report.passed = totalErrors === 0;

const reportPath = join(dataset, 'quality-report.json');
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({
  dataset, passed: report.passed, summary: report.summary,
  provenance: report.provenance_checks.map(entry => ({name: entry.name, status: entry.status})),
  first_errors: report.episodes.flatMap(entry => entry.errors.slice(0, 3)).slice(0, 10),
  report: reportPath,
}));
if (!report.passed) process.exitCode = 1;
if (!report.summary.provenance_ok) process.exitCode = 1;
