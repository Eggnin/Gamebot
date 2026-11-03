// Reader for the Phase 1 expert dataset. Encodes each observation with the same
// `encodeObservation` the Actor-Critic uses, so training never re-implements
// the observation contract.
import {createReadStream, existsSync, readFileSync} from 'node:fs';
import {createInterface} from 'node:readline';
import {join} from 'node:path';
import {encodeObservation, MOVE_ACTIONS, FIRE_ACTIONS} from '../../src/actor_critic.mjs';

export function datasetManifest(dataset) {
  const manifestPath = join(dataset, 'manifest.json');
  if (!existsSync(manifestPath)) throw new Error(`dataset manifest missing: ${manifestPath}`);
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (manifest.format !== 'gamebot-expert-dataset/1') throw new Error(`unexpected dataset format ${manifest.format}`);
  return manifest;
}

export function datasetSeeds(dataset) {
  const manifest = datasetManifest(dataset);
  return [...new Set(manifest.plan.map(plan => plan.seed))];
}

export function datasetScenes(dataset) {
  const manifest = datasetManifest(dataset);
  return [...new Set(manifest.plan.map(plan => plan.scene))];
}

/**
 * Load rows as `{encoded, action, weight, seed, scene, quality, episode, frame}`.
 * `encoded` is the fixed-size network input; observations themselves are not
 * retained so a 15k-frame dataset fits in a few tens of MB.
 */
export async function loadExpertDataset({dataset, seeds = null, scenes = null, maxSamples = Infinity, stride = 1, context = true}) {
  if (!Number.isInteger(stride) || stride < 1) throw new Error('stride must be a positive integer');
  const manifest = datasetManifest(dataset);
  const stats = {
    dataset, policy: manifest.policy, manifest_created_at: manifest.created_at,
    episodes: 0, samples: 0, skipped_episodes: 0, max_frame: 0,
    by_seed: {}, by_scene: {}, by_quality: {},
  };
  const rows = [];
  for (const plan of manifest.plan) {
    if (seeds && !seeds.includes(plan.seed)) continue;
    if (scenes && !scenes.includes(plan.scene)) continue;
    const directory = join(dataset, 'episodes', plan.episode);
    const episodePath = join(directory, 'episode.json');
    const samplesPath = join(directory, 'samples.jsonl');
    if (!existsSync(episodePath) || !existsSync(samplesPath)) { stats.skipped_episodes += 1; continue; }
    const episode = JSON.parse(readFileSync(episodePath, 'utf8'));
    if (episode.status !== 'complete') { stats.skipped_episodes += 1; continue; }
    const weight = Number.isFinite(episode.bc_weight)
      ? episode.bc_weight
      : (episode.quality === 'success' ? 1 : episode.quality === 'death' ? 0.25 : 0.6);
    const stream = createInterface({input: createReadStream(samplesPath, 'utf8'), crlfDelay: Infinity});
    let previousAction = null;
    for await (const line of stream) {
      if (!line) continue;
      const sample = JSON.parse(line);
      if (stride > 1 && sample.step % stride !== 0) { previousAction = sample.action; continue; }
      rows.push({
        encoded: encodeObservation(sample.obs, context ? previousAction : null),
        action: {move: sample.action.move, fire: sample.action.fire},
        weight,
        seed: plan.seed,
        scene: plan.scene,
        quality: episode.quality,
        episode: plan.episode,
        frame: sample.frame,
      });
      stats.samples += 1;
      stats.max_frame = Math.max(stats.max_frame, sample.frame);
      stats.by_seed[plan.seed] = (stats.by_seed[plan.seed] ?? 0) + 1;
      stats.by_scene[plan.scene] = (stats.by_scene[plan.scene] ?? 0) + 1;
      stats.by_quality[episode.quality] = (stats.by_quality[episode.quality] ?? 0) + 1;
      previousAction = sample.action;
      if (rows.length >= maxSamples) break;
    }
    stats.episodes += 1;
    if (rows.length >= maxSamples) break;
  }
  return {rows, stats};
}

/** Label histograms and the trivial majority-class baseline for each head. */
export function labelStats(rows) {
  const moveCounts = Object.fromEntries(MOVE_ACTIONS.map(name => [name, 0]));
  const fireCounts = Object.fromEntries(FIRE_ACTIONS.map(name => [name, 0]));
  for (const row of rows) {
    moveCounts[row.action.move] += 1;
    fireCounts[row.action.fire] += 1;
  }
  const total = Math.max(rows.length, 1);
  const majority = counts => Math.max(...Object.values(counts)) / total;
  return {
    samples: rows.length,
    move_counts: moveCounts,
    fire_counts: fireCounts,
    move_majority_baseline: majority(moveCounts),
    fire_majority_baseline: majority(fireCounts),
    joint_majority_baseline: (Math.max(...Object.values(moveCounts)) / total) * (Math.max(...Object.values(fireCounts)) / total),
  };
}

/** Teacher-forced action metrics, computed on pre-encoded rows. */
export function evaluateActionRows(policy, rows) {
  if (!rows.length) return {samples: 0, moveLoss: null, fireLoss: null, moveAccuracy: null,
    fireAccuracy: null, jointAccuracy: null, moveEntropy: null, fireEntropy: null,
    moveBalancedAccuracy: null, fireBalancedAccuracy: null};
  let moveLoss = 0;
  let fireLoss = 0;
  let moveCorrect = 0;
  let fireCorrect = 0;
  let jointCorrect = 0;
  let moveEntropy = 0;
  let fireEntropy = 0;
  let totalWeight = 0;
  const moveClassTotal = new Array(MOVE_ACTIONS.length).fill(0);
  const moveClassCorrect = new Array(MOVE_ACTIONS.length).fill(0);
  const fireClassTotal = new Array(FIRE_ACTIONS.length).fill(0);
  const fireClassCorrect = new Array(FIRE_ACTIONS.length).fill(0);
  for (const row of rows) {
    const {moveProbabilities, fireProbabilities} = policy.scoreEncoded(row.encoded);
    const moveIndex = MOVE_ACTIONS.indexOf(row.action.move);
    const fireIndex = FIRE_ACTIONS.indexOf(row.action.fire);
    if (moveIndex < 0 || fireIndex < 0) throw new Error(`invalid label ${JSON.stringify(row.action)}`);
    const moveClipped = Math.max(moveProbabilities[moveIndex], 1e-12);
    const fireClipped = Math.max(fireProbabilities[fireIndex], 1e-12);
    moveLoss += -Math.log(moveClipped) * row.weight;
    fireLoss += -Math.log(fireClipped) * row.weight;
    totalWeight += row.weight;
    let bestMove = 0;
    for (let index = 1; index < moveProbabilities.length; index++) if (moveProbabilities[index] > moveProbabilities[bestMove]) bestMove = index;
    let bestFire = 0;
    for (let index = 1; index < fireProbabilities.length; index++) if (fireProbabilities[index] > fireProbabilities[bestFire]) bestFire = index;
    moveCorrect += bestMove === moveIndex ? 1 : 0;
    fireCorrect += bestFire === fireIndex ? 1 : 0;
    jointCorrect += bestMove === moveIndex && bestFire === fireIndex ? 1 : 0;
    moveClassTotal[moveIndex] += 1;
    moveClassCorrect[moveIndex] += bestMove === moveIndex ? 1 : 0;
    fireClassTotal[fireIndex] += 1;
    fireClassCorrect[fireIndex] += bestFire === fireIndex ? 1 : 0;
    for (const probability of moveProbabilities) if (probability > 0) moveEntropy -= probability * Math.log(probability);
    for (const probability of fireProbabilities) if (probability > 0) fireEntropy -= probability * Math.log(probability);
  }
  const macroRecall = (totals, corrects) => {
    const present = totals.map((total, index) => total > 0 ? corrects[index] / total : null).filter(value => value !== null);
    return present.length ? present.reduce((sum, value) => sum + value, 0) / present.length : 0;
  };
  return {
    samples: rows.length,
    moveLoss: moveLoss / Math.max(totalWeight, 1e-9),
    fireLoss: fireLoss / Math.max(totalWeight, 1e-9),
    moveAccuracy: moveCorrect / rows.length,
    fireAccuracy: fireCorrect / rows.length,
    jointAccuracy: jointCorrect / rows.length,
    moveBalancedAccuracy: macroRecall(moveClassTotal, moveClassCorrect),
    fireBalancedAccuracy: macroRecall(fireClassTotal, fireClassCorrect),
    moveClassRecall: moveClassCorrect.map((correct, index) => moveClassTotal[index] ? correct / moveClassTotal[index] : null),
    fireClassRecall: fireClassCorrect.map((correct, index) => fireClassTotal[index] ? correct / fireClassTotal[index] : null),
    moveEntropy: moveEntropy / rows.length,
    fireEntropy: fireEntropy / rows.length,
  };
}

/** Deterministic subsample for cheap per-epoch training metrics. */
export function subsample(rows, count, seed = 1) {
  if (!Number.isFinite(count) || count >= rows.length) return rows;
  const selected = [];
  let state = seed >>> 0;
  const chosen = new Set();
  while (selected.length < count) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const index = state % rows.length;
    if (chosen.has(index)) continue;
    chosen.add(index);
    selected.push(rows[index]);
  }
  return selected;
}
