// Phase 2 — Behavior Cloning on the Phase 1 expert dataset.
//
// Reuses the existing ActorCriticNetwork (same encoder, same heads, same Adam
// state) and trains only the two action heads with cross-entropy on MPC labels.
// Splits are by battle seed; the same seed never appears in two splits.
//
// Usage:
//   node scripts/train_bc.mjs \
//     --train-datasets=D:/Isaac/validation/expert-dataset-20261103,D:/Isaac/validation/expert-dataset-long-20261103 \
//     --train-seeds='3JY1 6FLR' --val-dataset=D:/Isaac/validation/expert-dataset-20261103 \
//     --val-seeds='7W3L P1Z9' --test-dataset=D:/Isaac/validation/expert-dataset-heldout-20261103 \
//     --test-seeds='SWGF G7WV'
import {existsSync, mkdirSync, writeFileSync, readFileSync, copyFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {ActorCriticNetwork} from '../src/actor_critic.mjs';
import {
  loadExpertDataset, labelStats, evaluateActionRows, subsample, datasetSeeds,
} from './lib/expert_dataset.mjs';

function option(name, fallback) {
  const prefix = `--${name}=`;
  const value = process.argv.find(argument => argument.startsWith(prefix));
  return value === undefined ? fallback : value.slice(prefix.length);
}
function list(value) {
  return String(value ?? '').split(',').map(entry => entry.trim()).filter(Boolean);
}
function flag(name) { return process.argv.includes(`--${name}`); }

const trainDatasets = list(option('train-datasets', ''));
if (!trainDatasets.length) throw new Error('--train-datasets=<dir[,dir]> is required');
const trainSeeds = list(option('train-seeds', ''));
const valDataset = option('val-dataset', '');
const valSeeds = list(option('val-seeds', ''));
const testDataset = option('test-dataset', '');
const testSeeds = list(option('test-seeds', ''));
const epochs = Number(option('epochs', '8'));
const batchSize = Number(option('batch-size', '256'));
const learningRate = Number(option('learning-rate', '0.001'));
const entropyCoefficient = Number(option('entropy', '0'));
const valueCoefficient = Number(option('value-coeff', '0'));
const contextDropout = Number(option('context-dropout', '0'));
const seed = Number(option('seed', '42'));
const trainMetricSamples = Number(option('train-metric-samples', '2048'));
const maxTrainSamples = Number(option('max-train-samples', 'Infinity'));
const stride = Number(option('stride', '1'));
const resumePath = option('resume', '');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const output = resolve(option('output', join('training-results', `bc-${stamp}`)));
if (!Number.isInteger(epochs) || epochs < 1) throw new Error('--epochs must be a positive integer');
if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error('--batch-size must be a positive integer');
if (!Number.isFinite(learningRate) || learningRate <= 0) throw new Error('--learning-rate must be positive');
if (!Number.isFinite(seed)) throw new Error('--seed must be a number');
if (!Number.isInteger(stride) || stride < 1) throw new Error('--stride must be a positive integer');
if (!Number.isFinite(contextDropout) || contextDropout < 0 || contextDropout > 1) {
  throw new Error('--context-dropout must be between 0 and 1');
}
if (!existsSync(output)) mkdirSync(output, {recursive: true});

const trainRows = [];
const trainSources = [];
for (const dataset of trainDatasets) {
  const loaded = await loadExpertDataset({dataset, seeds: trainSeeds.length ? trainSeeds : null, stride,
    maxSamples: Number.isFinite(maxTrainSamples) ? Math.max(0, maxTrainSamples - trainRows.length) : Infinity});
  trainRows.push(...loaded.rows);
  trainSources.push(loaded.stats);
  if (trainRows.length >= maxTrainSamples) break;
}
if (!trainRows.length) throw new Error('no training rows matched the requested seeds/datasets');
const resolvedTrainSeeds = trainSeeds.length ? trainSeeds : [...new Set(trainRows.map(row => row.seed))];

const valRows = valDataset
  ? (await loadExpertDataset({dataset: valDataset, seeds: valSeeds.length ? valSeeds : null, stride})).rows
  : [];
const testRows = testDataset
  ? (await loadExpertDataset({dataset: testDataset, seeds: testSeeds.length ? testSeeds : null, stride})).rows
  : [];
if (valDataset && !valRows.length) throw new Error('validation dataset produced no rows');
if (testDataset && !testRows.length) throw new Error('test dataset produced no rows');

// Guard against seed leakage: a seed in train must not appear in val/test.
for (const [name, rows] of [['validation', valRows], ['test', testRows]]) {
  const overlap = [...new Set(rows.map(row => row.seed))].filter(seedValue => resolvedTrainSeeds.includes(seedValue));
  if (overlap.length) throw new Error(`seed leakage: ${name} shares ${overlap.join(', ')} with training seeds`);
}

const config = {
  epochs, batchSize, learningRate, entropyCoefficient, valueCoefficient, seed, stride, contextDropout,
  train_datasets: trainDatasets, train_seeds: resolvedTrainSeeds,
  val_dataset: valDataset || null, val_seeds: valSeeds,
  test_dataset: testDataset || null, test_seeds: testSeeds,
};
const datasetInfo = {
  train: {sources: trainSources, rows: trainRows.length, label_stats: labelStats(trainRows)},
  val: {dataset: valDataset || null, rows: valRows.length, label_stats: labelStats(valRows)},
  test: {dataset: testDataset || null, rows: testRows.length, label_stats: labelStats(testRows)},
};

// Inverse-frequency class weights for the two heads. The expert labels are
// dominated by `move: none` (~22%) and `fire: none` is rare; without this the
// network collapses toward standing still, which looks fine on teacher-forced
// accuracy but dies immediately in the engine.
const classBalance = option('class-balance', 'true') !== 'false';
function inverseFrequency(counts) {
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  const weights = Object.values(counts).map(count => count > 0 ? total / (Object.keys(counts).length * count) : 0);
  return weights;
}
const classWeights = classBalance ? {
  move: inverseFrequency(datasetInfo.train.label_stats.move_counts),
  fire: inverseFrequency(datasetInfo.train.label_stats.fire_counts),
} : {move: null, fire: null};
config.class_balance = classBalance;
config.class_weights = classWeights;

let policy;
let startEpoch = 0;
let best = null;
const history = [];
if (resumePath) {
  if (!existsSync(resumePath)) throw new Error(`resume checkpoint missing: ${resumePath}`);
  const checkpoint = JSON.parse(readFileSync(resumePath, 'utf8'));
  if (!checkpoint.model) throw new Error('resume checkpoint has no model');
  policy = ActorCriticNetwork.fromJSON(checkpoint.model);
  startEpoch = checkpoint.trainer?.epoch ?? 0;
  best = checkpoint.trainer?.best ?? null;
  if (Array.isArray(checkpoint.trainer?.history)) history.push(...checkpoint.trainer.history);
  // Keep the best-val weights with the resumed run so `best_test` stays valid.
  const resumeBest = join(dirname(resumePath), 'best-policy.json');
  if (best && existsSync(resumeBest)) copyFileSync(resumeBest, join(output, 'best-policy.json'));
  console.log(JSON.stringify({resumed: resumePath, epoch: startEpoch, best}));
} else {
  policy = new ActorCriticNetwork({seed});
}

writeFileSync(join(output, 'dataset-summary.json'), `${JSON.stringify(datasetInfo, null, 2)}\n`);
console.log(JSON.stringify({train_rows: trainRows.length, val_rows: valRows.length, test_rows: testRows.length,
  train_seeds: resolvedTrainSeeds, val_seeds: [...new Set(valRows.map(row => row.seed))],
  test_seeds: [...new Set(testRows.map(row => row.seed))]}));

const metricsSubset = subsample(trainRows, trainMetricSamples, seed + 1);
const curve = [];
let lastMetrics = null;

function writeCheckpoint(epoch, bestRecord) {
  const checkpoint = {
    format: 'gamebot-bc-checkpoint/1',
    createdAt: new Date().toISOString(),
    config,
    dataset: datasetInfo,
    model: policy.toJSON(),
    trainer: {epoch, best: bestRecord, history},
  };
  writeFileSync(join(output, 'latest-checkpoint.json'), JSON.stringify(checkpoint));
  return checkpoint;
}

for (let epoch = startEpoch + 1; epoch <= startEpoch + epochs; epoch++) {
  const started = performance.now();
  const trainMetrics = policy.updateSupervised(trainRows, {
    epochs: 1, batchSize, learningRate, entropyCoefficient, valueCoefficient,
    moveClassWeights: classWeights.move, fireClassWeights: classWeights.fire,
  });
  const trainEval = evaluateActionRows(policy, metricsSubset);
  const valEval = valRows.length ? evaluateActionRows(policy, valRows) : null;
  lastMetrics = {
    epoch,
    train_loss: trainMetrics,
    train_eval: trainEval,
    val_eval: valEval,
    wall_seconds: Number(((performance.now() - started) / 1000).toFixed(2)),
  };
  curve.push(lastMetrics);
  if (valEval && (!best || valEval.jointAccuracy > best.val_joint_accuracy)) {
    best = {
      epoch,
      val_move_accuracy: valEval.moveAccuracy,
      val_fire_accuracy: valEval.fireAccuracy,
      val_joint_accuracy: valEval.jointAccuracy,
      val_move_loss: valEval.moveLoss,
      val_fire_loss: valEval.fireLoss,
    };
    writeFileSync(join(output, 'best-policy.json'), policy.save());
  }
  writeCheckpoint(epoch, best);
  writeFileSync(join(output, 'training-curve.json'), `${JSON.stringify(curve, null, 2)}\n`);
  writeFileSync(join(output, 'training-curve.csv'),
    `epoch,train_move_loss,train_fire_loss,train_move_acc,train_fire_acc,train_joint_acc,` +
    `train_eval_move_acc,train_eval_fire_acc,train_eval_joint_acc,val_move_loss,val_fire_loss,val_move_acc,val_fire_acc,val_joint_acc,` +
    `val_move_entropy,val_fire_entropy,wall_seconds\n` +
    curve.map(row => [
      row.epoch, row.train_loss.moveLoss, row.train_loss.fireLoss, row.train_loss.moveAccuracy,
      row.train_loss.fireAccuracy, row.train_loss.jointAccuracy,
      row.train_eval.moveAccuracy, row.train_eval.fireAccuracy, row.train_eval.jointAccuracy,
      row.val_eval?.moveLoss ?? '', row.val_eval?.fireLoss ?? '', row.val_eval?.moveAccuracy ?? '',
      row.val_eval?.fireAccuracy ?? '', row.val_eval?.jointAccuracy ?? '',
      row.val_eval?.moveEntropy ?? '', row.val_eval?.fireEntropy ?? '', row.wall_seconds,
    ].join(',')).join('\n') + '\n');
  console.log(JSON.stringify({
    epoch, wall_s: lastMetrics.wall_seconds,
    train_move_loss: Number(trainMetrics.moveLoss.toFixed(4)),
    train_fire_loss: Number(trainMetrics.fireLoss.toFixed(4)),
    train_joint_acc: Number(trainMetrics.jointAccuracy.toFixed(4)),
    val_move_acc: valEval ? Number(valEval.moveAccuracy.toFixed(4)) : null,
    val_fire_acc: valEval ? Number(valEval.fireAccuracy.toFixed(4)) : null,
    val_joint_acc: valEval ? Number(valEval.jointAccuracy.toFixed(4)) : null,
    val_move_bal: valEval ? Number(valEval.moveBalancedAccuracy.toFixed(4)) : null,
    val_fire_bal: valEval ? Number(valEval.fireBalancedAccuracy.toFixed(4)) : null,
    val_move_loss: valEval ? Number(valEval.moveLoss.toFixed(4)) : null,
    val_fire_loss: valEval ? Number(valEval.fireLoss.toFixed(4)) : null,
  }));
}

const testEval = testRows.length ? evaluateActionRows(policy, testRows) : null;
const bestTestEval = (() => {
  if (!testRows.length || !existsSync(join(output, 'best-policy.json'))) return null;
  const bestPolicy = ActorCriticNetwork.fromJSON(readFileSync(join(output, 'best-policy.json'), 'utf8'));
  return evaluateActionRows(bestPolicy, testRows);
})();
const final = {
  format: 'gamebot-bc-run/1',
  createdAt: new Date().toISOString(),
  config,
  dataset: datasetInfo,
  final_epoch: startEpoch + epochs,
  final_train: lastMetrics,
  test: testEval,
  best,
  best_test: bestTestEval,
  notes: 'Teacher-forced action metrics only; closed-loop evaluation is separate.',
};
writeFileSync(join(output, 'run-summary.json'), `${JSON.stringify(final, null, 2)}\n`);
writeFileSync(join(output, 'summary.md'), [
  '# Behavior Cloning run', '',
  `- train rows: ${trainRows.length} (seeds ${resolvedTrainSeeds.join(', ')})`,
  `- val rows: ${valRows.length} (seeds ${[...new Set(valRows.map(row => row.seed))].join(', ') || '-'})`,
  `- test rows: ${testRows.length} (seeds ${[...new Set(testRows.map(row => row.seed))].join(', ') || '-'})`,
  `- config: epochs=${epochs} batch=${batchSize} lr=${learningRate} entropy=${entropyCoefficient} value_coeff=${valueCoefficient}`,
  '', '| epoch | train move CE | train fire CE | val move CE | val fire CE | val move acc | val fire acc | val joint acc |',
  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  ...curve.map(row => `| ${row.epoch} | ${row.train_loss.moveLoss.toFixed(4)} | ${row.train_loss.fireLoss.toFixed(4)} | ` +
    `${row.val_eval ? row.val_eval.moveLoss.toFixed(4) : ''} | ${row.val_eval ? row.val_eval.fireLoss.toFixed(4) : ''} | ` +
    `${row.val_eval ? row.val_eval.moveAccuracy.toFixed(4) : ''} | ${row.val_eval ? row.val_eval.fireAccuracy.toFixed(4) : ''} | ` +
    `${row.val_eval ? row.val_eval.jointAccuracy.toFixed(4) : ''} |`),
  '', testEval ? `- test (final epoch): move acc ${testEval.moveAccuracy.toFixed(4)}, fire acc ${testEval.fireAccuracy.toFixed(4)}, joint acc ${testEval.jointAccuracy.toFixed(4)}` : '',
  bestTestEval ? `- test (best val checkpoint): move acc ${bestTestEval.moveAccuracy.toFixed(4)}, fire acc ${bestTestEval.fireAccuracy.toFixed(4)}, joint acc ${bestTestEval.jointAccuracy.toFixed(4)}` : '',
  '', `Baselines: train joint majority ${datasetInfo.train.label_stats.joint_majority_baseline.toFixed(4)}, ` +
    `val joint majority ${datasetInfo.val.label_stats.joint_majority_baseline.toFixed(4)}.`,
  '',
  'Teacher-forced accuracy measures imitation, not combat skill. Closed-loop evaluation is the next step.',
].join('\n') + '\n');
console.log(JSON.stringify({output, best, test: testEval, best_test: bestTestEval,
  baselines: {train: datasetInfo.train.label_stats.joint_majority_baseline,
    val: datasetInfo.val.label_stats.joint_majority_baseline}}));
