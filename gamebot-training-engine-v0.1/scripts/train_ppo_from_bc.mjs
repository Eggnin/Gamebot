// Phase 3 — PPO fine-tuning from the Behavior Cloning checkpoint.
//
// Initializes ActorCriticNetwork from `best-policy.json` (never random), keeps
// the existing PPOTrainer/RolloutBuffer/GAE implementation, and only replaces
// the scalar reward with ShapedEnvironment's real-combat shaping.
//
// Usage:
//   node scripts/train_ppo_from_bc.mjs --policy=.../best-policy.json \
//     --seeds='3JY1 6FLR' --iterations=10 --rollout-steps=1024 --output=...
import {createHash} from 'node:crypto';
import {appendFileSync, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {GamebotTrainingEnvironment} from '../src/environment.mjs';
import {ActorCriticNetwork} from '../src/actor_critic.mjs';
import {PPOTrainer} from '../src/ppo_trainer.mjs';
import {ShapedEnvironment, DEFAULT_SHAPING} from './lib/shaped_environment.mjs';

function option(name, fallback) {
  const prefix = `--${name}=`;
  const value = process.argv.find(argument => argument.startsWith(prefix));
  return value === undefined ? fallback : value.slice(prefix.length);
}
function list(value) {
  return String(value ?? '').split(',').map(entry => entry.trim()).filter(Boolean);
}
const sha256File = path => createHash('sha256').update(readFileSync(path)).digest('hex');

const policyPath = resolve(option('policy', ''));
if (!option('policy', '')) throw new Error('--policy=<BC policy.json or checkpoint.json> is required');
if (!existsSync(policyPath)) throw new Error(`policy file missing: ${policyPath}`);
const raw = JSON.parse(readFileSync(policyPath, 'utf8'));
const policy = ActorCriticNetwork.fromJSON(raw.model ?? raw);

const iterations = Number(option('iterations', '10'));
const rolloutSteps = Number(option('rollout-steps', '1024'));
const batchSize = Number(option('batch-size', '256'));
const updateEpochs = Number(option('epochs', '4'));
const learningRate = Number(option('learning-rate', '0.0001'));
const gamma = Number(option('gamma', '0.995'));
const gaeLambda = Number(option('gae-lambda', '0.95'));
const clipRange = Number(option('clip-range', '0.2'));
const entropyCoefficient = Number(option('entropy', '0.002'));
const valueCoefficient = Number(option('value-coefficient', '0.5'));
const maxGradNorm = Number(option('max-grad-norm', '0.5'));
const actionRepeat = Number(option('action-repeat', '1'));
const maxSteps = Number(option('max-steps', '1800'));
const scenes = option('scenes', 'auto');
const seeds = list(option('seeds', '3JY1 6FLR'));
const resumePath = option('resume', '');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const output = resolve(option('output', join('training-results', `ppo-from-bc-${stamp}`)));
if (!Number.isInteger(iterations) || iterations < 1) throw new Error('--iterations must be a positive integer');
if (!Number.isInteger(rolloutSteps) || rolloutSteps < 1) throw new Error('--rollout-steps must be a positive integer');
if (!Number.isInteger(actionRepeat) || actionRepeat < 1 || actionRepeat > 16) throw new Error('--action-repeat must be 1..16');
if (!seeds.length) throw new Error('--seeds must not be empty');
mkdirSync(join(output, 'checkpoints'), {recursive: true});
mkdirSync(join(output, 'engine-runs'), {recursive: true});

const environment = new GamebotTrainingEnvironment({
  actionRepeat, maxSteps, researchRoot: process.env.GAMEBOT_RESEARCH_ROOT,
  runRoot: join(output, 'engine-runs'), requestTimeoutMs: 180000,
});
const shaped = new ShapedEnvironment(environment, {weights: DEFAULT_SHAPING});
const trainer = new PPOTrainer({
  environment: shaped,
  policy,
  checkpointDirectory: join(output, 'checkpoints'),
  resetOptions: {scene: scenes},
  seedSchedule: seeds,
  config: {
    rolloutSteps, minibatchSize: batchSize, updateEpochs,
    learningRate, gamma, gaeLambda, clipRange, entropyCoefficient,
    valueCoefficient, maxGradNorm,
  },
});

const metadata = {
  format: 'gamebot-ppo-from-bc/1',
  createdAt: new Date().toISOString(),
  init_policy: policyPath,
  init_policy_sha256: sha256File(policyPath),
  resumed_from: resumePath || null,
  config: {iterations, rolloutSteps, batchSize, updateEpochs, learningRate, gamma, gaeLambda,
    clipRange, entropyCoefficient, valueCoefficient, maxGradNorm, actionRepeat, maxSteps, scenes, seeds},
  reward_shaping: DEFAULT_SHAPING,
  note: 'Reward uses only real transition data (enemy HP removed, health lost, clear/death) plus anti-stall.',
};
if (resumePath) {
  if (!existsSync(resumePath)) throw new Error(`resume checkpoint missing: ${resumePath}`);
  const restored = await trainer.restoreCheckpoint(resolve(resumePath));
  console.log(JSON.stringify({resumed: restored}));
}
writeFileSync(join(output, 'run-metadata.json'), `${JSON.stringify(metadata, null, 2)}\n`);
console.log(JSON.stringify({init_policy: policyPath, policy_sha256: metadata.init_policy_sha256.slice(0, 12),
  config: metadata.config, shaping: DEFAULT_SHAPING}));

const history = [];
try {
  await trainer.train(iterations, {
    onIteration(summary) {
      const row = {
        iteration: summary.iteration,
        globalSteps: summary.globalSteps,
        globalFrames: summary.globalFrames,
        completedEpisodes: summary.completedEpisodes,
        roomsCleared: summary.roomsCleared,
        deaths: summary.deaths,
        meanRewardPerDecision: summary.rollout.meanRewardPerDecision,
        meanEpisodeReward: summary.rollout.meanEpisodeReward,
        damageDealt: summary.rollout.damageDealt,
        damageTaken: summary.rollout.damageTaken,
        framesAdvanced: summary.rollout.framesAdvanced,
        policyLoss: summary.update.policyLoss,
        valueLoss: summary.update.valueLoss,
        entropy: summary.update.entropy,
        approximateKl: summary.update.approximateKl,
        clipFraction: summary.update.clipFraction,
        explainedVariance: summary.update.explainedVariance,
        bestMeanEpisodeReward: summary.bestMeanEpisodeReward,
        bestPolicySaved: summary.bestPolicySaved,
      };
      history.push(row);
      appendFileSync(join(output, 'iterations.jsonl'), `${JSON.stringify(row)}\n`);
      console.log(JSON.stringify(row));
    },
  });
} finally {
  await shaped.close();
}

writeFileSync(join(output, 'training-history.json'), `${JSON.stringify(history, null, 2)}\n`);
writeFileSync(join(output, 'summary.md'), [
  '# PPO from BC', '',
  `- init policy: ${policyPath}`,
  `- iterations: ${iterations}, rollout: ${rolloutSteps}, action repeat: ${actionRepeat}, max steps: ${maxSteps}`,
  `- lr ${learningRate}, gamma ${gamma}, gae ${gaeLambda}, clip ${clipRange}, entropy ${entropyCoefficient}`,
  `- seeds: ${seeds.join(', ')}, scenes: ${scenes}`,
  '', '| iter | frames | episodes | clears | deaths | mean episode reward | damage dealt | damage taken | policy loss | value loss | explained var |',
  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  ...history.map(row => `| ${row.iteration} | ${row.globalFrames} | ${row.completedEpisodes} | ${row.roomsCleared} | ` +
    `${row.deaths} | ${row.meanEpisodeReward?.toFixed?.(2) ?? ''} | ${row.damageDealt.toFixed(1)} | ` +
    `${row.damageTaken.toFixed(1)} | ${row.policyLoss.toFixed(4)} | ${row.valueLoss.toFixed(4)} | ` +
    `${row.explainedVariance?.toFixed?.(4) ?? ''} |`),
  '', 'Checkpoints: `checkpoints/latest-training.json`, `checkpoints/best-policy.json`.',
].join('\n') + '\n');
console.log(JSON.stringify({output, iterations: history.length,
  checkpoints: {latest: join(output, 'checkpoints', 'latest-training.json'),
    best: join(output, 'checkpoints', 'best-policy.json')}}));
