import { GamebotTrainingEnvironment } from '../src/environment.mjs';
import { PPOTrainer } from '../src/ppo_trainer.mjs';
import { dirname, resolve } from 'node:path';
import { readFile } from 'node:fs/promises';

function parseArgs(args) {
  const options = {};
  for (let index = 0; index < args.length; index++) {
    const argument = args[index];
    if (argument === '--help' || argument === '-h') {
      options.help = true;
      continue;
    }
    if (!argument.startsWith('--')) throw new TypeError(`unexpected argument: ${argument}`);
    const equal = argument.indexOf('=');
    const key = argument.slice(2, equal < 0 ? undefined : equal);
    const value = equal >= 0 ? argument.slice(equal + 1) : args[++index];
    if (value === undefined || value.startsWith('--')) throw new TypeError(`missing value for --${key}`);
    if (Object.hasOwn(options, key)) throw new TypeError(`duplicate option: --${key}`);
    options[key] = value;
  }
  return options;
}

function integerOption(options, key, fallback, minimum = 1) {
  if (options[key] === undefined) return fallback;
  const value = Number(options[key]);
  if (!Number.isInteger(value) || value < minimum) throw new RangeError(`--${key} must be an integer >= ${minimum}`);
  return value;
}

function numberOption(options, key, fallback, minimum = 0) {
  if (options[key] === undefined) return fallback;
  const value = Number(options[key]);
  if (!Number.isFinite(value) || value < minimum) throw new RangeError(`--${key} must be a number >= ${minimum}`);
  return value;
}

const HELP = `Gamebot PPO trainer

Usage:
  npm run train -- [options]

Options:
  --iterations N          PPO iterations to run (default: 10)
  --rollout-steps N       decisions collected per iteration (default: 1024)
  --batch-size N          PPO minibatch size (default: 256)
  --epochs N              PPO update epochs (default: 4)
  --learning-rate N       Adam learning rate (default: 0.0003)
  --gamma N               reward discount (default: 0.999)
  --gae-lambda N          GAE lambda (default: 0.95)
  --clip-range N          PPO clip range (default: 0.2)
  --entropy N             entropy coefficient (default: 0.005)
  --action-repeat N       engine frames per decision, 1..16 (default: 4)
  --max-steps N           native frame limit per episode (default: 5000)
  --scene NAME            room name or auto (default: auto)
  --checkpoint-dir PATH   checkpoint directory (default: runs/ppo-checkpoints)
  --resume PATH           resume latest-training.json from PATH
`;

let environment;
try {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(HELP);
  } else {
    const iterations = integerOption(args, 'iterations', 10);
    const resumePayload = args.resume
      ? JSON.parse(await readFile(resolve(args.resume), 'utf8'))
      : null;
    const actionRepeat = integerOption(args, 'action-repeat', resumePayload?.trainer?.environment?.actionRepeat ?? 4, 1);
    if (actionRepeat > 16) throw new RangeError('--action-repeat must be <= 16');
    const maxSteps = integerOption(args, 'max-steps', resumePayload?.trainer?.environment?.maxSteps ?? 5000);
    const checkpointDirectory = args['checkpoint-dir']
      ?? (args.resume ? dirname(resolve(args.resume)) : 'runs/ppo-checkpoints');
    const trainerConfig = {
      rolloutSteps: integerOption(args, 'rollout-steps', 1024),
      minibatchSize: integerOption(args, 'batch-size', 256),
      updateEpochs: integerOption(args, 'epochs', 4),
      learningRate: numberOption(args, 'learning-rate', 3e-4),
      gamma: numberOption(args, 'gamma', 0.999),
      gaeLambda: numberOption(args, 'gae-lambda', 0.95),
      clipRange: numberOption(args, 'clip-range', 0.2),
      entropyCoefficient: numberOption(args, 'entropy', 0.005),
    };
    const knownOptions = new Set([
      'iterations', 'action-repeat', 'max-steps', 'checkpoint-dir', 'resume', 'scene',
      'rollout-steps', 'batch-size', 'epochs', 'learning-rate', 'gamma', 'gae-lambda',
      'clip-range', 'entropy',
    ]);
    const unknown = Object.keys(args).filter(key => !knownOptions.has(key));
    if (unknown.length) throw new TypeError(`unknown option: --${unknown[0]}`);

    environment = new GamebotTrainingEnvironment({ actionRepeat, maxSteps });
    const trainer = new PPOTrainer({
      environment,
      checkpointDirectory,
      resetOptions: { scene: args.scene ?? 'auto' },
      config: trainerConfig,
    });
    if (args.resume) {
      const optionToConfig = {
        'rollout-steps': 'rolloutSteps',
        'batch-size': 'minibatchSize',
        epochs: 'updateEpochs',
        'learning-rate': 'learningRate',
        gamma: 'gamma',
        'gae-lambda': 'gaeLambda',
        'clip-range': 'clipRange',
        entropy: 'entropyCoefficient',
      };
      const configOverrides = Object.fromEntries(Object.entries(optionToConfig)
        .filter(([option]) => Object.hasOwn(args, option))
        .map(([option, key]) => [key, trainerConfig[key]]));
      const resetOptionsOverride = Object.hasOwn(args, 'scene') ? { scene: args.scene } : undefined;
      const restored = await trainer.restoreCheckpoint(args.resume, { configOverrides, resetOptionsOverride });
      process.stdout.write(`Resumed ${JSON.stringify(restored)}\n`);
    }
    process.stdout.write(`PPO config ${JSON.stringify({ ...trainer.config, iterations, actionRepeat, maxSteps, scene: trainer.resetOptions.scene ?? 'auto' })}\n`);
    await trainer.train(iterations, {
      onIteration(summary) {
        process.stdout.write(`${JSON.stringify(summary)}\n`);
      },
    });
  }
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
} finally {
  await environment?.close();
}

