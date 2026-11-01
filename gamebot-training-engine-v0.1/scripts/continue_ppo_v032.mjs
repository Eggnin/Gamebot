import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { ActorCriticNetwork } from '../src/actor_critic.mjs';
import { GamebotTrainingEnvironment } from '../src/environment.mjs';
import { PPOTrainer } from '../src/ppo_trainer.mjs';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const firstRun = join(projectRoot, 'training-results', 'ppo-v0.3.2-first-run');
const output = join(projectRoot, 'training-results', 'ppo-v0.3.2-continuation');
const checkpointDirectory = join(output, 'checkpoints');
const startingCheckpoint = join(firstRun, 'checkpoints', 'latest-training.json');
const startingBest = join(firstRun, 'best-heldout-policy.json');
const heldoutSeeds = ['WE4Y Y4DM', 'F2HZ 6ZVD'];
const testScenes = ['gurdyJr', 'larryJr', 'dukeOfFlies', 'theBloat'];
const iterations = 4;
const actionRepeat = 4;
const maxSteps = 600;

function finiteMean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function summarize(episodes) {
  const fires = episodes.reduce((sum, episode) => sum + episode.firingDecisions, 0);
  const damagingFires = episodes.reduce((sum, episode) => sum + episode.damageBearingFireDecisions, 0);
  return {
    episodes: episodes.length,
    roomClearRate: episodes.length ? episodes.filter(episode => episode.cleared).length / episodes.length : null,
    deathRate: episodes.length ? episodes.filter(episode => episode.dead).length / episodes.length : null,
    meanEpisodeReward: finiteMean(episodes.map(episode => episode.reward)),
    meanDamageDealt: finiteMean(episodes.map(episode => episode.damageDealt)),
    meanDamageTaken: finiteMean(episodes.map(episode => episode.damageTaken)),
    damageBearingFireDecisionRate: fires ? damagingFires / fires : null,
  };
}

function better(candidate, incumbent) {
  if (!incumbent) return true;
  const a = [candidate.roomClearRate ?? -1, -(candidate.deathRate ?? Infinity), candidate.meanDamageDealt ?? -1, -(candidate.meanDamageTaken ?? Infinity)];
  const b = [incumbent.roomClearRate ?? -1, -(incumbent.deathRate ?? Infinity), incumbent.meanDamageDealt ?? -1, -(incumbent.meanDamageTaken ?? Infinity)];
  for (let index = 0; index < a.length; index++) {
    if (a[index] > b[index]) return true;
    if (a[index] < b[index]) return false;
  }
  return false;
}

async function atomicJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temp = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temp, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
    await rename(temp, path);
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
}

async function evaluate(environment, policy, label, iteration) {
  const episodes = [];
  for (const seed of heldoutSeeds) {
    for (const scene of testScenes) {
      let observation = await environment.reset({ seed, scene });
      const row = {
        policy: label,
        iteration,
        seed,
        scene,
        reward: 0,
        damageDealt: 0,
        damageTaken: 0,
        frames: 0,
        decisions: 0,
        firingDecisions: 0,
        damageBearingFireDecisions: 0,
        cleared: false,
        dead: false,
        terminated: false,
        truncated: false,
      };
      while (!row.terminated && !row.truncated) {
        const decision = policy.act(observation);
        const result = await environment.step(decision.action);
        const dealt = Number(result.info?.enemyDamage ?? 0);
        row.reward += result.reward;
        row.damageDealt += dealt;
        row.damageTaken += Number(result.info?.healthUnitLost ?? 0);
        row.frames += Number(result.info?.framesAdvanced ?? 0);
        row.decisions++;
        if (decision.action.fire !== 'none') {
          row.firingDecisions++;
          if (dealt > 0) row.damageBearingFireDecisions++;
        }
        observation = result.observation;
        row.terminated = result.terminated === true;
        row.truncated = result.truncated === true;
        row.cleared = observation.terminal?.cleared === true;
        row.dead = observation.terminal?.dead === true;
      }
      row.clearTimeFrames = row.cleared ? row.frames : null;
      row.damageBearingFireDecisionRate = row.firingDecisions
        ? row.damageBearingFireDecisions / row.firingDecisions : null;
      row.actualProjectileHitRate = null;
      episodes.push(row);
    }
  }
  return { summary: summarize(episodes), episodes };
}

let trainingEnvironment;
let evaluationEnvironment;
try {
  await mkdir(output, { recursive: true });
  const originalReport = JSON.parse(await readFile(join(firstRun, 'pilot-report.json'), 'utf8'));
  const originalBestData = JSON.parse(await readFile(startingBest, 'utf8'));
  let incumbentMetrics = originalBestData.selection.heldOutMetrics;
  let incumbentModel = originalBestData;
  let incumbentSelection = originalBestData.selection;
  const initialCheckpointData = JSON.parse(await readFile(startingCheckpoint, 'utf8'));
  const environmentConfig = initialCheckpointData.trainer.environment;
  if (environmentConfig.actionRepeat !== actionRepeat || environmentConfig.maxSteps !== maxSteps) {
    throw new Error('continuation environment settings must match the source checkpoint');
  }

  trainingEnvironment = new GamebotTrainingEnvironment({
    actionRepeat,
    maxSteps,
    runRoot: join(projectRoot, 'runs', 'ppo-v0.3.2-continuation-training'),
  });
  evaluationEnvironment = new GamebotTrainingEnvironment({
    actionRepeat,
    maxSteps,
    runRoot: join(projectRoot, 'runs', 'ppo-v0.3.2-continuation-evaluation'),
  });
  const trainer = new PPOTrainer({
    environment: trainingEnvironment,
    policy: new ActorCriticNetwork(),
    checkpointDirectory,
    resetOptions: { scene: 'auto' },
    config: initialCheckpointData.trainer.config,
  });
  const restored = await trainer.restoreCheckpoint(startingCheckpoint);
  trainer.curriculumStage = 'continued_mixed';

  await atomicJson(join(output, 'metadata.json'), {
    schema: 'gamebot-ppo-continuation/1',
    sourceCheckpoint: startingCheckpoint,
    sourceBestHeldoutPolicy: startingBest,
    startedFrom: restored,
    trainingSeeds: initialCheckpointData.trainer.seedSchedule,
    heldOutTestSeeds: heldoutSeeds,
    heldOutTestScenes: testScenes,
    iterations,
    config: trainer.config,
    changedRewardFunction: false,
    observationOrActionSchemaChanged: false,
  });
  await atomicJson(join(output, 'best-heldout-policy.json'), incumbentModel);

  const rounds = [];
  const seedBest = await evaluate(
    evaluationEnvironment,
    ActorCriticNetwork.fromJSON(trainer.policy.toJSON()),
    'ppo-continuation-start',
    trainer.iteration,
  );
  rounds.push({ iteration: trainer.iteration, stage: 'continuation-start', training: null, evaluation: seedBest.summary });
  if (better(seedBest.summary, incumbentMetrics)) {
    incumbentMetrics = seedBest.summary;
    incumbentModel = trainer.policy.toJSON();
    incumbentSelection = {
      iteration: trainer.iteration,
      stage: 'continuation-start',
      heldOutMetrics: incumbentMetrics,
      testSeeds: heldoutSeeds,
      testScenes,
      lexicographicCriterion: ['roomClearRate (higher)', 'deathRate (lower)', 'meanDamageDealt (higher)', 'meanDamageTaken (lower)'],
    };
    await atomicJson(join(output, 'best-heldout-policy.json'), {
      ...incumbentModel,
      selection: incumbentSelection,
    });
  }
  await atomicJson(join(output, 'heldout-progress.json'), { rounds });
  process.stdout.write(`${JSON.stringify({ event: 'continuation-baseline', iteration: trainer.iteration, evaluation: seedBest.summary })}\n`);

  const curveRows = [];
  for (let index = 0; index < iterations; index++) {
    const trained = await trainer.trainIteration();
    const evaluation = await evaluate(
      evaluationEnvironment,
      ActorCriticNetwork.fromJSON(trainer.policy.toJSON()),
      'ppo-continuation',
      trained.iteration,
    );
    const metrics = {
      iteration: trained.iteration,
      decision: trained.globalSteps,
      episodeReward: trained.rollout.meanEpisodeReward,
      actorLoss: trained.update.policyLoss,
      criticLoss: trained.update.valueLoss,
      policyEntropy: trained.update.entropy,
      approxKl: trained.update.approximateKl,
      clipFraction: trained.update.clipFraction,
      explainedVariance: trained.update.explainedVariance,
      trainDamageDealt: trained.rollout.damageDealt,
      trainDamageTaken: trained.rollout.damageTaken,
      heldoutClearRate: evaluation.summary.roomClearRate,
      heldoutDeathRate: evaluation.summary.deathRate,
      heldoutDamageDealt: evaluation.summary.meanDamageDealt,
      heldoutDamageTaken: evaluation.summary.meanDamageTaken,
      heldoutDamageBearingFireDecisionRate: evaluation.summary.damageBearingFireDecisionRate,
    };
    curveRows.push(metrics);
    rounds.push({
      iteration: trained.iteration,
      stage: 'continued_mixed',
      training: metrics,
      trainingEpisodes: trained.rollout.episodeSummaries,
      evaluation: evaluation.summary,
      heldOutEpisodes: evaluation.episodes,
    });
    if (better(evaluation.summary, incumbentMetrics)) {
      incumbentMetrics = evaluation.summary;
      incumbentModel = trainer.policy.toJSON();
      incumbentSelection = {
        iteration: trained.iteration,
        stage: 'continued_mixed',
        heldOutMetrics: incumbentMetrics,
        testSeeds: heldoutSeeds,
        testScenes,
        lexicographicCriterion: ['roomClearRate (higher)', 'deathRate (lower)', 'meanDamageDealt (higher)', 'meanDamageTaken (lower)'],
      };
      await atomicJson(join(output, 'best-heldout-policy.json'), {
        ...incumbentModel,
        selection: incumbentSelection,
      });
    }
    await atomicJson(join(output, 'heldout-progress.json'), {
      sourceReport: originalReport.results,
      rounds,
      bestHeldoutSelection: incumbentSelection,
    });
    await writeFile(join(output, 'training-curves.csv'), [
      Object.keys(curveRows[0]).join(','),
      ...curveRows.map(row => Object.values(row).join(',')),
    ].join('\n') + '\n', 'utf8');
    process.stdout.write(`${JSON.stringify({ event: 'continuation-round', metrics, evaluation: evaluation.summary, bestHeldoutSelection: incumbentSelection })}\n`);
  }

  process.stdout.write(`${JSON.stringify({ event: 'continuation-complete', output, iterations: rounds.length - 1, bestHeldoutSelection: incumbentSelection })}\n`);
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
} finally {
  await Promise.all([trainingEnvironment?.close(), evaluationEnvironment?.close()]);
}

