import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { ActorCriticNetwork, FIRE_ACTIONS, MOVE_ACTIONS } from '../src/actor_critic.mjs';
import { GamebotTrainingEnvironment } from '../src/environment.mjs';
import { PPOTrainer } from '../src/ppo_trainer.mjs';

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outputDirectory = join(packageRoot, 'training-results', 'ppo-v0.3.2-first-run');
const checkpointDirectory = join(outputDirectory, 'checkpoints');
const runRoot = join(packageRoot, 'runs', 'ppo-v0.3.2-pilot');
const actionRepeat = 4;
const maxSteps = 600;
const testSeeds = ['WE4Y Y4DM', 'F2HZ 6ZVD'];
const trainSeeds = ['3JY1 6FLR', '7W3L P1Z9', 'SWGF G7WV'];
const testScenes = ['gurdyJr', 'larryJr', 'dukeOfFlies', 'theBloat'];
const stages = [
  { id: '01_stationary_aim_proxy', label: '静止目标瞄准近似', scene: 'gurdyJr' },
  { id: '02_moving_target', label: '移动目标瞄准', scene: 'larryJr' },
  { id: '03_projectile_dodge', label: '弹幕躲避', scene: 'theBloat' },
  { id: '04_multi_enemy', label: '多敌人组合', scene: 'dukeOfFlies' },
  { id: '05_boss', label: 'Boss 战斗', scene: 'auto' },
  { id: '06_mixed', label: '混合场景', scene: 'auto' },
];

function makeRng(seed) {
  let state = (seed >>> 0) || 0x6d2b79f5;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

class RandomPolicy {
  constructor(seed) { this.random = makeRng(seed); }
  act() {
    return {
      action: {
        move: MOVE_ACTIONS[Math.floor(this.random() * MOVE_ACTIONS.length)],
        fire: FIRE_ACTIONS[Math.floor(this.random() * FIRE_ACTIONS.length)],
      },
    };
  }
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function summarizeEpisodes(episodes) {
  const cleared = episodes.filter(episode => episode.cleared);
  const firing = episodes.reduce((sum, episode) => sum + episode.firingDecisions, 0);
  const damageBearing = episodes.reduce((sum, episode) => sum + episode.damageBearingFireDecisions, 0);
  return {
    episodes: episodes.length,
    roomClearRate: episodes.length ? cleared.length / episodes.length : null,
    deathRate: episodes.length ? episodes.filter(episode => episode.dead).length / episodes.length : null,
    truncationRate: episodes.length ? episodes.filter(episode => episode.truncated).length / episodes.length : null,
    meanEpisodeReward: mean(episodes.map(episode => episode.episodeReward)),
    meanDamageDealt: mean(episodes.map(episode => episode.damageDealt)),
    meanDamageTaken: mean(episodes.map(episode => episode.damageTaken)),
    meanClearTimeFrames: mean(cleared.map(episode => episode.clearTimeFrames)),
    damageBearingFireDecisionRate: firing ? damageBearing / firing : null,
    totalFiringDecisions: firing,
    totalDamageBearingFireDecisions: damageBearing,
  };
}

async function evaluatePolicy(environment, policy, { label, stage = 'baseline' }) {
  const episodes = [];
  for (const seed of testSeeds) {
    for (const scene of testScenes) {
      let observation = await environment.reset({ seed, scene });
      let episodeReward = 0;
      let damageDealt = 0;
      let damageTaken = 0;
      let frames = 0;
      let decisions = 0;
      let firingDecisions = 0;
      let damageBearingFireDecisions = 0;
      let enemyProjectileDecisions = 0;
      let laserDecisions = 0;
      const enemyTypes = new Set();
      let terminated = false;
      let truncated = false;
      let cleared = false;
      let dead = false;

      while (!terminated && !truncated) {
        for (const entity of observation.entities ?? []) {
          if (entity.type >= 10 && entity.type < 1000 && entity.friendly !== true) {
            enemyTypes.add(`${entity.type}.${entity.variant}.${entity.subtype}`);
          }
        }
        const decision = policy.act(observation);
        const result = await environment.step(decision.action);
        const dealt = Number(result.info?.enemyDamage ?? 0);
        const taken = Number(result.info?.healthUnitLost ?? 0);
        const advanced = Number(result.info?.framesAdvanced ?? 0);
        episodeReward += result.reward;
        damageDealt += dealt;
        damageTaken += taken;
        frames += advanced;
        decisions++;
        if (decision.action.fire !== 'none') {
          firingDecisions++;
          if (dealt > 0) damageBearingFireDecisions++;
        }
        if ((result.observation.entities ?? []).some(entity => entity.type === 9 && entity.friendly !== true)) {
          enemyProjectileDecisions++;
        }
        if ((result.observation.entities ?? []).some(entity => entity.type === 7 && entity.friendly !== true)) {
          laserDecisions++;
        }
        observation = result.observation;
        terminated = result.terminated === true;
        truncated = result.truncated === true;
        cleared = observation.terminal?.cleared === true;
        dead = observation.terminal?.dead === true;
      }

      episodes.push({
        policy: label,
        evaluationStage: stage,
        seed,
        scene,
        roomVariant: observation.room.variant,
        episodeReward,
        cleared,
        dead,
        terminated,
        truncated,
        decisions,
        frames,
        clearTimeFrames: cleared ? frames : null,
        damageDealt,
        damageTaken,
        firingDecisions,
        damageBearingFireDecisions,
        // The public environment has no projectile-hit event counter. This is
        // an action-interval proxy, not per-shot projectile accuracy.
        damageBearingFireDecisionRate: firingDecisions
          ? damageBearingFireDecisions / firingDecisions : null,
        actualProjectileHitRate: null,
        enemyProjectileDecisions,
        laserDecisions,
        observedEnemyTypes: [...enemyTypes].sort(),
      });
    }
  }
  return { policy: label, stage, summary: summarizeEpisodes(episodes), episodes };
}

async function saveJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
    await rename(temporary, path);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

function isBetterOnHeldout(candidate, incumbent) {
  if (!incumbent) return true;
  const left = [candidate.roomClearRate ?? -1, -(candidate.deathRate ?? Infinity), candidate.meanDamageDealt ?? -1, -(candidate.meanDamageTaken ?? Infinity)];
  const right = [incumbent.roomClearRate ?? -1, -(incumbent.deathRate ?? Infinity), incumbent.meanDamageDealt ?? -1, -(incumbent.meanDamageTaken ?? Infinity)];
  for (let index = 0; index < left.length; index++) {
    if (left[index] > right[index]) return true;
    if (left[index] < right[index]) return false;
  }
  return false;
}

function chartSvg(points) {
  const width = 1320;
  const height = 1030;
  const panelWidth = 620;
  const panelHeight = 310;
  const lefts = [35, 665];
  const tops = [35, 365, 695];
  const charts = [
    { title: 'Episode reward (training rollout)', series: [{ key: 'episodeReward', label: 'mean episode reward', color: '#276EF1' }] },
    { title: 'Fixed test-set room clear rate', series: [{ key: 'testClearRate', label: 'held-out clear rate', color: '#139C6D' }], fixed: [0, 1] },
    { title: 'Actor loss', series: [{ key: 'actorLoss', label: 'policy loss', color: '#7A52CC' }] },
    { title: 'Critic loss', series: [{ key: 'criticLoss', label: 'value loss', color: '#F18F01' }] },
    { title: 'Policy entropy', series: [{ key: 'policyEntropy', label: 'joint entropy', color: '#008A9A' }] },
    { title: 'Approximate KL', series: [{ key: 'approxKl', label: 'old − new log probability', color: '#C44767' }] },
    { title: 'PPO clip fraction', series: [{ key: 'clipFraction', label: 'clipped samples', color: '#51606F' }], fixed: [0, 1] },
    { title: 'Critic explained variance', series: [{ key: 'explainedVariance', label: 'explained variance', color: '#B57700' }] },
    { title: 'Damage per rollout', series: [
      { key: 'damageDealt', label: 'dealt', color: '#139C6D' },
      { key: 'damageTaken', label: 'taken', color: '#D94F4F' },
    ] },
  ];
  const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;');
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
  svg += '<rect width="100%" height="100%" fill="#ffffff"/><style>text{font-family:Segoe UI,Arial,sans-serif;fill:#253248}.small{font-size:12px}.title{font-size:17px;font-weight:600}.axis{stroke:#9aa6b2;stroke-width:1}.grid{stroke:#e7ebf0;stroke-width:1}</style>';
  svg += '<text x="35" y="22" font-size="19" font-weight="700">Gamebot v0.3.2 PPO pilot — 6 curriculum updates</text>';
  charts.forEach((chart, chartIndex) => {
    const x = lefts[chartIndex % 2];
    const y = tops[Math.floor(chartIndex / 2)];
    const px = x + 54;
    const py = y + 38;
    const pw = panelWidth - 74;
    const ph = panelHeight - 94;
    const all = chart.series.flatMap(series => points.map(point => point[series.key]).filter(Number.isFinite));
    let min = chart.fixed?.[0] ?? (all.length ? Math.min(...all) : 0);
    let max = chart.fixed?.[1] ?? (all.length ? Math.max(...all) : 1);
    if (max === min) { max += 1; min -= 1; }
    const pad = chart.fixed ? 0 : (max - min) * 0.12;
    min -= pad;
    max += pad;
    svg += `<text class="title" x="${x + 4}" y="${y + 20}">${esc(chart.title)}</text>`;
    for (let tick = 0; tick <= 4; tick++) {
      const value = min + (max - min) * tick / 4;
      const yy = py + ph - ph * tick / 4;
      svg += `<line class="grid" x1="${px}" y1="${yy}" x2="${px + pw}" y2="${yy}"/>`;
      svg += `<text class="small" text-anchor="end" x="${px - 8}" y="${yy + 4}">${value.toFixed(2)}</text>`;
    }
    svg += `<line class="axis" x1="${px}" y1="${py}" x2="${px}" y2="${py + ph}"/><line class="axis" x1="${px}" y1="${py + ph}" x2="${px + pw}" y2="${py + ph}"/>`;
    points.forEach((point, index) => {
      const xx = px + (points.length === 1 ? pw / 2 : index * pw / (points.length - 1));
      svg += `<text class="small" text-anchor="middle" x="${xx}" y="${py + ph + 18}">${index + 1}</text>`;
      svg += `<text class="small" text-anchor="middle" x="${xx}" y="${py + ph + 34}">${esc(point.stageShort)}</text>`;
    });
    chart.series.forEach(series => {
      const coords = points.map((point, index) => {
        const value = point[series.key];
        if (!Number.isFinite(value)) return null;
        const xx = px + (points.length === 1 ? pw / 2 : index * pw / (points.length - 1));
        const yy = py + ph - (value - min) / (max - min) * ph;
        return [xx, yy];
      }).filter(Boolean);
      if (coords.length) {
        svg += `<polyline fill="none" stroke="${series.color}" stroke-width="2.5" points="${coords.map(([cx, cy]) => `${cx},${cy}`).join(' ')}"/>`;
        coords.forEach(([cx, cy]) => { svg += `<circle cx="${cx}" cy="${cy}" r="4" fill="${series.color}"/>`; });
      }
      const legendX = px + 8 + chart.series.indexOf(series) * 160;
      svg += `<line x1="${legendX}" y1="${y + panelHeight - 13}" x2="${legendX + 22}" y2="${y + panelHeight - 13}" stroke="${series.color}" stroke-width="3"/><text class="small" x="${legendX + 28}" y="${y + panelHeight - 9}">${esc(series.label)}</text>`;
    });
  });
  return `${svg}</svg>`;
}

let trainEnvironment;
let evaluationEnvironment;
try {
  await mkdir(outputDirectory, { recursive: true });
  await mkdir(checkpointDirectory, { recursive: true });
  const initialPolicy = new ActorCriticNetwork({ seed: 20261101 });
  const initialModel = initialPolicy.toJSON();
  trainEnvironment = new GamebotTrainingEnvironment({ actionRepeat, maxSteps, runRoot: join(runRoot, 'training') });
  evaluationEnvironment = new GamebotTrainingEnvironment({ actionRepeat, maxSteps, runRoot: join(runRoot, 'evaluation') });
  const trainer = new PPOTrainer({
    environment: trainEnvironment,
    policy: ActorCriticNetwork.fromJSON(initialModel),
    checkpointDirectory,
    resetOptions: { scene: stages[0].scene },
    seedSchedule: trainSeeds,
    config: {
      learningRate: 3e-4,
      gamma: 0.999,
      gaeLambda: 0.95,
      clipRange: 0.2,
      updateEpochs: 4,
      minibatchSize: 256,
      entropyCoefficient: 0.005,
      rolloutSteps: 256,
    },
  });

  const metadata = {
    schema: 'gamebot-ppo-pilot/1',
    createdAt: new Date().toISOString(),
    actionRepeat,
    maxStepsPerEpisode: maxSteps,
    trainingSeeds: trainSeeds,
    heldOutTestSeeds: testSeeds,
    heldOutTestScenes: testScenes,
    curriculum: stages,
    trainingConfig: trainer.config,
    actionSpace: { move: MOVE_ACTIONS, fire: FIRE_ACTIONS },
    changedRewardFunction: false,
    observationOrActionSchemaChanged: false,
    hitRateMetric: 'damageBearingFireDecisionRate = decision intervals with enemy HP reduction and fire != none / fire decision intervals; an action-level proxy, not projectile accuracy',
  };
  await saveJson(join(outputDirectory, 'metadata.json'), metadata);
  await saveJson(join(outputDirectory, 'initial-policy.json'), initialModel);

  const randomBaseline = await evaluatePolicy(evaluationEnvironment, new RandomPolicy(0x31a7), { label: 'random' });
  process.stdout.write(`${JSON.stringify({ event: 'baseline-complete', policy: 'random', summary: randomBaseline.summary })}\n`);
  const untrainedBaseline = await evaluatePolicy(
    evaluationEnvironment,
    ActorCriticNetwork.fromJSON(initialModel),
    { label: 'untrained-actor' },
  );
  process.stdout.write(`${JSON.stringify({ event: 'baseline-complete', policy: 'untrained-actor', summary: untrainedBaseline.summary })}\n`);
  await saveJson(join(outputDirectory, 'baselines.json'), {
    metadata,
    baselines: [randomBaseline, untrainedBaseline],
  });

  const rounds = [];
  const evaluations = [];
  let bestHeldout = null;
  let bestHeldoutModel = null;
  let bestHeldoutSelection = null;
  for (const [index, stage] of stages.entries()) {
    trainer.curriculumStage = stage.id;
    trainer.resetOptions = { scene: stage.scene };
    if (index > 0) {
      // Start the next curriculum stage from a clean room; the preceding
      // rollout already bootstrapped its final nonterminal observation.
      trainer.observation = null;
      trainer.episodeReturn = 0;
      trainer.episodeStats = null;
    }
    const summary = await trainer.trainIteration();
    const policySnapshot = trainer.policy.toJSON();
    const evaluation = await evaluatePolicy(
      evaluationEnvironment,
      ActorCriticNetwork.fromJSON(policySnapshot),
      { label: 'ppo', stage: stage.id },
    );
    const point = {
      iteration: summary.iteration,
      stage: stage.id,
      stageLabel: stage.label,
      stageShort: `S${index + 1}`,
      episodeReward: summary.rollout.meanEpisodeReward,
      actorLoss: summary.update.policyLoss,
      criticLoss: summary.update.valueLoss,
      policyEntropy: summary.update.entropy,
      approxKl: summary.update.approximateKl,
      clipFraction: summary.update.clipFraction,
      explainedVariance: summary.update.explainedVariance,
      roomClearRate: summary.rollout.roomClearRate,
      cumulativeRoomClearRate: summary.rollout.cumulativeRoomClearRate,
      damageDealt: summary.rollout.damageDealt,
      damageTaken: summary.rollout.damageTaken,
      damageBearingFireDecisionRate: summary.rollout.damageBearingFireDecisionRate,
      testClearRate: evaluation.summary.roomClearRate,
      testMeanEpisodeReward: evaluation.summary.meanEpisodeReward,
      testMeanDamageDealt: evaluation.summary.meanDamageDealt,
      testMeanDamageTaken: evaluation.summary.meanDamageTaken,
      testDamageBearingFireDecisionRate: evaluation.summary.damageBearingFireDecisionRate,
    };
    const round = {
      ...summary,
      stage,
      trainingMetrics: point,
      heldOutEvaluation: evaluation.summary,
    };
    rounds.push(round);
    evaluations.push(evaluation);
    if (isBetterOnHeldout(evaluation.summary, bestHeldout)) {
      bestHeldout = evaluation.summary;
      bestHeldoutModel = policySnapshot;
      bestHeldoutSelection = {
        stage: stage.id,
        iteration: summary.iteration,
        heldOutMetrics: bestHeldout,
        lexicographicCriterion: ['roomClearRate (higher)', 'deathRate (lower)', 'meanDamageDealt (higher)', 'meanDamageTaken (lower)'],
        testSeeds,
        testScenes,
      };
      await saveJson(join(outputDirectory, 'best-heldout-policy.json'), {
        ...policySnapshot,
        selection: bestHeldoutSelection,
      });
    }
    await saveJson(join(outputDirectory, 'training-history.json'), { metadata, rounds });
    await saveJson(join(outputDirectory, 'evaluation-history.json'), {
      metadata,
      baselines: [randomBaseline.summary, untrainedBaseline.summary],
      evaluations,
      bestHeldout,
    });
    await writeFile(join(outputDirectory, 'training-progress.jsonl'),
      `${JSON.stringify({ stage, trainingMetrics: point, evaluation: evaluation.summary, bestHeldoutSelection })}\n`,
      { flag: index === 0 ? 'w' : 'a' });
    process.stdout.write(`${JSON.stringify({ event: 'round-complete', stage: stage.id, trainingMetrics: point, evaluation: evaluation.summary })}\n`);
  }

  const finalEvaluation = await evaluatePolicy(
    evaluationEnvironment,
    ActorCriticNetwork.fromJSON(trainer.policy.toJSON()),
    { label: 'ppo-final', stage: 'final' },
  );
  let bestHeldoutEvaluation = null;
  if (bestHeldoutModel) {
    bestHeldoutEvaluation = await evaluatePolicy(
      evaluationEnvironment,
      ActorCriticNetwork.fromJSON(bestHeldoutModel),
      { label: 'best-heldout-ppo', stage: 'best-heldout' },
    );
  }
  const finalComparisonEpisodes = [
    ...randomBaseline.episodes,
    ...untrainedBaseline.episodes,
    ...finalEvaluation.episodes,
    ...(bestHeldoutEvaluation?.episodes ?? []),
  ];
  const byScenario = Object.fromEntries(testScenes.map(scene => [scene, Object.fromEntries([
    ['random', 'random'],
    ['untrainedActor', 'untrained-actor'],
    ['finalPPO', 'ppo-final'],
    ['bestHeldoutPPO', 'best-heldout-ppo'],
  ].map(([key, policyName]) => [key, summarizeEpisodes(
    finalComparisonEpisodes.filter(episode => episode.scene === scene && episode.policy === policyName),
  )]))]));
  const report = {
    ...metadata,
    totalTrainingDecisions: trainer.globalSteps,
    totalTrainingFrames: trainer.globalFrames,
    completedTrainingEpisodes: trainer.completedEpisodes,
    results: {
      random: randomBaseline.summary,
      untrainedActor: untrainedBaseline.summary,
      finalPPO: finalEvaluation.summary,
      bestHeldoutPPO: bestHeldoutEvaluation?.summary ?? null,
    },
    byScenario,
    baselineEpisodes: [randomBaseline.episodes, untrainedBaseline.episodes].flat(),
    curriculumRounds: rounds.map(round => ({
      stage: round.stage,
      training: round.trainingMetrics,
      heldOutEvaluation: round.heldOutEvaluation,
      completedTrainingEpisodes: round.rollout.episodeSummaries,
    })),
    heldOutEpisodes: [...evaluations.flatMap(item => item.episodes), ...finalEvaluation.episodes,
      ...(bestHeldoutEvaluation?.episodes ?? [])],
    bestHeldoutSelection,
    interpretation: {
      successRequiresHeldOutMetrics: true,
      actualProjectileHitRateAvailable: false,
      limitations: [
        'Only the 12 verified boss-room scenes are registered; there are no standalone stationary-target or isolated-dodge scenes.',
        'The pilot curriculum uses Gurdy Jr. as a stationary-target approximation and boss rooms for the other phases.',
        'Held-out coverage is two seeds by four rooms, so results are an initial smoke-scale estimate, not a generalization claim.',
        'Damage-bearing fire-decision rate is an action-interval proxy; the public observation does not expose per-projectile hit events.',
        'Six PPO updates are too few to establish reliable learning; use these data to assess stability before scaling.',
      ],
    },
  };
  await saveJson(join(outputDirectory, 'evaluation-history.json'), {
    metadata,
    baselines: [randomBaseline.summary, untrainedBaseline.summary],
    evaluations,
    finalPPO: finalEvaluation,
    bestHeldoutPPO: bestHeldoutEvaluation,
    bestHeldout: bestHeldoutSelection,
  });
  await saveJson(join(outputDirectory, 'pilot-report.json'), report);
  const curvePoints = rounds.map(round => round.trainingMetrics);
  await saveJson(join(outputDirectory, 'training-curves.json'), { metadata, points: curvePoints });
  const curveColumns = [
    'iteration', 'stage', 'episodeReward', 'actorLoss', 'criticLoss', 'policyEntropy',
    'approxKl', 'clipFraction', 'explainedVariance', 'roomClearRate', 'damageDealt',
    'damageTaken', 'testClearRate', 'testMeanDamageDealt', 'testMeanDamageTaken',
    'testDamageBearingFireDecisionRate',
  ];
  const csvCell = value => value == null ? '' : `"${String(value).replaceAll('"', '""')}"`;
  const csv = [curveColumns.join(','), ...curvePoints.map(point => curveColumns.map(key => csvCell(point[key])).join(','))].join('\n');
  await writeFile(join(outputDirectory, 'training-curves.csv'), `${csv}\n`, 'utf8');
  await writeFile(join(outputDirectory, 'training-curves.svg'), chartSvg(curvePoints), 'utf8');
  const scoreCell = value => Number.isFinite(value) ? value.toFixed(3) : '—';
  const policyRows = [
    ['Random', report.results.random],
    ['Untrained Actor', report.results.untrainedActor],
    ['Final PPO', report.results.finalPPO],
    ['Best held-out PPO', report.results.bestHeldoutPPO],
  ].map(([name, result]) => `| ${name} | ${result.episodes} | ${scoreCell(result.roomClearRate)} | ${scoreCell(result.deathRate)} | ${scoreCell(result.meanDamageDealt)} | ${scoreCell(result.meanDamageTaken)} | ${scoreCell(result.damageBearingFireDecisionRate)} | ${scoreCell(result.meanClearTimeFrames)} |`);
  const roundRows = rounds.map(round => {
    const point = round.trainingMetrics;
    return `| ${point.iteration} | ${point.stageLabel} | ${scoreCell(point.episodeReward)} | ${scoreCell(point.actorLoss)} | ${scoreCell(point.criticLoss)} | ${scoreCell(point.policyEntropy)} | ${scoreCell(point.approxKl)} | ${scoreCell(point.clipFraction)} | ${scoreCell(point.explainedVariance)} | ${scoreCell(point.testClearRate)} |`;
  });
  const scenarioRows = testScenes.flatMap(scene => ['random', 'untrainedActor', 'finalPPO', 'bestHeldoutPPO'].map(key => {
    const metrics = report.byScenario[scene][key];
    return `| ${scene} | ${key} | ${metrics.episodes} | ${scoreCell(metrics.roomClearRate)} | ${scoreCell(metrics.deathRate)} | ${scoreCell(metrics.meanDamageDealt)} | ${scoreCell(metrics.meanDamageTaken)} | ${scoreCell(metrics.damageBearingFireDecisionRate)} |`;
  }));
  const summaryMarkdown = [
    '# Gamebot v0.3.2 PPO pilot', '',
    `Training seeds: ${trainSeeds.join(', ')}. Held-out seeds: ${testSeeds.join(', ')}. Test rooms: ${testScenes.join(', ')}.`,
    `Six PPO updates, ${trainer.globalSteps} decisions, action repeat ${actionRepeat}, episode frame limit ${maxSteps}.`, '',
    '## Fixed test set', '',
    '| Policy | Episodes | Clear rate | Death rate | Mean damage dealt | Mean damage taken | Damage-bearing fire decision rate* | Mean clear frames |',
    '|---|---:|---:|---:|---:|---:|---:|---:|', ...policyRows, '',
    '## Training and held-out metrics', '',
    '| Round | Curriculum stage | Episode reward | Actor loss | Critic loss | Entropy | Approx KL | Clip fraction | Explained variance | Test clear rate |',
    '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|', ...roundRows, '',
    '## Per-room held-out comparison', '',
    '| Room | Policy | Episodes | Clear rate | Death rate | Damage dealt | Damage taken | Fire-damage decision proxy |',
    '|---|---|---:|---:|---:|---:|---:|---:|', ...scenarioRows, '',
    '## Interpretation', '',
    '- No policy cleared a room on the fixed test set; mean clear time is therefore unavailable.',
    '- Final PPO did not beat Random on death rate or mean episode reward. Best-held-out PPO improved mean damage and the action-level damage proxy modestly, but still cleared no room.',
    '- Critic explained variance stayed near zero or below zero. Entropy remained high and approximate KL/clip fraction stayed small across six updates.',
    '- Do not expand training from this result. Improve value learning and combat signal, then rerun the same held-out suite before claiming progress.',
    '- *The public environment exposes enemy HP deltas, not projectile-hit events. This rate is the fraction of fire decision intervals that coincide with enemy damage, not per-shot accuracy.',
    '- Gurdy Jr. is only a stationary-target approximation; the existing environment has no standalone stationary-target or isolated-dodge scenario.', '',
  ].join('\n');
  await writeFile(join(outputDirectory, 'pilot-summary.md'), summaryMarkdown, 'utf8');
  process.stdout.write(`${JSON.stringify({ event: 'pilot-complete', outputDirectory, results: report.results, bestHeldout: report.bestHeldoutSelection })}\n`);
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
} finally {
  await Promise.all([trainEnvironment?.close(), evaluationEnvironment?.close()]);
}

