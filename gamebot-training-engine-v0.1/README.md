# Gamebot single-room training environment v0.1

This adapter runs the existing J460 training build of **The Browsing of Isaac** as an offline, headless Node environment. It reuses the local WASM engine, verified memory image, resource bundle, room observer, and current Gamebot research data. It does not copy or alter the installed game, the Gamebot project, or the upstream checkout.

## Scope

- Twelve verified single-room boss scenes are available: Monstro, Larry Jr., Chub, Duke of Flies, Blastocyst, Loki, Gurdy Jr., The Bloat, Widow, Famine, Pestilence, and The Cage. The `frowningGaperGroup` scene adds the original `d.841` room with five `10.0.0` Frowning Gapers. Each episode stops on room clear, player death, or the frame limit. `scene: 'auto'` continues to cycle the twelve boss scenes and a fixed set of confirmed seeds.
- Actions are exactly two fields: `move` (`none`, eight compass directions) and `fire` (`none`, `up`, `right`, `down`, `left`). Bombs, active items, cards, pills, inventory and room transitions are rejected or omitted.
- Observations expose player position/velocity/health, combat entities and projectiles, and room grid collision/danger cells. Memory pointers and resource entities are removed.
- The first `reset()` cold-starts a Node engine process. Repeating the same seed/scene restores its 640 MiB native-memory snapshot; changing seed or room uses the game's native `seed` / `stage` / `goto` console path and verifies the seed, room identity, and target NPC before returning. Console setup briefly re-enables presentation frames because the original console types against that counter, then returns to headless mode.
- `snapshot()` / `restore()` save and restore the full WASM linear memory and the worker's held-input / frame state. `fork()` creates named sequential branches. `exportSnapshot()` plus `forkParallel()` move a build-checked snapshot into a separate Node engine process for an independent branch.
- The default reward is configurable: enemy HP removed `+1`, health unit lost `−5`, room clear `+25`, death `−25`, and each game frame `−0.001`.

The reward weights are initial engineering defaults, not a claim that they are optimal. They can be changed through `rewardWeights` in the JavaScript API.

Each in-memory snapshot is 640 MiB; two may be held per worker. `fork()` handles share an immutable base snapshot. `exportSnapshot()` writes a hash-checked file under the run directory, and `forkParallel()` imports it into an independent worker with matching executable/WASM hashes. Call `dropSnapshot()` to remove a snapshot and any export file it owns. A restore checks the resulting observation hash. Game save persistence is disabled for training runs. This WASM build does not export Emscripten FS; resource reads use positional reads and snapshots cover linear memory plus worker input/frame state.

`forkParallel()` starts a new worker and imports the snapshot; starting two independent fork workers took about **7.3 seconds total** in the measured parallel run because each new WASM module must boot once. For repeated batches, prewarm workers with `deferInitialSnapshot: true`, then call `importSnapshot(exported.path)` on each worker and reuse those processes.

## Runtime required on this PC

The adapter expects the existing local training runtime at `D:\Isaac\research`:

- `offline/` observer/planner modules used by the isolated worker at `src/engine_worker.mjs`
- `runtime-inputs/host/isaac.segs.bin` plus the passing `memimage-verify.json`
- `the-browsing-of-isaac/output/recomp/lift/boot-training-fast/boot.mjs` and `boot.wasm`
- `runtime-instance/resources/packed`
- `D:\Isaac\data\npc\engine_monster_static_mechanism_candidates.json`

The runtime manifest and installed EXE are checked before reset. This v0.1 build is pinned to the locally verified J460 executable SHA-256 `7122AC28779925B24E23E2416F231322B1470388BD25E2C08665AD8D53B3EA4F`; it fails closed on a mismatch. Set `GAMEBOT_RESEARCH_ROOT` if the local research directory differs.

Node.js 20 or newer is required. No Python package or desktop game window is needed.

## 10.0.0 多敌人躲避评测

`scripts/evaluate_frowning_gaper.ps1` 用本目录的隔离训练 worker 和 `D:\Isaac` 的正式 C++ Reader / Planner 逐帧评测皱眉裂口尸。`natural` 使用原版地下室 `d.841` 房间中生成的 5 只敌人；`wall` 和 `encircled` 在原版横向大房间中分别布置 6 只敌人。后两者是受控场景，不是原版自然房间分布。脚本核对房间、敌人身份和数量、生成位置的地形及初始生命；伤害同时与引擎中的原生伤害回调对照。

普通房也可直接用于训练：`await env.reset({ scene: 'frowningGaperGroup', seed: '3JY1 6FLR' })`，或执行 `npm run train -- --scene=frowningGaperGroup`。重置会核对 5 只敌人的精确身份及初始生命；自动轮换 `auto` 仍保持原有 Boss 列表。靠墙和包围两种受控布局目前通过下方评测脚本运行。

```powershell
Set-Location D:\Gamebot\gamebot-training-engine-v0.1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\evaluate_frowning_gaper.ps1 -Layout natural
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\evaluate_frowning_gaper.ps1 -Layout encircled -Seed 'SWGF G7WV'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\evaluate_frowning_gaper.ps1 -Layout wall -DodgeOnly
```

`-DodgeOnly` 同时关闭规划器攻击决策和输出射击键，最多观察 60 秒；无攻击时应比较实际存活时长、首次受伤及生命变化，不能把未清房的短暂零伤当作无伤清房。报告及冻结的 Reader/数据位于 `D:\Isaac\validation\gaper-*`，引擎记录位于本项目的 `runs/frowning-gaper/`。此评测不启动桌面游戏，场景生成只作用于隔离训练进程。

该评测调用重新构建的本机 `ExternalReader.exe`，不依赖 `research/offline/planner_policy.wasm` 来做决策。本轮也重新构建了后者，并把原版房间形状传入离线规划器；无法取得有效形状时拒绝规划。一个普通房纯躲避场景中，两条接口的 6 次扣血帧和死亡帧一致，见 `D:\Isaac\validation\gaper-wasm-reader-parity-20261101.json`。这只核对一个场景的伤害与结束帧，不代表所有字段和场景均等价。

## Run

```powershell
$env:GAMEBOT_RESEARCH_ROOT = 'D:\Isaac\research'
$env:GAMEBOT_ACTION_REPEAT = '4'
Set-Location 'D:\Gamebot\gamebot-training-engine-v0.1'
node .\src\stdio_server.mjs
```

The server accepts one JSON object per line and emits one JSON result per line:

```json
{"command":"reset","scene":"auto"}
{"command":"observe"}
{"command":"step","action":{"move":"up_right","fire":"left"}}
{"command":"snapshot","label":"candidate-root"}
{"command":"fork","snapshotId":2,"label":"left-branch"}
{"command":"restore","fork":"fork-1"}
{"command":"fork_parallel","snapshotId":2,"label":"parallel-candidate"}
{"command":"branch_step","branchId":"parallel-1","action":{"move":"left","fire":"down"}}
{"command":"close"}
```

Reset automatically keeps snapshot `1`; the first explicit snapshot in a fresh worker is therefore `2`. In a client, use the `snapshot.id` returned by the server instead of hard-coding the example ID.
`fork_parallel` returns a separate `branchId`; use `parallel_step` with several `branchId`/`action` pairs to advance those workers together, then `branch_close` when finished.

Each `step` is one decision interval and advances four native logic frames by default. The external trainer therefore exchanges one action/result pair per four game frames, not once per game frame. Set `GAMEBOT_ACTION_REPEAT` to another integer from 1 to 16 if needed. Within v0.1, the Node coordinator still requests individual frame results from the existing engine worker; collapsing that internal Node-to-Node traffic is a later optimization if benchmarks show it matters.

Every episode gets its own `trajectory.jsonl` and an engine log in the run directory. A Windows junction shares the existing research resource bundle read-only. Game saves are not persisted to disk.

## JavaScript API

```js
import { GamebotTrainingEnvironment } from './src/environment.mjs';

const env = new GamebotTrainingEnvironment({ actionRepeat: 4 });
try {
  let observation = await env.reset({ seed: '3JY1 6FLR' });
  const snapshot = await env.snapshot('candidate-root');
  const branch = await env.fork(snapshot.id, 'left-branch');
  observation = await env.restore(branch);
  const result = await env.step({ move: 'right', fire: 'up' });
  observation = result.observation;
  const parallel = await env.forkParallel(snapshot.id, 'parallel-left');
  const [baseStep, forkStep] = await Promise.all([
    env.step({ move: 'right', fire: 'up' }),
    parallel.step({ move: 'left', fire: 'down' }),
  ]);
  await parallel.close();
  // result also contains reward, terminated, truncated and reward details.
} finally {
  await env.close();
}
```

## Actor-Critic 网络

`src/actor_critic.mjs` 提供无外部依赖的离散 Actor-Critic 网络。共享编码器分别处理玩家状态、最近的 8 个敌人、最近的 16 个投射物，以及映射到 15×9 网格的地形；四路特征合并后经过 128 和 96 单元的共享主干。

- 移动 Actor 输出 9 类：停止和八方向。
- 射击 Actor 输出 5 类：停止和上下左右。
- 两个 Actor 头根据同一个状态表示分别采样，然后把 `{ move, fire }` 一起传给环境。联合 log-probability 是两头 log-probability 之和。
- Critic 只输出状态价值 `V(s)`，不参与动作选择。
- `updatePPO()` 提供 clipped PPO 小批次更新；优势值和回报由采集端提供，例如用 GAE 计算。

```js
import { ActorCriticNetwork } from './src/actor_critic.mjs';

const policy = new ActorCriticNetwork({ seed: 42 });
const decision = policy.act(observation);
const result = await env.step(decision.action);

// 收集一段 rollout 并计算好 advantage / return 后：
policy.updatePPO([{
  observation,
  action: decision.action,
  oldLogProb: decision.logProb,
  advantage,
  return: targetValue,
}], { epochs: 4, batchSize: 256 });

const checkpoint = policy.save();
const restoredPolicy = ActorCriticNetwork.fromJSON(checkpoint);
```

`forward(observation)` 返回两个头的 logits、概率和 `value`；`act(observation, { deterministic: true })` 用各头概率最大项做确定性选择。`evaluateActions()` 可计算给定动作对的联合概率与熵。模型输入字段定义和网络层宽度见 `ActorCriticNetwork.architecture`。

## PPO Trainer

`src/ppo_trainer.mjs` 提供 `RolloutBuffer`、`computeGAE()` 和 `PPOTrainer`。每条 rollout 保存状态、移动/射击动作、联合 `oldLogProb`、奖励、当前价值、下一状态价值，以及终止/截断标记。PPO Ratio 使用两个独立 Actor 头 log-probability 的和：

```text
log π(a|s) = log π_move(move|s) + log π_fire(fire|s)
ratio = exp(newLogProb - oldLogProb)
```

GAE 用 `δ = r + γ V(s') - V(s)` 和 `A = δ + γλ A_next`。真正终止状态不做价值 bootstrap；时间截断仍以末状态 `V(s')` 计算 TD residual，但会切断跨回合的优势递推。rollout 刚好结束而回合仍继续时，也会用最后状态的价值 bootstrap。

第一版默认参数：Learning Rate `3e-4`、Gamma `0.999`、GAE Lambda `0.95`、PPO Clip `0.2`、Update Epochs `4`、Minibatch `256`、Entropy Coefficient `0.005`、每轮收集 `1024` 个决策。Gamma 是每个环境决策步的折扣值；环境默认每个决策推进 4 个原生逻辑帧，应按 action repeat 和目标时间跨度校准。

启动训练（运行环境前需设置上文的 `GAMEBOT_RESEARCH_ROOT`）：

```powershell
npm run train -- --iterations=20 --rollout-steps=1024 --scene=auto
```

每轮更新后写入 `runs/ppo-checkpoints/latest-training.json`，其中包含模型、Adam 状态、轮数和步数；有新的已完成回合平均奖励纪录时写入 `best-policy.json`。可用 `--resume=PATH` 继续，例如：

```powershell
npm run train -- --iterations=20 --resume=runs/ppo-checkpoints/latest-training.json
```

恢复会继续模型、优化器和训练计数，并从新回合开始；WASM 环境快照不包含在 PPO 检查点中。`best-policy.json` 可直接交给 `ActorCriticNetwork.fromJSON()` 加载作策略推理。

### v0.3.2 小规模首跑

`npm run pilot` 会先评估 Random 与未训练 Actor，再跑六阶段课程，每次更新后都评估固定 held-out 测试集。报告、逐 episode 数据、CSV/SVG 曲线、训练 latest checkpoint 和 held-out best policy 写入 `training-results/ppo-v0.3.2-first-run/`。这次试验使用 3 个训练种子、2 个独立测试种子和 4 个现有 Boss 房；Gurdy Jr. 只作为静止目标的近似场景。

从首跑的 latest checkpoint 继续 4 次混合场景更新、每轮评估同一 held-out 集，可运行 `npm run continue-pilot`。续训结果独立写到 `training-results/ppo-v0.3.2-continuation/`，不会覆盖首跑文件。

## 专家策略数据采集（Phase 1）

`scripts/collect_expert_dataset.mjs` 把现有 C++ 专家策略跑在同一个离线 J460 引擎里，逐帧记录 `(observation, expert action, transition outcome)`：

- 专家不是新写的：`D:/Isaac/research/external-validation/ExternalReader.exe` 用生产 `Reader` 通过 `ReadProcessMemory` 读取本环境 WASM 堆；`reader_main.cpp` 新增 `policy:"mpc"`，走 `RealtimeMpcController → MpcCombatPlanner`，`policy:"quick"` 仍保留原 `Planner` 作为对照。
- 训练侧新增两个纯增量接口：`GamebotTrainingEnvironment({ externalValidation: true })` 打开 worker 的 `externalAddress` / `externalTruth`；`env.request(command, action)` 暴露该传输。不改变 observation、action、reward 或 PPO 行为。
- 每个 episode 用固定 scene + seed，第一帧检查 `expert.clock === observation.frame` 和玩家坐标一致，错位即中止该回合；记录每步的真实生命/敌人 HP 变化、reward、终止原因和专家诊断（MPC stats、SafetyGate、预测瞄准、攻击窗口、Shot progress）。

```powershell
node .\scripts\collect_expert_dataset.mjs --scenes=monstro,theBloat --seeds='3JY1 6FLR,7W3L P1Z9' --frames=900 --output=D:/Isaac/validation/expert-dataset-<date>
node .\scripts\check_expert_dataset.mjs --dataset=D:/Isaac/validation/expert-dataset-<date>
node .\scripts\verify_expert_replay.mjs --dataset=D:/Isaac/validation/expert-dataset-<date>
```

产物：`manifest.json`（参数、专家时钟、文件哈希、逐回合汇总）、`index.jsonl`、`episodes/<id>/samples.jsonl`、`episodes/<id>/episode.json`（结果、指标、专家统计、动作/观察链哈希）。`--diagnostics` 另写逐帧完整 Reader JSON。`--policy=quick` 用原 Planner 作对照。

采集时会把专家二进制、worker、environment、`address.mjs` 和采集器自身复制到 `<dataset>/provenance/`，门禁校验归档副本：之后重新编译专家或改训练代码不会让已采数据失效。

2026-11-03 实跑结果（数据保存在本机 `D:/Isaac/validation/expert-dataset-*`，报告见 [Phase 1 报告](D:/Isaac/validation/EXPERT-DATASET-PHASE1-20261103.md)）：

| 数据集 | policy | 回合 | 帧 | 清房 | 掉血 | 门禁 | 重放 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `expert-dataset-20261103` | mpc | 12 | 10800 | 0 | 12 | 通过 | 10800/10800 帧一致 |
| `expert-dataset-long-20261103` | mpc | 3 | 4966 | 3 | 1 | 通过 | 4966/4966 帧一致 |
| `expert-dataset-quick-long-20261103` | quick | 3 | 5783 | 3 | 3 | - | - |

MPC 长局 3/3 清房，比原 Planner 少 14% 帧数、少 2 点掉血；但 30 秒短局里原 Planner 掉血更少，说明 MPC 不是全面更强的教师。逐发命中率当前不能用 C++ 弹幕归因（在 30 Hz 桥接中失效），报告里改用"真实 HP 下降 ÷ 3.5 ÷ 弹幕数"的估算口径。

`npm run collect-expert`、`npm run check-dataset`、`npm run verify-replay` 是同一组脚本的快捷入口。

质量门禁检查 schema、时间对齐、动作域、生命/伤害记账、帧与观察链连续性、哈希和来源文件哈希；重放验证用同一 scene/seed 重放记录动作并逐帧比对观察哈希。任一失败都返回非零退出码。

## Behavior Cloning（Phase 2）

`scripts/train_bc.mjs` 在**同一个** `ActorCriticNetwork` 上用专家动作做监督学习：新增的 `updateSupervised()` 只对两个动作头做交叉熵，共享编码器/主干与 Adam 状态；`updatePPO()` 未改动。训练/验证/测试按**战斗种子**切分，脚本会拒绝训练种子出现在 val/test 的数据。

```powershell
node .\scripts\train_bc.mjs `
  --train-datasets=D:/Isaac/validation/expert-dataset-20261103,D:/Isaac/validation/expert-dataset-long-20261103 `
  --train-seeds='3JY1 6FLR' --val-dataset=D:/Isaac/validation/expert-dataset-20261103 --val-seeds='7W3L P1Z9' `
  --test-dataset=D:/Isaac/validation/expert-dataset-heldout-20261103 --test-seeds='SWGF G7WV' `
  --epochs=20 --output=D:/Isaac/validation/bc-balanced-20261103
node .\scripts\evaluate_policy.mjs --policy=D:/Isaac/validation/bc-balanced-20261103/best-policy.json `
  --scenes=monstro,theBloat,theCage,gurdyJr --seeds='SWGF G7WV' --frames=900 `
  --compare-dataset=D:/Isaac/validation/expert-dataset-heldout-20261103 --output=D:/Isaac/validation/bc-eval
```

2026-11-03 实跑（held-out 种子，4 场景 × 900 帧）：随机 4 死/122 输出，未训练 Actor 3 死/65 输出，BC 2–3 死/319 输出，MPC 1 死/519 输出；BC 射击方向准确率 0.85，移动只有 ~0.40。结论：BC 已不是随机策略，但还打不过 MPC，**不替换、不接实机**，只作为 PPO 初始化。训练两次的 best-policy 哈希完全相同，闭环评估重复结果一致。完整数据与逐场景对照见 [Phase 2 报告](D:/Isaac/validation/BC-PHASE2-20261103.md)。

DAgger 采集用 `scripts/collect_dagger.mjs`：让当前策略跑引擎，在它访问到的状态上查询 MPC 并输出同格式数据集（`--policy=` / `--scenes=` / `--seeds=` / `--frames=`）。实测两轮把 BC↔MPC 一致率从 5.0% 提到 9.7%，但只改善了训练种子上的表现；按闭环验证集选择，最佳模型仍是 `bc-balanced-20261103`（val 1 死/13 掉血/308 输出，held-out 2 死/19 掉血/319 输出）。DAgger 的下一步必须多训练种子并保留更高比例专家状态，不能再单种子加轮数。

## PPO 微调（Phase 3）

`scripts/train_ppo_from_bc.mjs` 从 BC 权重继续 PPO（复用 `PPOTrainer`/GAE/Adam/断点续训，不重新初始化），奖励由 `scripts/lib/shaped_environment.mjs` 包装环境生成：真实敌人 HP 下降 +1/HP、真实掉血 −8/半心、清房 +40、死亡 −40、时间 −0.002/帧、连续 120 帧无伤害后 −0.02/帧、空动作 −0.01/次。

```powershell
node .\scripts\train_ppo_from_bc.mjs --policy=D:/Isaac/validation/bc-balanced-20261103/best-policy.json `
  --iterations=12 --rollout-steps=1024 --seeds='3JY1 6FLR' --output=D:/Isaac/validation/ppo-from-bc-20261103
```

2026-11-03 实跑 12,288 帧/12 迭代：15 回合 0 清房 14 死亡，熵 2.06–2.39，critic 解释方差 −0.089–0.025。验证集（7W3L）：BC 1 死/13 掉血/308 输出，PPO best 2 死/21/332，PPO latest 3 死/23/307，MPC 0 死/7/656。**PPO 没有改进，按纪律不替换 BC，也不接实机**；断点续训已验证（恢复后从第 13 次迭代继续）。完整记录见 [Phase 3 报告](D:/Isaac/validation/PPO-PHASE3-20261103.md)。

## Cross-process Forks

Given a reset environment `base` and its `snapshot`, prewarm worker processes once and import the exported base snapshot into them:

```js
const snapshot = await base.snapshot('rollout-root');
const exported = await base.exportSnapshot(snapshot.id);
const pool = await Promise.all(Array.from({ length: 2 }, async () => {
  const worker = new GamebotTrainingEnvironment({ deferInitialSnapshot: true });
  await worker.reset({ seed: snapshot.seed, scene: snapshot.scene });
  await worker.importSnapshot(exported.path);
  return worker;
}));

try {
  const results = await Promise.all(pool.map((worker, index) =>
    worker.step({ move: index ? 'left' : 'right', fire: 'up' })
  ));
} finally {
  await Promise.all(pool.map(worker => worker.close()));
}
```

The initial pool boot costs about 6–7 seconds per worker in parallel. Snapshot import then restores the saved game memory without restarting that worker; subsequent forks can reuse the same pool processes.

## Checks and benchmark

```powershell
Set-Location 'D:\Gamebot\gamebot-training-engine-v0.1'
npm test
$env:GAMEBOT_RESEARCH_ROOT = 'D:\Isaac\research'
node .\scripts\protocol_smoke.mjs
node .\scripts\check_determinism.mjs
node .\scripts\check-snapshot-fork.mjs
node .\scripts\check-parallel-fork.mjs
node .\scripts\check-auto-reset.mjs
node .\scripts\bench_parallel.mjs --instances=2 --decisions=120 --action-repeat=4
```

The benchmark runs independent engine processes and reports logic frames/second, 30 Hz real-time multiple, and before/after resident and private memory samples per instance. The built-in memory values are point samples, not a peak. Run output and trajectories are written under `runs/` by default; set `GAMEBOT_RUN_ROOT` to choose another location.

Fresh measurements on 2026-11-01 with the 640 MiB snapshot held in each worker: one instance reached **632.1 logic frames/s (21.1× real time)**. Two concurrent instances reached **900.0 aggregate frames/s (30.0× aggregate real time)**, or **432.5 / 467.5 frames/s per instance**. Cold setup took **6.9 seconds** for one instance and **7.2 seconds** when two started together. The short speed-test actions ended in player death; these figures measure engine throughput, not policy quality. See [benchmark results](evidence/benchmark-results.json).

The snapshot/fork replay check restored a 640 MiB checkpoint in **37 ms** and reproduced the same 60-frame branch twice. Same-seed/same-room automatic reset from the saved checkpoint took **32 ms**. A sweep of all 12 room scenarios plus the automatic cycle passed: one cold setup, then 12 verified warm console resets averaging about **0.70 seconds**. Two independent workers imported the same snapshot and replayed matching actions; file-backed restore took about **406 ms**. Explicitly dropping the source snapshot removed its file while the worker-owned branch link remained usable, then both branch files cleaned up correctly. See [snapshot/fork results](evidence/snapshot-fork-check.json), [parallel fork results](evidence/parallel-fork-check.json), and [auto-reset results](evidence/auto-reset-check.json).

## Current limits

- The scenario catalogue contains the 12 listed boss rooms and the one original Frowning Gaper room; arbitrary ordinary-room templates are not yet available through `reset()`.
- `fork()` branches share one immutable base snapshot and are evaluated sequentially within that environment. `forkParallel()` exports/imports the same snapshot into independent workers; each forked worker pays one cold module startup before import.
- In-memory snapshots are process-local. Exported snapshot files persist under the run directory and are pinned to the verified J460/WASM hashes.
- The package provides the training environment and transport, not PPO or a learned policy. It is ready for an external optimizer or a later in-process training loop.
- `actionRepeat=4` is the recommended trainer boundary. The existing Node engine worker still reports each native frame internally.
