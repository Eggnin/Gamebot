# Gamebot 训练环境

这是以撒 Gamebot 的离线单房间训练环境。项目使用 Node.js 协调原版 J460 引擎的本地 WASM 构建，提供回合重置、动作与观察接口、快照分叉以及 PPO 训练脚本。机器人本体和游戏研究资料位于配套的 [Isaac 仓库](https://github.com/Eggnin/Isaac)。

## 目录

- [`gamebot-training-engine-v0.1/`](gamebot-training-engine-v0.1/)：训练环境源码、测试、脚本、样本证据和训练结果。
- [训练环境详细说明](gamebot-training-engine-v0.1/README.md)：支持的场景、动作协议、配置、快照和训练方式。

## 快速开始

需要 Windows 和 Node.js 20 或更新版本。以下命令从本仓库根目录执行：

```powershell
cd .\gamebot-training-engine-v0.1
npm test
```

这组测试检查动作、观察数据和场景约定，不需要启动游戏引擎。运行环境和训练任务还需要本机准备好的 J460 训练运行时、游戏资源和配套研究数据；仓库本身不包含这些运行时文件。默认位置为 `D:\Isaac\research`，也可通过 `GAMEBOT_RESEARCH_ROOT` 指定。具体文件、版本校验和运行命令见[详细说明](gamebot-training-engine-v0.1/README.md#runtime-required-on-this-pc)。

例如，准备好运行时后可启动标准输入/输出服务：

```powershell
$env:GAMEBOT_RESEARCH_ROOT = 'D:\Isaac\research'
npm run serve
```

## 仓库内容

源码、测试、`evidence/` 样本和 `training-results/` 中现有的汇总及策略文件会纳入版本管理。`runs/` 中的轨迹、日志和大型快照以及根目录的重复压缩包留在本机，不随仓库上传。训练结果不代表实机无伤通过。
