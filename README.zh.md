<div align="center">

# 🌳 PHASE-Tree
### 在长程角色扮演对话中建模角色状态的演化

[![Paper](https://img.shields.io/badge/Paper-EMNLP%202026-b31b1b.svg)](https://anonymous.4open.science/r/PHASE-Tree)
[![Code](https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github)](https://github.com/MemTensor/PHASE-Tree)
[![Data](https://img.shields.io/badge/🤗%20Data-Dataset-yellow.svg)](https://huggingface.co/datasets/IAAR-Shanghai/phase_tree_data)
[![Model](https://img.shields.io/badge/🤗%20Model-Model-yellow.svg)](https://huggingface.co/IAAR-Shanghai/phase_tree_models)
[![Results](https://img.shields.io/badge/🤗%20Results-Results-yellow.svg)](https://huggingface.co/datasets/Mathematics-Yang/phase_tree_results)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**English**](README.md) | **中文**

<p>
  <em>一个心理学驱动、多时间尺度的角色状态表示——让角色扮演模型从角色<strong>当前的叙事状态</strong>出发回应，而不是被一个固定的人物档案锁死。</em>
</p>

<img src="figures/phase-tree-framework.png" width="92%" alt="PHASE-Tree 角色状态层级"/>

<p><sub><b>图 1.</b> PHASE-Tree 将一个角色分解为<b>不可变身份</b>根节点 + 三个不同时间尺度的可变层：长期 <b>persona</b>（人格）、会话级 <b>session</b>（适应）、瞬时 <b>moment</b>（情绪/情境）。每个字段都是独立可寻址的更新目标，遵循<em>阻力</em>、<em>证据</em>、<em>冷却</em>三重门控。</sub></p>

</div>

---

## 📖 摘要

> 长程角色扮演要求角色在故事推进中既保持可辨识度，又能跟随情节演化；而现有方法与基准大多在测"人格保持"或"记忆召回"，并未检验模型能否从角色**当前演化后的状态**出发回应——我们把这种失败模式称为 **stale-state failure（陈旧状态失败）**。我们提出 **PHASE-Tree**：一棵多时间尺度的角色状态树，根节点是不可变身份，下面是可变的 `persona`、`session`、`moment` 三层；每个字段都是可独立寻址的更新目标，支持场景内与跨剧集的局部演化。我们同时提出 **LongEvoRoleBench**——一个面向长程角色状态演化的评测套件：4 个长对话语料构成"跨剧集演化"的核心测试；4 个短对话语料则在同一生成格式下提供"场景内状态跟踪"的检查。PHASE-Tree 可以通过 **explicit textual provision**（把树序列化进 prompt）或 **implicit parametric adaptation**（把树编码进 LoRA 权重）两种范式条件生成。在 LongEvoRoleBench 上，textual provision 在 **24 个内部消融格子中拿下 21 个最佳**，并在长对话宏平均上相对各指标最强的 textual-provision 外部 baseline 分别提升 **+0.49 Char（+19.6%）**、**+0.41 Sem（+12.4%）**、**+0.04 Emb（+15.0%）**。

## 🧭 速览

**PHASE-Tree**（*Psychology-grounded Hierarchical Attribute-Structured Evolving Tree*，心理学驱动的层次化属性结构演化树）由三部分组成：

1. **角色状态表示**：一棵四层角色状态树——不可变身份根节点 + 三个可编辑层（`persona / session / moment`）。每个字段都是独立可寻址的更新目标，支持场景内状态跟踪与跨剧集的人格演化（受 *阻力—证据—冷却* 三重门控约束）。
2. **两种条件生成范式**：同一份扁平化的 PHASE-Tree 通过两条互补路径驱动生成——**explicit textual provision**（把树序列化进 prompt，主验证路径，状态完全可检视）与 **implicit parametric adaptation**（用 profile-to-adapter 超网络编码为 LoRA 权重，token 高效的部署变体）。
3. **评测基准 — LongEvoRoleBench**：面向"长程角色状态演化"的评测套件，把 **8 个角色扮演语料库** 统一为下一句生成任务，含随机/OOD 切分、与状态对齐的指标、以及两种范式下的完整 baseline 分数。

<div align="center">
<img src="figures/training-pipeline.png" width="82%" alt="文本提供 vs 参数适配 两种范式"/>
<p><sub><b>图 2.</b> 同一份扁平化的 PHASE-Tree 走两条条件生成范式——<b>Explicit Textual Provision</b>（上方蓝色）：角色档案放在 prompt 中，状态可被完全检视；<b>Implicit Parametric Adaptation</b>（下方红色）：角色档案被压缩到超网络生成的 LoRA 权重里，对话 prompt 不再携带档案，零额外 token 开销。</sub></p>
</div>

---

## 📰 动态

- **2026-05** &nbsp; 代码、模型、数据、完整评测结果发布于 GitHub + Hugging Face。
- **2026-05** &nbsp; 论文投稿 EMNLP 2026。

---

## 🏆 主要结果

在 **LongEvoRoleBench**（backbone = `Qwen2.5-7B-Instruct`）上。下面统一用 **Ours (textual)** 和 **Ours (parametric)** 来区分两种条件生成范式——它们都是 PHASE-Tree，只是分别处在不同的 baseline 块里（论文 Table 1 = 内部消融，Table 2 = 外部对比，里面有两个 `Ours` 列并排）。

| 设置 | 结果 |
|------|------|
| 🏅 **内部消融（textual）** | PHASE-Tree 在 **24 个数据集–指标格子中拿下 21 个最佳**。 |
| 📈 **外部 baseline（长对话宏平均, textual 块）** | **Ours (textual)** 相比该指标上最强的 textual-provision baseline 分别提升 **+0.49 Char（+19.6%）**（vs PAG）、**+0.41 Sem（+12.4%）**（vs RAG）、**+0.04 Emb（+15.0%）**（vs RAG）。 |
| 💸 **短对话 token 效率** | **Ours (textual)** 平均仅 **471** 个 prompt token，比 CFG **少 43%**、不到 PAG 的 **一半**，同时 Sem 领先。 |
| 🧩 **Parametric Adaptation** | **Ours (parametric)** 在零额外 profile-token 开销下，短对话 Sem 3.748 / 长对话 Sem 3.434 双双第一，长对话 Emb 0.283 并列第一。 |
| 🔬 **统计显著性** | 内部 PT vs. NR、PT vs. ST，以及外部"Ours vs. 同块最强 baseline"全部通过 paired *t*-test，*p* < 0.001。 |

### 头号指标（长对话，4 个语料 × {random, OOD} 的宏平均）

| 范式 | 方法 | Char ↑ | Sem ↑ | Emb ↑ |
|---|---|:---:|:---:|:---:|
| —                       | Base（无 profile）         | 2.326 | 3.323 | 0.268 |
| Textual Provision       | RAG                        | 2.405 | 3.289 | 0.273 |
| Textual Provision       | PAG                        | 2.510 | 2.889 | 0.255 |
| Textual Provision       | CFG                        | 2.389 | 2.429 | 0.225 |
| Textual Provision       | **Ours (textual)**         | **3.003** | **3.697** | **0.314** |
| Parametric Adaptation   | MT-LoRA                    | 2.269 | 3.428 | 0.283 |
| Parametric Adaptation   | Activation Steering        | 2.381 | 2.350 | 0.249 |
| Parametric Adaptation   | OPPU                       | 2.376 | 3.141 | 0.283 |
| Parametric Adaptation   | P2P                        | 2.396 | 3.410 | 0.276 |
| Parametric Adaptation   | **Ours (parametric)**      | 2.307 | **3.434** | **0.283** |

> 加粗格子表示在所属范式块内的最佳。完整逐数据集结果见 [§ 复现论文](#-复现论文)。

### 成本 vs. 质量

<div align="center">
<img src="figures/token_pareto.png" width="92%" alt="不同对话长度下的 prompt 开销 vs. 质量"/>
<p><sub><b>图 3.</b> Prompt token 开销（最左列条形图，越短越省）与 Char / Sem / Emb 指标的对照。短对话场景下 <b>Ours (textual)</b> 是所有"带 profile"方法里最省 token 的，同时 Sem 领先；长对话场景下 prompt 因要承载累积的演化历史而变长，但仍在 Sem 与 Emb 上取得对比中所有方法的最佳值。所有 parametric-adaptation 方法都收敛到 context-only 的 token 开销。</sub></p>
</div>

---

## 📁 仓库结构

```
PHASE-Tree/
├── preprocessing/         # 逐语料的角色档案抽取 + 对话格式化脚本
├── src/
│   ├── tree_pipeline/     # 构建 raw → static → dynamic → PHASE-Tree 档案
│   ├── hyper_llm_modulator/  # Hyper-LoRA 调制器（编码器、mixer、输出头）
│   ├── scripts/           # 超网络训练入口 + 启动脚本
│   └── configs/           # 训练 YAML
├── tasks/                 # 每个 split 的 metadata YAML（被 SFT 训练器消费）
├── evaluation/            # predict_* + judge.py + report.py + visualize.py
├── figures/               # 论文配图
├── .env                   # API key 占位模板——本地填入后再使用
├── requirements.txt       # 核心 Python 依赖
├── requirements-flash-attn.txt   # 可选的 FlashAttention-2（推荐安装）
└── LICENSE                # MIT
```

> ⚠️ 三个大目录**不**纳入 Git，需要从 Hugging Face 拉取：`phase_tree_data/`、`phase_tree_models/`，以及（可选）`results/`。

---

## 🤗 发布的资源

| 类型     | 仓库 | Hugging Face | 默认本地路径 | 大约体积 |
|---------|------|--------------|--------------|----------|
| 数据集   | `IAAR-Shanghai/phase_tree_data`        | [🤗 链接](https://huggingface.co/datasets/IAAR-Shanghai/phase_tree_data) | `phase_tree_data/`   | ≈ 8.4 GB |
| 模型     | `IAAR-Shanghai/phase_tree_models`      | [🤗 链接](https://huggingface.co/IAAR-Shanghai/phase_tree_models)       | `phase_tree_models/` | ≈ 1.7 GB |
| 评测结果 | `Mathematics-Yang/phase_tree_results`  | [🤗 链接](https://huggingface.co/datasets/Mathematics-Yang/phase_tree_results) | `results/`           | ≈ 4.4 GB |

**一键下载（在仓库根目录执行）：**

```bash
hf download IAAR-Shanghai/phase_tree_data    --repo-type=dataset --local-dir phase_tree_data
hf download IAAR-Shanghai/phase_tree_models                       --local-dir phase_tree_models
# 可选 —— 仅当你想跳过重新跑 prediction / judge 时下载：
hf download Mathematics-Yang/phase_tree_results --repo-type=dataset --local-dir results
```

你还需要把两个基座模型放到磁盘上（默认期望路径在 `models/` 下）：

```bash
hf download Qwen/Qwen2.5-7B-Instruct  --local-dir models/Qwen2.5-7B-Instruct
hf download Qwen/Qwen3-Embedding-4B   --local-dir models/Qwen3-Embedding-4B
```

---

## 🔧 安装

测试环境：**Python 3.10 + CUDA 12.x (Linux)**。Textual-provision 路径 1 张 A100/H100 即可；超网络 SFT 训练推荐 1 张 A100 80 GB。

```bash
git clone https://github.com/MemTensor/PHASE-Tree.git
cd PHASE-Tree

python -m venv .venv && source .venv/bin/activate

# 1) 核心依赖（torch、transformers、peft、vllm、openai ……）
pip install -r requirements.txt

# 2)（可选，推荐）FlashAttention-2 内核
#    必须在 requirements.txt 安装完成之后再装，并加 --no-build-isolation
pip install -r requirements-flash-attn.txt --no-build-isolation
```

如果你的环境无法构建 FlashAttention，代码会自动回退到 `attn_implementation="sdpa"`——吞吐略降，但训练和推理仍可正常运行。

> **PyTorch / CUDA 不匹配？** 先按 [官方选择器](https://pytorch.org/get-started/locally/) 装一份与本机驱动匹配的 PyTorch wheel，再装 `requirements.txt`。

---

## 🔑 API 配置

所有依赖 API 的步骤（角色档案抽取、人格演化、LLM-as-Judge、Embedding 相似度）都通过 `python-dotenv` 从仓库根目录的 `.env` 读取凭据。仓库自带的 `.env` 是**占位符模板**：

```bash
# 方法 A —— 直接在 .env 里填，注意不要把真实 key 提交到仓库
$EDITOR .env

# 方法 B —— 保留 .env 作为公开模板，把真实 key 写到本地副本（推荐）
cp .env .env.local
$EDITOR .env.local         # `.env.local` 已被 git-ignore
```

三组环境变量可以指向同一个 OpenAI-兼容 endpoint，也可以分别配置（例如 OpenAI 跑 judge + 本地 vLLM 跑 embedding）：

| 变量组      | 被谁使用 |
|------------|---------|
| `LLM_*`    | `preprocessing/*.py`、`src/tree_pipeline/*.py`（档案抽取 + 人格演化） |
| `JUDGE_*`  | `evaluation/judge.py`（LLM-as-Judge：Char + Sem） |
| `EMBED_*`  | `evaluation/judge.py`（预测 vs 参考的 cosine 相似度） |

---

## 🚀 快速上手

在一个短对话数据集上跑完整端到端流程（两条部署路径都走一遍）：

```bash
# 0) 拉数据 + 检查点（只需一次）
hf download IAAR-Shanghai/phase_tree_data   --repo-type=dataset --local-dir phase_tree_data
hf download IAAR-Shanghai/phase_tree_models                       --local-dir phase_tree_models

# 1) Textual provision：在 RAIDEN 上跑 predict + judge + report（两个 split）
bash evaluation/run_prompt_eval.sh RAIDEN

# 2) Parametric adaptation（超网络 → LoRA）：在 RAIDEN 上跑 predict + judge + report
bash evaluation/run_phase_tree_eval.sh RAIDEN

# 3) 一个外部 baseline（例如 RAG）作为对照
bash evaluation/run_comparison_eval.sh RAIDEN rag
```

每个启动脚本会读取 `phase_tree_data/processed/<DATASET>/<METHOD>/{random_test,ood_test}.json`，把预测写到 `results/<DATASET>/<paradigm>/main/<METHOD>/<SPLIT>/predictions.jsonl`，然后依次串起来 `judge.py → report.py → visualize.py`。

---

## 🧬 Pipeline

PHASE-Tree 是一个 4 阶段的流水线，每个阶段都可以独立运行。我们已经把第 1、2 阶段的产出（在 `phase_tree_data/`）和第 3 阶段的产出（在 `phase_tree_models/`）打包好了，你可以从任意一个阶段插入。

### Stage 1 · 逐语料预处理

`preprocessing/` 下每个源语料都对应一对脚本：**档案抽取** + **对话转换**。产物写到 `phase_tree_data/processed/<Dataset>/intermediate/`。

```bash
# 以 Friends（长对话）为例：从第 1 季对白播种初始角色档案
python preprocessing/extract_profiles_friends.py

# 把 1–10 季的剧本转成"下一句生成"任务并做时序切分
python preprocessing/preprocess_dialogues_friends.py
```

短对话语料（`RAIDEN`、`CharacterEval`、`SimsConv`、`ChatHaruhi`）用人格聚类切分；长对话语料（`Friends`、`HPD`、`StarTrek_TNG`、`TheOffice`）用按时间顺序的 train / OOD-temporal 切分。

### Stage 2 · 构建 PHASE-Tree 的 6 个档案变体

消融链的 6 个档案变体放在 `phase_tree_data/processed/<Dataset>/` 下：

| 变体               | 含义                                                     | 论文标签 |
|--------------------|----------------------------------------------------------|----------|
| `m1_context_only`  | 不带 profile —— 仅有对话上下文。                          | Base     |
| `m2_raw_profile`   | 原始抽取的档案文本。                                       | RP       |
| `m3_naive_rewrite` | 用 LLM 改写过的档案，无结构。                              | NR       |
| `m4_static_tree`   | 扁平化 PHASE-Tree，仅含 identity + persona。              | ST       |
| `m5_dynamic_tree`  | 跨剧集演化、但只演化 persona（仅长对话）。                  | DT       |
| `m6_phase_tree`    | **完整 PHASE-Tree**：identity + 演化 persona + session + moment。 | **PT（本文）** |

长对话的演化编排器（`pipeline_evolve_full`）会顺序执行：Stage A 证据累积 → Stage B 阻力门控更新 → Stage C 确定性后处理 patches：

```bash
# 全流程（LLM 演化 + 所有 patch）
python -m src.tree_pipeline.pipeline_evolve_full --dataset Friends

# 只跑 patches（跳过昂贵的 LLM 演化步骤）
python -m src.tree_pipeline.pipeline_evolve_full --dataset Friends --skip_evolve

# 把参数转给内部演化脚本（并行度、单集测试 ……）
python -m src.tree_pipeline.pipeline_evolve_full --dataset Friends --workers 8 --test_episode S05E10
```

单个子步骤（`evolve_persona`、`decay_stale_romantic`、`repair_inter_main_reciprocity`、`align_inverse_pair`、`forward_fill_continuity` ……）也可独立运行，详见 `pipeline_evolve_full.py` 顶部 docstring。

### Stage 3 · 超网络训练（Implicit Parametric Adaptation 路径）

Hyper-LoRA 调制器是一个把"角色档案 → 适配器权重"的超网络，包在 `Qwen2.5-7B-Instruct` 外面。我们提供的启动脚本会以 **warm-start** 的方式在 8 个 PHASE-Tree 训练集上做 SFT：

```bash
# 用发布的预训练 hypermod 做 warm-start（默认 INIT_CKPT）
bash src/scripts/train_phase_tree_qwen_7b.sh

# 从零训（不 warm-start）
INIT_CKPT="" bash src/scripts/train_phase_tree_qwen_7b.sh

# 覆盖超参
LR=1e-5 EPOCHS=20000 WARMUP=0.1 bash src/scripts/train_phase_tree_qwen_7b.sh
```

默认配置在 `src/configs/phase_tree_hyper_lora.yaml`（lr 5e-6、warmup 0.05、40 000 步、hierarchical batch sampler、sqrt-size mixture）。Checkpoint 会写到 `phase_tree_models/sft/<run>/{hypermod.pt, args.yaml, adapter_config.json}`，可被推理时的同一套 checkpoint reader 加载。

### Stage 4 · 推理

| 范式 | 脚本 | 作用 |
|------|------|------|
| Textual Provision — **Ours (textual)**     | `evaluation/predict_prompt.py`     | 把 PHASE-Tree 序列化进 prompt；vLLM 或 HF 后端均可。 |
| Parametric Adaptation — **Ours (parametric)** | `evaluation/predict_phase_tree.py` | 用 PHASE-Tree SFT hypermod 生成角色级 LoRA；*profile 不进 prompt*。 |
| Parametric Adaptation — P2P（baseline）        | `evaluation/predict_hypernet.py`   | 同样架构，但用原始档案训练的 P2P hypermod。 |
| 外部 baseline：RAG / PAG / CFG / Steering / MT-LoRA / OPPU | `evaluation/predict_{rag,cfg,steering,mt_lora,oppu}.py`（PAG 复用 `predict_rag.py` + `--profile_data`） | 两种范式各 3 个对照方法。 |

推荐用法是各范式自带的**启动脚本**——它会自动判断短对话/长对话、把任务分发到多 GPU，并依次跑 `predict → judge → report → visualize`：

```bash
# Textual Provision —— 消融链（短对话 m1+m2+m3+m4+m6；长对话再加 m5）
bash evaluation/run_prompt_eval.sh <DATASET>

# Parametric Adaptation —— PHASE-Tree SFT hypermod，所有方法、两个 split、多 GPU
bash evaluation/run_phase_tree_eval.sh                       # 全部 8 个数据集
bash evaluation/run_phase_tree_eval.sh Friends long-term     # 单数据集，显式指定模式

# Parametric Adaptation —— P2P 预训练 baseline
bash evaluation/run_hypernet_p2p_eval.sh <DATASET>

# 外部 baseline
bash evaluation/run_comparison_eval.sh <DATASET> rag         # 也支持：pag, cfg, steering, mt_lora, oppu
```

### 评测指标

只要 `predictions.jsonl` 存在，`evaluation/judge.py` 就会写出两个评分文件（支持断点续跑）：

| 文件                       | 取值范围 | 测什么 |
|----------------------------|----------|--------|
| `judge_scores.jsonl`       | 1 – 5    | `character_score`（角色档案一致性）+ `semantic_score`（上下文连贯性）。GPT-4.1 LLM-as-Judge，rubric 见 `evaluation/persona_rubric.md`。 |
| `embedding_scores.jsonl`   | [-1, 1]  | 用 `text-embedding-3-small` 算预测 vs. 参考的 cosine 相似度。 |

汇总 + 可视化：

```bash
python evaluation/report.py    --results_dir results/RAIDEN/prompt/main --baseline m2_raw_profile --per_character
python evaluation/visualize.py --results_dir results/RAIDEN/prompt/main --format pdf
python evaluation/autoreport.py                                          # 跨数据集汇总
```

参考侧的一个消融（用 `m2_raw_profile` 而不是完整 PHASE-Tree 作为 judge 的角色参考重新评分）见 `evaluation/run_ablation.sh`。

---

## 📊 LongEvoRoleBench

八个角色扮演语料库被统一标准化为下一句生成任务，每个数据集都带有**随机**与 **OOD** 两个 test 切分：

| 数据集        | 语言   | 流水线类型 | 主要角色数 | OOD 维度 |
|---------------|--------|-----------|:----------:|----------|
| CharacterEval | 中文   | 短对话     | 77 | 未见角色聚类 |
| ChatHaruhi    | 中+英  | 短对话     | 31 | 未见角色聚类 |
| RAIDEN        | 中文   | 短对话     | 30 | 未见角色聚类 |
| SimsConv      | 英文   | 短对话     | 68 | 未见角色聚类 |
| Friends       | 英文   | 长对话     |  6 | 后期季的时序留出 |
| HPD（Harry Potter Dialogue）| 英文 | 长对话 | 6 | 后期书的时序留出 |
| StarTrek_TNG  | 英文   | 长对话     |  6 | 后期季的时序留出 |
| TheOffice     | 英文   | 长对话     |  6 | 后期季的时序留出 |

每个 `tasks/<name>/metadata.yaml` 都会向超网络 SFT 训练器注册一个 split；同一份 JSON 也直接被预测脚本消费。生成与评分都基于同一时刻 `t` 的角色状态，所以 Character Score 衡量的是"当前状态"，而不是"冻结的初始档案"。

---

## 🧪 复现论文

复现论文中每一个数字：

```bash
# 0) 准备好数据 + 推荐的 SFT 检查点
hf download IAAR-Shanghai/phase_tree_data   --repo-type=dataset --local-dir phase_tree_data
hf download IAAR-Shanghai/phase_tree_models                       --local-dir phase_tree_models

# 1) 在 8 个语料上跑所有 textual-provision 消融
for D in RAIDEN CharacterEval HPD SimsConv ChatHaruhi Friends StarTrek_TNG TheOffice; do
    bash evaluation/run_prompt_eval.sh "$D"
done

# 2) 所有 parametric-adaptation 消融
bash evaluation/run_phase_tree_eval.sh        # PHASE-Tree SFT hypermod —— Ours (parametric)
bash evaluation/run_hypernet_p2p_eval.sh      # P2P baseline

# 3) 外部 baseline（RAG、PAG、CFG、Steering、MT-LoRA、OPPU）
for D in RAIDEN CharacterEval HPD SimsConv ChatHaruhi Friends StarTrek_TNG TheOffice; do
    for M in rag pag cfg steering mt_lora oppu; do
        bash evaluation/run_comparison_eval.sh "$D" "$M"
    done
done

# 4) 一键汇总成论文风格的总表
bash evaluation/run_autoreport.sh
```

Token-pareto 图（论文 Fig. 3）由 `evaluation/make_token_figures.py` 生成，前提是各 `results/<DATASET>/<paradigm>/main/` 下已有 `summary.json`。

如果你只想看我们报告的数字，可以直接下载预计算结果包：

```bash
hf download Mathematics-Yang/phase_tree_results --repo-type=dataset --local-dir results
bash evaluation/run_autoreport.sh
```

---

## ⚠️ 局限性

论文 § Limitations 中讨论的已知问题：

- **单一 backbone、单次运行、语料覆盖有限**：所有实验都用 `Qwen2.5-7B-Instruct` 在一种解码配置和一个 seed 下完成，没有做跨模型族、跨规模、多 seed 或多语言的验证；长对话部分只覆盖 4 个英文剧本/小说语料，更宽的题材和自发对话尚未测试。
- **基于 LLM 的评测与提取**：Char、Sem 依赖 GPT-4.1 judge，没有人类一致性研究；状态抽取和更新判断也由 LLM 完成，未用 gold 标注审计；显著性检验是单次运行的 question-level 配对检验，无法反映 run-to-run 方差。
- **手工调的门控 + parametric adaptation**：阻力、证据、冷却阈值是实现默认值，没有 sweep 也没有学习版本；隐式 parametric adaptation 变体走的是 P2P 风格架构，没有把多时间尺度状态完全表达出来。

---

## 🙏 致谢

Hyper-LoRA 调制器架构基于 P2P 代码库 \[[Tan et al., 2025](https://arxiv.org/abs/2501.04652)\]。八个评测语料来自此前的公开发布——RAIDEN、CharacterEval、SimsConv、ChatHaruhi、HPD、[ConvoKit Friends Corpus](https://convokit.cornell.edu/documentation/friends.html)、以及 The Office 与 Star Trek: TNG 的公开剧本——感谢原始作者与维护者把这些资源开放出来。生成 backbone（`Qwen2.5-7B-Instruct`）和 embedding 编码器（`Qwen3-Embedding-4B`）来自 [Qwen 团队](https://github.com/QwenLM)。

---

## 📄 许可证

- 本仓库的**代码**采用 [MIT License](LICENSE) 发布。
- 发布在 Hugging Face 上的**模型检查点**与**评测结果**采用 **CC-BY-NC-4.0** —— 详见各自 Hugging Face 页面的 model / dataset card。
- 底层**对话语料**保留各自的原始许可证；二次分发请参阅各源数据集的条款。

---

<div align="center">
<sub>⭐ 如果这份工作对你有帮助，欢迎 Star——这能帮助更多人发现它。</sub>
</div>
