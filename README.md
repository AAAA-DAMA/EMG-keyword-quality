声明
本仓库代码对应论文《面向中文科技文献的关键词质量评估方法》的官方实现。论文已投稿至《北京邮电大学学报（自然科学版）》并处于审稿阶段。仓库公开的目的在于增强研究透明性、回应审稿意见并支持学术交流。

未经作者书面许可，任何个人或组织不得：
1.将本仓库代码用于商业用途；
2.转载、再发布或镜像分发本仓库全部或部分代码；
3.修改后作为本人或本团队原创成果进行投稿或申报；
4.删除或篡改本仓库中的作者署名、论文信息与版权声明。

如需转载、合作或进一步使用，请先联系作者。
# EMG-keyword-quality

This repository contains the code used for the paper on keyword quality evaluation for Chinese scientific literature.

## Repository structure

- `src/emg_main.py`: main EMG training/evaluation script used for the core experiments.
- `src/baseline_main.py`: baseline BERT/RoBERTa training script.
- `src/dynamic_compare_gmu.py`: dynamic interaction comparison script (GMU-oriented version).
- `src/dynamic_compare_concat.py`: static concat / dynamic comparison script.
- `src/emg_sensitivity.py`: hyperparameter sensitivity version for `Pdel` and `delta`.
- `src/emg_sensitivity_schedule.py`: schedule on/off sensitivity version.
- `src/emg_efficiency.py`: EMG efficiency profiling script.
- `src/baseline_efficiency.py`: baseline efficiency profiling script.

- `preprocessing/build_hard_csl_dataset.py`: build CSL-style hard-negative data.
- `preprocessing/clean_csl_dataset.py`: post-clean generated CSL-style datasets.
- `preprocessing/build_fintech_binary_dataset.py`: reconstruct Fintech-Key-Phrase to binary keyword-quality data.

- `llm_eval/qwen_zero_shot.py`: Qwen zero-shot evaluation.
- `llm_eval/qwen_four_shot_label_scoring.py`: Qwen 4-shot label scoring evaluation.

- `scripts/run_main_4models_clean.py`: run four main models on clean CSL-style data.
- `scripts/run_fintech_4models.py`: run four models on reconstructed external-domain data.
- `scripts/run_fintech_4models_clean.py`: clean external-domain four-model runner.
- `scripts/run_noise_eval.py`: train EMG and evaluate on clean/noise subsets.
- `scripts/run_keyword_length_eval.py`: keyword-length distribution evaluation.
- `scripts/run_hparam_sensitivity.py`: sensitivity runner.
- `scripts/recover_efficiency_from_logs.py`: recover efficiency metrics from logs.

- `archive/original_exploration/`: earlier exploratory scripts kept for completeness but not required for the main paper reproduction.

## Recommended minimal reproduction order

1. Prepare CSL-format data with fields: `id`, `abst`, `keyword`, `label`.
2. Run `src/baseline_main.py` and `src/emg_main.py` for the main results.
3. Run `scripts/run_main_4models_clean.py` for 4-model summary tables.
4. Run `src/dynamic_compare_gmu.py` / `src/dynamic_compare_concat.py` for dynamic interaction comparisons.
5. Run `scripts/run_noise_eval.py`, `scripts/run_keyword_length_eval.py`, and `scripts/run_fintech_4models.py` for robustness and cross-domain experiments.
6. Run `scripts/run_hparam_sensitivity.py` and efficiency scripts for Section 2.6.

## Notes

- Some paths in the scripts still use AutoDL-style absolute paths and should be changed to your local environment before public release.
- The `archive/` folder keeps intermediate variants for transparency; the main experiments should prioritize the scripts under `src/`, `preprocessing/`, `llm_eval/`, and `scripts/`.

# EMG-keyword-quality

本仓库为论文《面向中文科技文献的关键词质量评估方法》的配套实现代码。

**论文状态：已投稿至《北京邮电大学学报（自然科学版）》，目前处于审稿阶段。**

**版权声明：Copyright © 2026 作者保留所有权利。**

本仓库代码仅用于学术交流、论文评审与研究复现说明。未经作者书面许可，任何个人或组织不得将本仓库代码用于商业用途、二次分发、改写后投稿，或以任何形式据为己有。

## 仓库结构

### 1. 核心模型与训练代码（`src/`）

- `src/emg_main.py`：EMG 主模型训练与评测脚本，对应论文核心实验。
- `src/baseline_main.py`：BERT / RoBERTa 基线模型训练脚本。
- `src/dynamic_compare_gmu.py`：动态交互对照实验脚本（GMU 版本）。
- `src/dynamic_compare_concat.py`：静态拼接 / 动态交互对照实验脚本。
- `src/emg_sensitivity.py`：关键超参数敏感性分析脚本（主要用于 `Pdel` 与 `delta`）。
- `src/emg_sensitivity_schedule.py`：渐进式先验调度开关敏感性分析脚本。
- `src/emg_efficiency.py`：EMG 模型效率与部署开销统计脚本。
- `src/baseline_efficiency.py`：基线模型效率与部署开销统计脚本。

### 2. 数据构建与清洗代码（`preprocessing/`）

- `preprocessing/build_hard_csl_dataset.py`：构造 CSL 风格 hard-negative 数据集。
- `preprocessing/clean_csl_dataset.py`：对生成的 CSL 风格数据进行后处理与清洗。
- `preprocessing/build_fintech_binary_dataset.py`：将 Fintech-Key-Phrase 数据重构为二分类关键词质量评估数据。

### 3. 大语言模型评测代码（`llm_eval/`）

- `llm_eval/qwen_zero_shot.py`：Qwen 零样本（zero-shot）评测脚本。
- `llm_eval/qwen_four_shot_label_scoring.py`：Qwen 四样本（4-shot）label scoring 评测脚本。

### 4. 批量运行与结果汇总代码（`scripts/`）

- `scripts/run_main_4models_clean.py`：在 clean CSL 风格数据上运行四模型并生成汇总结果。
- `scripts/run_fintech_4models.py`：在外部领域重构数据上运行四模型。
- `scripts/run_fintech_4models_clean.py`：外部领域 clean 数据四模型运行脚本。
- `scripts/run_noise_eval.py`：训练 EMG 并在 clean / noise 子集上进行评测。
- `scripts/run_keyword_length_eval.py`：不同关键词长度分布条件下的性能评测脚本。
- `scripts/run_hparam_sensitivity.py`：关键超参数敏感性分析总控脚本。
- `scripts/recover_efficiency_from_logs.py`：从训练日志中恢复效率统计指标。


---

## 推荐的最小复现流程

1. 准备 CSL 风格的数据文件，字段格式为：`id`、`abst`、`keyword`、`label`。
2. 运行 `src/baseline_main.py` 与 `src/emg_main.py`，复现主实验结果。
3. 运行 `scripts/run_main_4models_clean.py`，生成四模型主结果汇总表。
4. 运行 `src/dynamic_compare_gmu.py` 与 `src/dynamic_compare_concat.py`，复现动态交互对照实验。
5. 运行 `scripts/run_noise_eval.py`、`scripts/run_keyword_length_eval.py` 与 `scripts/run_fintech_4models.py`，复现复杂条件与跨领域补充实验。
6. 运行 `scripts/run_hparam_sensitivity.py` 以及效率统计相关脚本，复现论文第 2.6 节实验结果。

---

## 数据格式说明

输入样本统一采用 JSON Lines 格式，每行为一个样本，格式如下：

```json
{"id": 1, "abst": "摘要文本", "keyword": ["关键词1", "关键词2"], "label": 1}
其中：
id：样本编号
abst：论文摘要或文本内容
keyword：候选关键词列表
label：二分类标签，1 表示候选关键词集合整体有效，0 表示集合中含有伪关键词或与文本语义不一致

使用说明
本仓库默认基于中文预训练模型与 JSONL 数据格式实现。
部分脚本仍保留 AutoDL 环境下的绝对路径，公开前请根据本地环境修改为相对路径或统一配置路径。
论文主结果的复现应优先使用 src/、preprocessing/、llm_eval/ 和 scripts/ 目录下的代码。
