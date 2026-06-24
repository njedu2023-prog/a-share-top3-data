# A-share Top10 Chain Orchestrator

这个目录是三仓库链路的外部编排器，不包含业务模型逻辑。

执行顺序：

1. `njedu2023-prog/a-share-top3-data` / `Daily Data Fetch`
2. `njedu2023-prog/a-top10` / `Run Top10 Engine (Auto Daily)`
3. `njedu2023-prog/top10-decision` / `Run Top10 Decision (Auto Daily)`

编排器会等待每一步 GitHub Actions 成功，并验证对应的发布入口可访问后，才执行下一步。

## 定时

`.github/workflows/orchestrate_top10_chain.yml` 每周一到周五北京时间 `19:10` 运行。

脚本会先用 A 股交易日历判断当天是否交易日；非交易日直接跳过。

## 必需 Secret

在 `a-share-top3-data` 仓库配置：

- `ORCHESTRATOR_TOKEN`
  - 需要能触发并读取三个仓库的 GitHub Actions。
  - 建议使用 fine-grained PAT，授权仓库：
    - `njedu2023-prog/a-share-top3-data`
    - `njedu2023-prog/a-top10`
    - `njedu2023-prog/top10-decision`
  - 权限：
    - Actions: Read and write
    - Contents: Read

- `TUSHARE_TOKEN`
  - 已被数据抓取流程使用。
  - 编排器用它读取 `trade_cal`，判断 A 股交易日。

## 失败策略

- 任一步失败，立即停止，不执行下游。
- 输出失败仓库、workflow run URL、失败 job/step 和日志摘要。
- 原有三个系统的自动 schedule 保留，不受这个编排器影响。

## 手动执行

在 GitHub Actions 中运行 `Orchestrate A-share Top10 Chain`。

可选输入：

- `trade_date`: 指定 `YYYYMMDD`
- `skip_calendar`: 跳过交易日历检查，强制执行
