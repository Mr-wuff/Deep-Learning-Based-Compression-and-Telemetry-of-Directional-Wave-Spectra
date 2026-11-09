# ICWDS 波浪方向谱压缩回传研究：复现仓库

本仓库用于整理 **Deep Learning-Based Compression and Telemetry of Directional Wave Spectra** 论文所涉及的正式模型、数据和实验归档，**不是**“所有实验都复现成功”的认证文件。

- 方向谱网格：47 × 72；传输报文：严格30字节（28B载荷+2B CRC）。
- 数据主仓库：[ATLAS-WDS-v4](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v4)、[ATLAS-WDS-v2](https://huggingface.co/datasets/wuff-mann/ATLAS-WDS-v2)。
- 模型及实验报告：[Wave_compress_system](https://huggingface.co/wuff-mann/Wave_compress_system)。
- 公开可定位的最终冻结工程：C7C5；但其总体跨域指标存在负收益，不能夸称所有条件下都优于Direct。
- 生产稳定性审计有明确不合格记录（见英文证据矩阵），不得当作成功结果。
- 论文中“2006–2009年、35站、1200条”压力测试的独立原始结果包与脚本尚未核实；图表数字不能代替原始报告。

执行 `python scripts/fetch_artifacts.py --list` 查看核实过的正式文件路径；`--group reports` 下载原始HF报告。

论文投稿的 Data Availability Statement 草稿位于 [docs/DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)，其中待填内容必须由作者核验后才能用于正式投稿。
