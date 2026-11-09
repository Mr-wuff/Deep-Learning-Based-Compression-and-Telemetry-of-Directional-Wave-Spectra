# WaveSystemSetParserV5 stage-final qualification

Source: [HF WaveSystemSetParserV5 checkpoints](https://huggingface.co/wuff-mann/Wave_compress_system/tree/main/WaveSystemSetParserV5/checkpoints).

The `experiment_decision.json` record reports `pass_iou_gate=true`, `pass_count_gate=true`, and recommends **a later faithful benchmark against traditional methods**. The per-case summary is included in `final_validation_by_case.csv`.

These are successful **stage-specific evaluation artifacts**; they are not sufficient evidence that the final paper's traditional-method comparison or 1,200-case stress test has been fully archived.

All values are unchanged from the public HF file. For byte-exact retrieval, run `python scripts/fetch_artifacts.py --group reports`.
