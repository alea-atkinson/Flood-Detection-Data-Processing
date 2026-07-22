# Seed Replication Experiments

Purpose:
Test whether weak SimSiam's LOFPO improvement over random initialization is stable across downstream fine-tuning seeds.

Comparison:
- random initialization
- weak SimSiam initialization

Folds:
- fp1 through fp7

Seeds:
- 1, 2, 3

Total planned runs:
- 7 folds × 2 methods × 3 seeds = 42 downstream fine-tuning runs

Do not overwrite previous LOFPO outputs.

Local/generated outputs:
- logs/
- models/
- raw_results/

Tracked/report outputs:
- scripts/
- slurm/
- tables/
- figures/
- summaries/
