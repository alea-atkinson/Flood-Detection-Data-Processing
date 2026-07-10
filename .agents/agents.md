Important environment and execution rules for this repository:

1. Do not create, activate, modify, repair, or use `.venv` unless I explicitly request it.

2. Do not install or upgrade Python packages.

3. Do not run full model training jobs.

4. Do not run commands that may take more than a few minutes.

5. Use `python3`, not `.venv/bin/python`, for lightweight script validation.

6. For CUDA training, assume I will run the final training command manually in my own terminal environment.

7. When modifying a training script:

   * inspect the code;
   * make the requested changes;
   * run only lightweight checks such as syntax validation, argument parsing, imports, or a very small smoke test;
   * do not start full training.

8. After making changes, provide the exact terminal command I should run manually.

9. Never silently fall back to CPU for a training run.

10. The training script should fail immediately if CUDA is required but unavailable.
