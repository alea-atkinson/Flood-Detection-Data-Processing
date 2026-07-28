Drought Augmentation Handoff for Mac

Main experiment:
Random initialization vs drought-negative initialization under strict Florence LOFPO.

Important interpretation:
Drought all-zero labels are assumed-negative weak labels, not verified no-flood ground truth.

Expected files:
- test_metrics/fp1_random_init_test_metrics.csv ... fp7_random_init_test_metrics.csv
- test_metrics/fp1_drought_negative_init_test_metrics.csv ... fp7_drought_negative_init_test_metrics.csv

Next step on Mac:
Aggregate test metrics by fold and compare mean Dice, IoU, precision, recall, and drought-minus-random deltas.

Do not use the smoke-test files as final results.
