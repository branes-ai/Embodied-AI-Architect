Run SWaP-C smoke tests:
1. `.venv/bin/pytest tests/test_swap_profiles.py tests/test_swap_analysis.py tests/test_swap_cli.py -v`
2. `.venv/bin/branes swap estimate --area 50 --power 5 --process 28`
3. `.venv/bin/branes swap score --area 50 --power 5 --process 28 --profile drone`
4. `.venv/bin/branes swap sensitivity --area 50 --power 5 --process 28 --mode tornado`
Report results concisely.
