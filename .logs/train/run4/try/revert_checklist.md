# ConservativeMix revert checklist
1. Delete pufferlib/conservative/
2. Delete both drive_conservative_mix.ini files
3. Delete tests/test_conservative_*.py
4. Remove drive_conservative_mix entry from MAKE_FUNCTIONS
5. python -c "from pufferlib.pufferl import load_config; load_config('puffer_drive')"
6. pytest tests/test_mix_ppo.py tests/test_adversarial_mix_env.py -q
