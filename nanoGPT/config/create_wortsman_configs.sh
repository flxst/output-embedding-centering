# Usage: cd nanoGPT/config; bash create_wortsman_configs.sh
# Arguments: METHOD GAMMA(=LAMBDA) SEED

SEED=1

bash create_wortsman_configs_helper.sh A 0.0 ${SEED}
bash create_wortsman_configs_helper.sh E 0.0001 ${SEED}
bash create_wortsman_configs_helper.sh R 0.0 ${SEED}
bash create_wortsman_configs_helper.sh Z 0.0001 ${SEED}