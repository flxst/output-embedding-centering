# Usage: cd nanoGPT/config; bash create_wortsman_configs.sh
# Arguments: METHOD GAMMA(=LAMBDA) SEED

SEED=1

bash create_wortsman_configs_helper.sh A 0.0 ${SEED}
bash create_wortsman_configs_helper.sh E 0.0001 ${SEED}
bash create_wortsman_configs_helper.sh R 0.0 ${SEED}
bash create_wortsman_configs_helper.sh Z 0.0001 ${SEED}
bash create_wortsman_configs_helper.sh S 30.0 ${SEED}
bash create_wortsman_configs_helper.sh W 0.0001 ${SEED}
bash create_wortsman_configs_helper.sh a 0.0 ${SEED}
bash create_wortsman_configs_helper.sh e 0.0001 ${SEED}
bash create_wortsman_configs_helper.sh r 0.0 ${SEED}
bash create_wortsman_configs_helper.sh z 0.0001 ${SEED}
bash create_wortsman_configs_helper.sh s 30.0 ${SEED}
bash create_wortsman_configs_helper.sh w 0.0001 ${SEED}