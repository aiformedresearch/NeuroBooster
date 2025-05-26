for script in /Ironman/scratch/Andrea/med-booster/MAIN/RUN_ABLATION_launched/*.sh; do
  nohup bash "$script" > "${script%.sh}.out" 2>&1 &
done
