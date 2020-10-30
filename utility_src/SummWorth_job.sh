#!/bin/bash
#SBATCH --job-name=SummWorth_Parser
#SBATCH -A research
#SBATCH -c 25
#SBATCH -G 1
#SBATCH --gres=gpu:0
#SBATCH -o SummWorth_Experiment_Logs.out
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

echo "Begin"
cd ./SummWorth
python run_experiment.py
echo "Done"

