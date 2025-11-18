# Project-ML
An implementation of standard ML algorithms applied to taxy v3 of gymnasium
Machine Learning Project - Taxi-v3 (Q-learning, DQN, Double DQN)
================================================================

Setup:
1) crea env (opzionale con conda):
   conda env create -f environment.yml
   conda activate taxi_rl
   oppure:
   pip install -r requirements.txt

2) struttura project_root come descritto.

Eseguire singoli training:
python src/train_q.py --config config.yaml --seed 0
python src/train_dqn.py --config config.yaml --seed 0
python src/train_double_dqn.py --config config.yaml --seed 0

Eseguire tutti i run (multi-seed) e salvare summary:
python src/run_experiments.py --config config.yaml

Generare figura comparativa:
python src/results_aggregator.py

Valutare un modello salvato:
python src/evaluate.py --config config.yaml --path results/dqn/seed_0 --method dqn --episodes 200
