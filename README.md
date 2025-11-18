# Project-ML
An implementation of standard ML algorithms applied to taxy v3 of gymnasium
Machine Learning Project - Taxi-v3 (Q-learning, DQN, Double DQN)
===============================================================

Requisiti (installare in virtualenv / conda):
- Python 3.8+
- gymnasium
- numpy
- torch (PyTorch)
- matplotlib

Esempio di installazione pip:
pip install gymnasium numpy torch matplotlib

Struttura:
  utils.py
  q_learning.py
  dqn.py
  double_dqn.py
  train_q.py
  train_dqn.py
  train_double_dqn.py
results/      <- output (creare o generato automaticamente)

Comandi per eseguire (da /src o dalla root, modificare i path -- gli argomenti sono opzionali):
python src/train_q.py --episodes 5000 --save_dir results/q_learning
python src/train_dqn.py --episodes 2000 --save_dir results/dqn
python src/train_double_dqn.py --episodes 3000 --save_dir results/double_dqn

File di output generati:
- results/<method>/learning_curve.png
- results/<method>/rewards.npy
- results/<method>/losses.npy (per DQN / Double DQN)
- results/<method>/dqn.pt or double_dqn.pt (modello salvato)

Suggerimenti:
- Usare seed per riproducibilità (--seed)
- Eseguire più volte gli esperimenti e prendere medie/varianze
- Salvare iperparametri sperimentati in report (tabella)
