TRAIN.PY (headless)
-------------------
Scopo: Addestramento PPO
Parametri:
--steps_per_epoch    int    default=2048       # passi per epoch
--epochs             int    default=500        # numero di epoch
--max_ep_len         int    default=1000       # lunghezza massima episodio
--gamma              float  default=0.99       # fattore di sconto
--clip_ratio         float  default=0.2        # clipping PPO
--lr                 float  default=1e-4       # learning rate
--lam                float  default=0.97       # lambda GAE
--pi_iters           int    default=80         # policy update iterations
--v_iters            int    default=80         # value update iterations
--target_kl          float  default=0.01       # KL target
--save_dir           str    default="results/checkpoints"   # cartella checkpoint
--save_freq          int    default=10         # salva ogni N epoch
--log_path           str    default="results/logs"          # percorso log CSV

ROLLOUT.PY (headless)
---------------------
Scopo: generare rollout da modello
Parametri:
--output   str  default="results/rollouts/humanoid_rollout.npz"   # percorso salvataggio rollout
--steps    int  default=1000                                         # passi da simulare
--model    str  default=None                                         # percorso checkpoint PPO

PLOT_LOG.PY (headless)
----------------------
Scopo: analisi training da CSV
Parametri:
--file    str  obbligatorio            # percorso CSV log (es. log/..._metrics.csv)
--save    str  default=None             # percorso figura opzionale
--ma      int  default=10               # window size moving average

EVAL.PY (headless)
------------------
Scopo: valutazione standing time da rollout salvato
Parametri:
--input   str  default="data/humanoid_rollout.npz"   # percorso file .npz

PLAY.PY (GUI)
-------------
Scopo: riproduzione rollout visuale
Parametri:
--file    str  default="results/rollouts/humanoid_rollout.npz"  # percorso file .npz
--fps     int  default=60                                          # frame rate visualizzazione

ESEMPIO DI UTILIZZO DEGLI SCRIPT IN ORDINE
------------------------------------------
python3 train.py --save_dir results/checkpoints --log_path results/logs
python3 rollout.py --model results/checkpoints/model_epoch_500.pth --output results/rollouts/humanoid_rollout.npz
python3 plot_log.py --file results/logs/metrics.csv
python3 eval.py --input results/rollouts/humanoid_rollout.npz
python3 play.py --file results/rollouts/humanoid_rollout.npz --fps 60

