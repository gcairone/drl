TRAIN.PY (headless)
-------------------
Scopo: Addestramento PPO
Parametri:
--steps_per_epoch    int    default=2048       # passi per epoch
--epochs             int    default=500        # numero di epoch
--max_ep_len         int    default=1000       # lunghezza massima episodio
--gamma              float  default=0.99       # fattore di sconto
--clip_ratio         float  default=0.2        # clipping PPO
--lr                 float  default=3e-4       # learning rate
--lam                float  default=0.97       # lambda GAE
--update_epochs      int    default=10         # passate sul batch per ogni update
--minibatch_size     int    default=64         # dimensione minibatch (<= 0: batch intero)
--max_grad_norm      float  default=0.5        # clipping del gradiente (<= 0: disattivato)
--target_kl          float  default=0.01       # early stop se KL > 1.5 * target_kl (<= 0: disattivato)
--save_dir           str    default="results/checkpoints"   # cartella checkpoint
--save_freq          int    default=50         # salva ogni N epoch
--log_path           str    default="results/logs"          # percorso log CSV
--num_envs           int    default=8          # environment paralleli (inferenza in batch)
--seed               int    default=0          # seed per torch e environment
--obs_norm / --no-obs_norm   default=on       # normalizzazione osservazioni (media/varianza mobili)
--noise_scale        float  default=0.01       # rumore uniforme su qpos/qvel al reset
--min_height         float  default=1.0        # altezza torso sopra cui si riceve la reward
--term_height        float  default=1.0        # altezza torso sotto cui l'episodio termina
--frame_skip         int    default=3          # passi di fisica per azione (dt = 0.015 s)
--resume             str    default=None       # checkpoint da cui riprendere

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
--input   str  default="results/rollouts/humanoid_rollout.npz"   # percorso file .npz

PLAY.PY (GUI)
-------------
Scopo: riproduzione rollout visuale
Parametri:
--file    str  default="results/rollouts/humanoid_rollout.npz"  # percorso file .npz
--fps     int  default=60                                          # frame rate visualizzazione

ESEMPIO DI UTILIZZO DEGLI SCRIPT IN ORDINE
------------------------------------------
python3 -m scripts.train --save_dir results/checkpoints --log_path results/logs
python3 -m scripts.rollout --model results/checkpoints/model_epoch_500.pth --output results/rollouts/humanoid_rollout.npz
python3 -m scripts.plot_log --file results/logs/metrics.csv
python3 -m scripts.eval --input results/rollouts/humanoid_rollout.npz
python3 -m scripts.play --file results/rollouts/humanoid_rollout.npz --fps 60

