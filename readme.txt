
drl_humanoid/
│
├── assets/
│   └── humanoid.xml           # modello standard mujoco
│
├── envs/
│   ├── __init__.py
│   └── humanoid_env.py         # lo abbiamo
│
├── agents/
│   ├── __init__.py
|   |── policy.py               # lo abbiamo
│   └── ppo.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── config.py
│
├── scripts/
│   ├── train.py
│   ├── rollout.py              # lo abbiamo
│   ├── play.py              # lo abbiamo
│   └── test_mujoco.py         # lo abbiamo
│
├── models/                    # checkpoint
│
└── README.md



Funziona test_mujoco.py con
hostuser@137e5a6a1e40:/workspace/drl$ python3 -m scripts.test_mujoco

Funziona rollout + play con 
hostuser@137e5a6a1e40:/workspace/drl$ python3 -m scripts.rollout
hostuser@137e5a6a1e40:/workspace/drl$ python3 -m scripts.play
----------------------------------------------------------