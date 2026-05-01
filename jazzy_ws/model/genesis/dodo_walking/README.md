# 🦤 Dodo Walking Policy (Pretrained)

This folder contains a pretrained reinforcement learning policy for the Dodo bipedal robot trained in the Genesis simulator.

It allows you to **run and test a working walking policy immediately** without retraining.

---

## 📦 Contents

- **model_final.pt**  
  Final training checkpoint (PyTorch).  
  → Used by the training/play pipeline.

- **policy_1.pt**  
  Exported TorchScript (JIT) policy.  
  → Used for sim2sim / sim2real deployment (e.g. C++ / ROS2).  
  → Not required for `play`, but included for convenience.

- **dodo_config.txt**  
  Snapshot of the configuration used during training.  
  → For reference only (not loaded automatically).

- **README.md**  
  This file.

---

## 🚀 How to Run (Play Mode)

The `play` script expects models inside the `logs/` directory.  
Therefore, you need to copy this folder into the correct location.

---

### 1️⃣ Copy the pretrained model

Copy this folder to:

```text
logs/<experiment_name>/<run_name>/
```

For example:

```bash
 logs/dodo_walking/pretrained/
```

 The structure should then be:

```bash
logs/
└── dodo_walking/
    └── pretrained/
        ├── model_final.pt
```

### 2️⃣ Run play

```bash
python -m robot_gym.scripts.play \
    --task dodo \
    --experiment_name dodo_walking \
```

---

## ⚠️ Important Notes

- The play script loads:

```bash
logs/<experiment_name>/<run_name>/model_*.pt
```

- If multiple checkpoints exist, the latest one is used automatically.
- The JIT file (policy_1.pt) is not used by play:
  - It is meant for deployment (sim2sim / sim2real)
  - It can be loaded independently via TorchScript

You also have to make sure, that the "dodo" task is registered in

```text
envs/__init__.py
```

```bash
task_registry.register(
    "dodo",
    DodoEnv,
    DodoCfg(),
    DodoCfgPPO(),
)
```

-> This should already be done initially but if not double check please.
