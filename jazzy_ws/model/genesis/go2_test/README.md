# GO2 Walking Policy (Pretrained)

This folder contains a pretrained reinforcement learning policy for the Go2 quadruped robot trained in the Genesis simulator.

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

- **go2_config.txt**  
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
 logs/go2_test/pretrained/
```

 The structure should then be:

```bash
logs/
└── go2_test/
    └── pretrained/
        ├── model_final.pt
```

### 2️⃣ Run play

```bash
python -m robot_gym.scripts.play \
    --task go2 \
    --experiment_name go2_test \
    --load_run pretrained
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
    "go2",
    Go2Env,
    GO2Cfg(),
    GO2CfgPPO(),
)
```

-> This should already be done initially but if not double check please.
