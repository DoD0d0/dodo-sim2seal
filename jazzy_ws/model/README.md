# Dodo Policy Files

Place your trained PPO policy file here:

- `dodo_policy.pt` - TorchScript model exported from your PPO training

## Exporting Your Policy

If you trained your policy with PyTorch, export it to TorchScript:

```python
import torch

# Load your trained policy model
policy = YourPolicyNetwork()
policy.load_state_dict(torch.load('your_checkpoint.pth'))
policy.eval()

# Export to TorchScript
scripted_policy = torch.jit.script(policy)
scripted_policy.save('dodo_policy.pt')
```

## Configuration

Update `dodo_env.yaml` with your training environment parameters if needed.
