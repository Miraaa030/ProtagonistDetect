import json
import matplotlib.pyplot as plt


with open('training_history.json') as f:
    h = json.load(f)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))


ax1.plot(h['train_loss'], label='train')
ax1.plot(h['val_loss'], label='val')
ax1.set_title('Loss curve')
ax1.legend()
ax1.grid(True)


ax2.plot(h['val_protagonist_ratio'], label='protagonist_ratio', color='orange')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='ideal~0.5')
ax2.set_title('Protagonist ratio')
ax2.legend()
ax2.grid(True)


plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
