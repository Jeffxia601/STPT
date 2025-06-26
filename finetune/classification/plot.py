import pickle
import matplotlib.pyplot as plt

# Define paths to the files
file_paths = {
    "base_no_pretrain": "new_loss/base_model_ViT_5_class_0.pkl",
    "base_pretrain_5k": "new_loss/base_model_ViT_5_class_halfyear_5000_1_epochs_1.pkl",
    "base_pretrain_10k": "new_loss/base_model_ViT_5_class_halfyear_10000_1_epochs_2.pkl",
    "base_pretrain_20k": "new_loss/base_model_ViT_5_class_halfyear_20000_1_epochs_3.pkl",
    "large_no_pretrain": "new_loss/large_model_ViT_5_class_0.pkl",
    "large_pretrain_5k": "new_loss/large_model_ViT_5_class_halfyear_5000_1_epochs_1.pkl",
    "large_pretrain_10k": "new_loss/large_model_ViT_5_class_halfyear_10000_1_epochs_2.pkl",
    "large_pretrain_20k": "new_loss/large_model_ViT_5_class_halfyear_20000_1_epochs_3.pkl"
}

# Load data from pickle files
data = {}
for key, path in file_paths.items():
    with open(path, "rb") as file:
        data[key] = pickle.load(file)

# Data for plotting test accuracy comparison
test_acc_data = {
    'base': {
        'samples': ['0k', '5k', '10k', '20k'],
        'test_loss': [0.64778, 0.60289, 0.57179, 0.57198],
        'test_acc': [0.78125, 0.7875, 0.8, 0.79375]
    },
    'large': {
        'samples': ['0k', '5k', '10k', '20k'],
        'test_loss': [0.53894, 0.39583, 0.41421, 0.41795],
        'test_acc': [0.76875, 0.8625, 0.8125, 0.83125]
    }
}

# Combined plotting function for test accuracy
def plot_combined_test_accuracy(data, ax):
    ax.plot(data['base']['samples'], data['base']['test_acc'], marker='o', label='Base Model')
    ax.plot(data['large']['samples'], data['large']['test_acc'], marker='o', label='Large Model')
    ax.set_title('Test Accuracy Comparison')
    ax.set_xlabel('Pretraining Data Size')
    ax.set_ylabel('Test Accuracy')
    ax.legend()

# Combined plotting function for validation loss and accuracy
def plot_validation_loss_and_accuracy(data, model_type, ax1):
    epochs = len(data[f"{model_type}_no_pretrain"][1])  # Assuming all data lengths are the same
    x = range(1, epochs+1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Plot validation loss for each dataset
    for i, key in enumerate(['no_pretrain', 'pretrain_5k', 'pretrain_10k', 'pretrain_20k']):
        ax1.plot(x, data[f"{model_type}_{key}"][1], label=f'Loss ({key})', linestyle=':', linewidth=1, alpha=1)
    ax1.tick_params(axis='y')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    
    # Plot validation accuracy for each dataset
    for i, key in enumerate(['no_pretrain', 'pretrain_5k', 'pretrain_10k', 'pretrain_20k']):
        ax2.plot(x, data[f"{model_type}_{key}"][3], label=f'Accuracy ({key})', linewidth=1, alpha=1)
    ax2.tick_params(axis='y')
    
    ax1.set_title(f'{model_type.capitalize()} Model Metrics')
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

# Create a single figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plotting the combined test accuracy
plot_combined_test_accuracy(test_acc_data, axes[0])

# Plotting the combined validation loss and accuracy for base and large models
plot_validation_loss_and_accuracy(data, "base", axes[1])
plot_validation_loss_and_accuracy(data, "large", axes[2])

# fig.text(0.18, -0.05, '(a)', ha='center')
# fig.text(0.5, -0.05, '(b)', ha='center')
# fig.text(0.82, -0.05, '(c)', ha='center')

plt.tight_layout()
plt.savefig("task_1.png")
plt.show()
