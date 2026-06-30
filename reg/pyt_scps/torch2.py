import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. Generate (x, y) data pairs
# ==========================================
def generate_data(func, x_range=(-5, 5), n_samples=1000, noise=0.1):
    """
    Generate training data with noise
    func: ground truth function
    x_range: range of x values
    n_samples: number of samples
    noise: noise standard deviation
    """
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = func(x) + np.random.normal(0, noise, n_samples)
    return x, y

# Define the true function (e.g., sinusoidal function)
def true_func(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)

# Generate training and test data
x_train, y_train = generate_data(true_func, n_samples=1000, noise=0.1)
x_test, y_test = generate_data(true_func, n_samples=200, noise=0.0)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"Training data shape: x={x_train_tensor.shape}, y={y_train_tensor.shape}")

# ==========================================
# 2. Define neural network model
# ==========================================
class FunctionFitter(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) for function fitting
    """
    def __init__(self, input_dim=1, hidden_dims=[64, 128, 64], output_dim=1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # Activation function
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create model
model = FunctionFitter(input_dim=1, hidden_dims=[64, 128, 64], output_dim=1)
print(model)

# ==========================================
# 3. Define loss function and optimizer
# ==========================================
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# ==========================================
# 4. Train the model
# ==========================================
def train_model(model, x_train, y_train, x_test, y_test, 
                epochs=100, batch_size=64, print_every=100):
    """
    Train the neural network model
    """
    train_losses = []
    test_losses = []
    
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle data
        indices = torch.randperm(n_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0.0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
            test_loss = criterion(y_test_pred, y_test).item()
            test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}")
    
    return train_losses, test_losses

# Start training
print("\nStarting training...")
train_losses, test_losses = train_model(
    model, 
    x_train_tensor, y_train_tensor,
    x_test_tensor, y_test_tensor,
    epochs=2000, 
    batch_size=64, 
    print_every=200
)

# ==========================================
# 5. Visualize results
# ==========================================
def plot_results(model, x_test, y_test, train_losses, test_losses):
    """
    Visualize training results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: fitting curve
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(-5, 5, 500).unsqueeze(1)
        y_pred = model(x_plot).numpy().flatten()
        x_plot = x_plot.numpy().flatten()
    
    ax1.scatter(x_test.numpy().flatten(), y_test.numpy().flatten(), 
                s=10, alpha=0.5, label='Test Data', color='blue')
    ax1.plot(x_plot, y_pred, 'r-', linewidth=2, label='Neural Network Fit')
    ax1.plot(x_plot, true_func(x_plot), 'g--', linewidth=2, label='True Function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Function Fitting Result')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: loss curve
    ax2.plot(train_losses, label='Train Loss', alpha=0.7)
    ax2.plot(test_losses, label='Test Loss', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale
    
    plt.tight_layout()
    plt.savefig('function_fitting_result.png', dpi=150)
    plt.show()

# Plot results
plot_results(model, x_test_tensor, y_test_tensor, train_losses, test_losses)
print("\nResults saved as function_fitting_result.png")

# ==========================================
# 6. Test: use trained model for prediction
# ==========================================
def predict(model, x_value):
    """
    Predict using the trained model
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor([[x_value]], dtype=torch.float32)
        y_pred = model(x_tensor).item()
    return y_pred

# Test a few points
print("\nPrediction test:")
for x_val in [0.0, 1.0, 2.0, -1.0]:
    y_true = true_func(x_val)
    y_pred = predict(model, x_val)
    print(f"x={x_val: .2f}, True={y_true:.4f}, Predicted={y_pred:.4f}, Error={abs(y_true-y_pred):.4f}")
