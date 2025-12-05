import numpy as np

# ============================================================================
# Decision Tree Regressor
# ============================================================================
# This class implements a regression decision tree from scratch using NumPy.
# The tree predicts continuous values by recursively splitting the data based
# on feature thresholds that minimize Mean Squared Error (MSE).

class decisionTree:
    """
    A regression decision tree that predicts continuous target values.
    
    Hyperparameters:
    - min_samples_leaf: Minimum number of samples required in a leaf node
    - min_samples_split: Minimum number of samples required to split a node
    - max_depth: Maximum depth of the tree (prevents overfitting)
    """
    def __init__(self, min_samples_leaf=100, min_samples_split=20, max_depth=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tree = None  # Will store the tree structure after fitting
    
    def fit(self, X, y):
        """
        Build the decision tree from training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # Start building the tree from the root (depth 0)
        self.tree = self.buildTree(X, y, 0)

    def predict(self, X):
        """
        Predict target values for input samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Array of predicted values (n_samples,)
        """
        X = np.asarray(X)
        # Predict each row individually by traversing the tree
        return np.array([self.predict_row(row, self.tree) for row in X])

    def mse(self, y):
        """
        Calculate Mean Squared Error for a set of target values.
        MSE measures the average squared difference from the mean.
        
        Args:
            y: Target values
        
        Returns:
            Mean squared error (scalar)
        """
        return ((y-y.mean())**2).mean()

    def buildTree(self, X, y, depth):
        """
        Recursively build the decision tree by finding the best splits.
        
        This is the core algorithm:
        1. Create a node with the mean of y as its value
        2. Check stopping criteria (max depth, min samples, pure node)
        3. Try all possible splits on all features
        4. Choose the split that minimizes weighted MSE
        5. Recursively build left and right subtrees
        
        Args:
            X: Feature matrix for this node
            y: Target values for this node
            depth: Current depth in the tree
        
        Returns:
            Dictionary representing the node with keys:
            - 'value': mean of y (prediction if leaf)
            - 'is_leaf': whether this is a leaf node
            - 'feature_index': feature to split on (if not leaf)
            - 'threshold': threshold value for split (if not leaf)
            - 'left': left child node (if not leaf)
            - 'right': right child node (if not leaf)
        """
        n_samples, n_features = X.shape
        node = {}
        # The node's value is the mean of all target values in this node
        # This is what we predict if this becomes a leaf
        node["value"] = float(y.mean())

        # Initialize variables to track the best split found
        best_loss = np.inf
        best_thresh = None
        best_feat = None

        # Stopping criteria: make this a leaf node if any condition is met
        if (depth >= self.max_depth or                    # Tree is too deep
            n_samples < self.min_samples_split or         # Not enough samples to split
            np.unique(y).size == 1):                      # All targets are the same (pure node)
            node['is_leaf'] = True
            return node

        # Try splitting on each feature
        for feat_idx in range(n_features):
            values = X[:, feat_idx]
            unique_vals = np.unique(values)
            
            # Skip if all values are the same (can't split)
            if unique_vals.size == 1:
                continue
            
            # Create candidate thresholds as midpoints between consecutive unique values
            trashholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            
            # Try each threshold
            for t in trashholds:
                # Split data into left (<=) and right (>) based on threshold
                left_mask = values <= t
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                # Skip if either child would be too small
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Get target values for each child
                y_left = y[left_mask]
                y_right = y[right_mask]

                # Calculate MSE for each child
                mse_left = self.mse(y_left)
                mse_right = self.mse(y_right)

                # Calculate weighted average loss for this split
                # This is the total MSE after splitting
                loss = (n_left * mse_left + n_right * mse_right) / (n_samples)

                # Keep track of the best split (lowest loss)
                if loss < best_loss:
                    best_loss = loss
                    best_thresh = t
                    best_feat = feat_idx
        
        # If no valid split was found, make this a leaf
        if best_feat is None:
            node['is_leaf'] = True
            return node

        # We found a good split, so this is not a leaf
        node['is_leaf'] = False
        node['feature_index'] = best_feat
        node['threshold'] = best_thresh

        # Split the data using the best feature and threshold
        values = X[:, best_feat]
        left_mask = values <= best_thresh
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        node['left'] = self.buildTree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self.buildTree(X[right_mask], y[right_mask], depth + 1)

        return node

    def predict_row(self, row, node):
        """
        Predict the target value for a single sample by traversing the tree.
        
        Start at the root and follow the tree down:
        - If current node is a leaf, return its value
        - Otherwise, check the split condition and go left or right
        
        Args:
            row: Single feature vector (n_features,)
            node: Current node in the tree
        
        Returns:
            Predicted value (scalar)
        """
        # Traverse the tree until we reach a leaf
        while not node['is_leaf']:
            feat_idx = node['feature_index']
            thresh = node['threshold']
            # Go left if feature value <= threshold, otherwise go right
            if row[feat_idx] <= thresh:
                node = node['left']
            else:
                node = node['right']
        # Return the mean value stored in the leaf
        return node['value']


# ============================================================================
# Utility Functions
# ============================================================================

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error between true and predicted values.
    
    RMSE is a common metric for regression tasks. It measures the average
    magnitude of prediction errors in the same units as the target variable.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        Root mean squared error (scalar)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(((y_true - y_pred) ** 2).mean())




# ============================================================================
# Feed-Forward Neural Network Regressor
# ============================================================================
# This class implements a simple 2-layer neural network from scratch using NumPy.
# Architecture: Input -> Hidden Layer (ReLU) -> Output Layer (Linear)
# The network is trained using gradient descent with backpropagation.

class FFNNRegressor:
    """
    A feed-forward neural network for regression tasks.
    
    Architecture:
    - Input layer: input_dim neurons
    - Hidden layer: hidden_dim neurons with ReLU activation
    - Output layer: 1 neuron (linear activation for regression)
    
    Hyperparameters:
    - input_dim: Number of input features
    - hidden_dim: Number of neurons in the hidden layer
    - lr: Learning rate for gradient descent
    - epochs: Number of training epochs
    - batch_size: Number of samples per mini-batch
    - l2: L2 regularization coefficient (weight decay)
    - random_state: Random seed for reproducibility
    - verbose: Whether to print training progress
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        lr=0.01,
        epochs=80,
        batch_size=2048,
        l2=0.0,
        random_state=0,
        verbose=True,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.verbose = verbose

        # Initialize random number generator for reproducibility
        rng = np.random.RandomState(random_state)

        # Initialize weights using Xavier/He initialization
        # W1: weights from input to hidden layer (input_dim x hidden_dim)
        # Divided by sqrt(input_dim) to prevent exploding/vanishing gradients
        self.W1 = rng.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))  # Biases for hidden layer

        # W2: weights from hidden to output layer (hidden_dim x 1)
        self.W2 = rng.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, 1))  # Bias for output layer

    @staticmethod
    def relu(z):
        """
        ReLU (Rectified Linear Unit) activation function.
        
        ReLU(z) = max(0, z)
        This introduces non-linearity and helps the network learn complex patterns.
        
        Args:
            z: Pre-activation values
        
        Returns:
            Activated values (same shape as z)
        """
        return np.maximum(0.0, z)

    @staticmethod
    def relu_deriv(z):
        """
        Derivative of ReLU activation function.
        
        d/dz ReLU(z) = 1 if z > 0, else 0
        Used during backpropagation to compute gradients.
        
        Args:
            z: Pre-activation values
        
        Returns:
            Gradient (same shape as z)
        """
        return (z > 0.0).astype(np.float64)

    @staticmethod
    def mse(y_true, y_pred):
        """
        Calculate Mean Squared Error loss.
        
        MSE = mean((y_true - y_pred)^2)
        This is the loss function we're trying to minimize.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        
        Returns:
            Mean squared error (scalar)
        """
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true, y_pred):
        """
        Calculate Root Mean Squared Error.
        
        RMSE = sqrt(MSE)
        More interpretable than MSE as it's in the same units as the target.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        
        Returns:
            Root mean squared error (scalar)
        """
        return np.sqrt(FFNNRegressor.mse(y_true, y_pred))

    @staticmethod
    def r2_score(y_true, y_pred):
        """
        Calculate R² (coefficient of determination) score.
        
        R² = 1 - (SS_res / SS_tot)
        where SS_res = sum of squared residuals
              SS_tot = total sum of squares
        
        R² ranges from -∞ to 1, where 1 is perfect prediction.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        
        Returns:
            R² score (scalar)
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)  # Total sum of squares
        return 1.0 - ss_res / ss_tot

    def _forward(self, X):
        """
        Forward pass through the network.
        
        Computation flow:
        1. z1 = X @ W1 + b1           (linear combination for hidden layer)
        2. a1 = ReLU(z1)               (activation of hidden layer)
        3. z2 = a1 @ W2 + b2           (linear combination for output layer)
        4. y_pred = z2                 (no activation for regression output)
        
        Args:
            X: Input features (batch_size, input_dim)
        
        Returns:
            Tuple of (z1, a1, z2, y_pred) - intermediate values needed for backprop
        """
        z1 = X @ self.W1 + self.b1      # Pre-activation of hidden layer
        a1 = self.relu(z1)               # Activation of hidden layer
        z2 = a1 @ self.W2 + self.b2      # Pre-activation of output layer
        y_pred = z2                      # Output (no activation for regression)
        return z1, a1, z2, y_pred

    def fit(self, X, y):
        """
        Train the neural network using mini-batch gradient descent.
        
        Training process:
        1. For each epoch, shuffle the data
        2. Split data into mini-batches
        3. For each batch:
           a. Forward pass: compute predictions
           b. Compute loss and gradients (backpropagation)
           c. Update weights using gradient descent
        4. Optionally print progress
        
        Args:
            X: Training features (n_samples, input_dim)
            y: Training targets (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        N, D = X.shape

        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data at the start of each epoch for better generalization
            idx = np.random.permutation(N)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            # Process data in mini-batches
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                if X_batch.shape[0] == 0:
                    continue

                # ===== FORWARD PASS =====
                z1, a1, z2, y_pred = self._forward(X_batch)

                # ===== BACKPROPAGATION =====
                # Compute gradients using the chain rule
                
                batch_size = X_batch.shape[0]
                # Gradient of MSE loss with respect to predictions
                # d/dy_pred MSE = 2 * (y_pred - y_true) / batch_size
                dL_dy = 2.0 * (y_pred - y_batch) / batch_size

                # Gradients for output layer (W2, b2)
                # dL/dW2 = a1^T @ dL/dy
                dL_dW2 = a1.T @ dL_dy
                dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)
                # Add L2 regularization gradient if enabled
                if self.l2 > 0.0:
                    dL_dW2 += 2.0 * self.l2 * self.W2

                # Backpropagate through hidden layer
                # dL/da1 = dL/dy @ W2^T
                dL_da1 = dL_dy @ self.W2.T
                # Apply ReLU derivative: dL/dz1 = dL/da1 * ReLU'(z1)
                dL_dz1 = dL_da1 * self.relu_deriv(z1)

                # Gradients for hidden layer (W1, b1)
                # dL/dW1 = X^T @ dL/dz1
                dL_dW1 = X_batch.T @ dL_dz1
                dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
                # Add L2 regularization gradient if enabled
                if self.l2 > 0.0:
                    dL_dW1 += 2.0 * self.l2 * self.W1

                # ===== GRADIENT DESCENT UPDATE =====
                # Update weights and biases by moving in the opposite direction of gradients
                self.W2 -= self.lr * dL_dW2
                self.b2 -= self.lr * dL_db2
                self.W1 -= self.lr * dL_dW1
                self.b1 -= self.lr * dL_db1

            # Print training progress every 10 epochs
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                _, _, _, y_pred_full = self._forward(X)
                loss = self.mse(y, y_pred_full)
                rmse_val = self.rmse(y, y_pred_full)
                print(f"Epoch {epoch+1:3d}/{self.epochs} - MSE: {loss:.6f} - RMSE: {rmse_val:.4f}")

    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features (n_samples, input_dim)
        
        Returns:
            Predicted values (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        # Run forward pass and extract predictions
        _, _, _, y_pred = self._forward(X)
        return y_pred.ravel()  # Flatten to 1D array

