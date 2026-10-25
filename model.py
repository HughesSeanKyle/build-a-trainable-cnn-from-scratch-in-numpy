"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    """Return indices of the maximum values along each row."""
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    """Return the maximum value of each row, reshaping to a column vector (N, 1)."""
    return np.max(matrix, axis=1, keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return the sum of each row, reshaping to a column vector (N, 1)."""
    return np.sum(matrix, axis=1, keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Compute the element-wise exponential after shifting rows by their maximum."""
    return np.exp(logits - row_max(logits))

# Step 5 - stable_softmax
def stable_softmax(logits):
    """Compute numerically stable softmax probabilities for each row."""
    exps = exp_shifted(logits)
    return exps / row_sum(exps)

# Step 6 - one_hot
def one_hot(labels, num_classes):
    """Convert an array of integer labels (N,) into a one-hot encoded matrix (N, num_classes)."""
    return np.eye(num_classes)[labels]

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    """Extract the probability assigned to the true class label for each sample."""
    # Uses advanced indexing to grab probs[0, labels[0]], probs[1, labels[1]], etc.
    return probs[np.arange(len(labels)), labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    """Compute the mean cross-entropy loss over a batch, clipped for safety."""
    true_probs = gather_true_class_probs(probs, labels)
    # Clip probabilities to avoid taking log(0)
    true_probs = np.clip(true_probs, eps, 1.0)
    return -np.mean(np.log(true_probs))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    """Compute the classification accuracy fraction."""
    predictions = argmax_rows(logits_or_probs)
    return np.mean(predictions == labels)

# Step 10 - he_std
def he_std(fan_in):
    """Calculate the standard deviation for He initialization."""
    return np.sqrt(2.0 / fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed=0):
    """
    Initialize weights using He normal distribution given a shape tuple and seed.
    Returns a float64 NumPy array.
    """
    # Explicitly seed NumPy's global RNG before sampling
    np.random.seed(seed)
    
    # Draw from a standard normal distribution and scale by He standard deviation
    return np.random.randn(*shape).astype(np.float64) * he_std(fan_in)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    """Initialize a zero-filled vector for layer biases."""
    return np.zeros((length,))

# Step 13 - pad_2d
def pad_2d(images, pad):
    """
    Apply zero-padding to the height and width dimensions of a 4D tensor X.
    X shape: (N, C, H, W)
    """
    if pad == 0:
        return images
    # pad width tuple format: ((before_axis0, after_axis0), (before_axis1, after_axis1)...)
    return np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    """Calculate the output spatial size (Height or Width) for a 2D convolution."""
    return int(np.floor((input_size + 2 * padding - kernel) / stride)) + 1

# Step 15 - im2col
# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    """
    Transform a 4D image tensor into a 2D matrix where each ROW represents 
    a sliding window patch, matching the framework's layout.
    
    Expected output shape: (N * H_out * W_out, C * kernel_h * kernel_w)
    """
    N, C, H, W = images.shape
    H_out = output_spatial_size(H, kernel_h, stride, padding)
    W_out = output_spatial_size(W, kernel_w, stride, padding)
    
    X_pad = pad_2d(images, padding)
    
    # Initialize the output matrix: each row is one patch
    cols = np.zeros((N * H_out * W_out, C * kernel_h * kernel_w), dtype=images.dtype)
    
    row_idx = 0
    # Loop order to match row-major sample stacking requirements
    for n in range(N):
        for i in range(H_out):
            h_start = i * stride
            h_end = h_start + kernel_h
            for j in range(W_out):
                w_start = j * stride
                w_end = w_start + kernel_w
                
                # Extract the patch volume: shape (C, kernel_h, kernel_w)
                patch = X_pad[n, :, h_start:h_end, w_start:w_end]
                
                # Flatten the patch into a 1D row array and assign it
                cols[row_idx, :] = patch.ravel()
                row_idx += 1
                
    return cols

# Step 16 - col2im
# Step 16 - col2im
def col2im(cols, X_shape, kernel_h, kernel_w, stride, padding):
    """
    The exact structural inverse to aggregate row-major column matrices 
    back into a 4D image tensor layout during backprop.
    """
    N, C, H, W = X_shape
    H_out = output_spatial_size(H, kernel_h, stride, padding)
    W_out = output_spatial_size(W, kernel_w, stride, padding)
    
    X_pad = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)
    
    row_idx = 0
    for n in range(N):
        for i in range(H_out):
            h_start = i * stride
            h_end = h_start + kernel_h
            for j in range(W_out):
                w_start = j * stride
                w_end = w_start + kernel_w
                
                # Reshape the flat row slice back into a 3D patch and accumulate
                X_pad[n, :, h_start:h_end, w_start:w_end] += cols[row_idx, :].reshape(C, kernel_h, kernel_w)
                row_idx += 1
                
    if padding > 0:
        return X_pad[:, :, padding:-padding, padding:-padding]
    return X_pad

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    """
    Compute forward pass for 2D convolution using row-major im2col layout.
    Stores all metadata safely inside a dictionary cache.
    """
    N, C_in, H, W = x.shape
    C_out, _, kh, kw = weights.shape
    H_out = output_spatial_size(H, kh, stride, padding)
    W_out = output_spatial_size(W, kw, stride, padding)
    
    # x_col shape: (N * H_out * W_out, C_in * kh * kw)
    x_col = im2col(x, kh, kw, stride, padding)
    
    # weights_row shape: (C_in * kh * kw, C_out)
    weights_row = weights.reshape(C_out, -1).T
    
    # Matrix multiply and add bias across columns
    out = x_col @ weights_row + bias.reshape(1, -1)
    
    # Reshape back to 4D layout: (N, C_out, H_out, W_out)
    out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)
    
    # Pack multiple variations of the tracking key for total cross-compatibility
    cache = {
        'x': x,
        'weights': weights,
        'bias': bias,
        'stride': stride,
        'padding': padding,
        'kernel_h': kh,
        'kernel_w': kw,
        'cols': x_col,
        'x_col': x_col
    }
    return out, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    """Compute gradient with respect to input x from the cache safely."""
    x = cache['x']
    weights = cache['weights']
    stride = cache['stride']
    padding = cache['padding']
    kh = cache['kernel_h']
    kw = cache['kernel_w']
    
    C_out, C_in, _, _ = weights.shape
    weights_row = weights.reshape(C_out, -1).T
    
    dout_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)
    dx_col = dout_reshaped @ weights_row.T
    
    return col2im(dx_col, x.shape, kh, kw, stride, padding)

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    """Compute gradient with respect to filters weights from the cache safely."""
    weights = cache['weights']
    
    # Fallback structure using .get() guarantees key-safety 
    cols = cache.get('cols') if cache.get('cols') is not None else cache.get('x_col')
    
    C_out, C_in, kh, kw = weights.shape
    dout_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)
    
    dweights_row = cols.T @ dout_reshaped
    return dweights_row.T.reshape(weights.shape)

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    """Compute gradient with respect to convolution bias vector."""
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    """Execute complete backpropagation sequence seamlessly."""
    dx = conv2d_grad_input(d_out, cache)
    dweights = conv2d_grad_weights(d_out, cache)
    dbias = conv2d_grad_bias(d_out)
    return dx, dweights, dbias

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel=2, stride=2):
    """
    Forward pass for 2D Max Pooling layer tracking flat argmax indices.
    x: (N, C, H, W)
    Returns (out, cache) matching the strict key validation requirements.
    """
    N, C, H, W = x.shape
    # Step 14 output_spatial_size is available globally
    H_out = output_spatial_size(H, kernel, stride, padding=0)
    W_out = output_spatial_size(W, kernel, stride, padding=0)
    
    # Initialize output activation and flat argmax tracking maps
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    argmax = np.zeros((N, C, H_out, W_out), dtype=np.int64)
    
    for i in range(H_out):
        h_start = i * stride
        h_end = h_start + kernel
        for j in range(W_out):
            w_start = j * stride
            w_end = w_start + kernel
            
            # Extract spatial window across all batch elements and channels
            patch = x[:, :, h_start:h_end, w_start:w_end]
            
            # Reshape the spatial patch to flat rows to pull max and flat index values
            # patch shape becomes: (N, C, kernel * kernel)
            patch_flat = patch.reshape(N, C, -1)
            
            out[:, :, i, j] = np.max(patch_flat, axis=2)
            argmax[:, :, i, j] = np.argmax(patch_flat, axis=2)
            
    cache = {
        'x_shape': x.shape,
        'argmax': argmax,
        'kernel': kernel,
        'stride': stride
    }
    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    """
    Construct a standalone 2D array of shape (kernel, kernel) filled with zeros,
    and place the single scalar upstream gradient value at the specified flat 
    argmax index location.
    
    Returns:
        An ndarray of shape (kernel, kernel) and float64 data type.
    """
    # Create the standalone window filled with zeros
    window_grad = np.zeros((kernel, kernel), dtype=np.float64)
    
    # Convert the flat row-major index into relative 2D coordinate offsets
    h_idx = argmax_index // kernel
    w_idx = argmax_index % kernel
    
    # Place the incoming gradient value at the wining coordinate location
    window_grad[h_idx, w_idx] = grad_value
    
    return window_grad

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    """
    Backward pass for 2D Max Pooling layer utilizing the optimized scatter_grad_window helper.
    """
    N, C, H, W = cache['x_shape']
    argmax = cache['argmax']
    kernel = cache['kernel']
    stride = cache['stride']
    
    _, _, H_out, W_out = d_out.shape
    dx = np.zeros((N, C, H, W), dtype=np.float64)
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h_start = i * stride
                h_end = h_start + kernel
                for j in range(W_out):
                    w_start = j * stride
                    w_end = w_start + kernel
                    
                    # Pull values to hand off to our standalone scatter primitive
                    dout_val = d_out[n, c, i, j]
                    argmax_index = argmax[n, c, i, j]
                    
                    # Generate the localized window gradient map
                    window_grad = scatter_grad_window(dout_val, argmax_index, kernel)
                    
                    # Accumulate the window directly into the target spatial slice of dx
                    dx[n, c, h_start:h_end, w_start:w_end] += window_grad
                    
    return dx

# Step 25 - relu_forward
def relu_forward(x):
    """Forward pass for element-wise rectified linear unit (ReLU)."""
    out = np.maximum(0, x)
    cache = {'x': x}
    return out, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    """Backward pass for element-wise rectified linear unit (ReLU)."""
    x = cache['x']
    dx = np.copy(d_out)
    dx[x <= 0] = 0
    return dx

# Step 27 - flatten_forward
def flatten_forward(x):
    """
    Flatten a multi-dimensional array into a 2D batch row matrix,
    preserving the batch axis (axis 0).
    
    Returns (out, cache) matching the required 'x_shape' dictionary lookup.
    """
    x_shape = x.shape
    # Flatten everything from axis 1 onwards while leaving axis 0 intact
    out = x.reshape(x_shape[0], -1)
    
    cache = {
        'x_shape': x_shape
    }
    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    """
    Restore the 4D or volumetric dimensional layout of an incoming 
    matrix gradient using the recorded 'x_shape'.
    """
    x_shape = cache['x_shape']
    return d_out.reshape(x_shape)

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    """Forward pass for a dense linear layer. Excludes bias from cache keys."""
    out = x @ weights + bias
    cache = {
        'x': x,
        'weights': weights
    }
    return out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """
    Compute linear gradient with respect to inputs x.
    Accepts either a raw weights matrix or a cache dictionary safely.
    """
    if isinstance(cache, dict):
        weights = cache['weights']
    else:
        weights = cache
    return d_out @ weights.T

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """
    Compute linear gradient with respect to weights using the exact formula: x.T @ d_out.
    Robustly handles both standalone call signatures and integrated backward dictionaries.
    """
    if isinstance(dout, dict):
        # Integrated backward mode: linear_grad_weights(d_out, cache)
        d_out = x
        x = dout['x']
    elif isinstance(x, dict):
        # Integrated backward fallback: linear_grad_weights(cache, d_out)
        d_out = dout
        x = x['x']
    else:
        # Standalone isolated mode: linear_grad_weights(x, d_out)
        x = x
        d_out = dout
        
    return x.T @ d_out

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    """Compute linear gradient with respect to bias vector."""
    return np.sum(dout, axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    """
    Execute full backpropagation sequence for a fully-connected layer.
    Passes the unified cache directly to primitives to avoid layout bugs.
    """
    dx = linear_grad_input(dout, cache)
    dW = linear_grad_weights(dout, cache)
    db = linear_grad_bias(dout)
    return dx, dW, db

# Step 34 - softmax_cross_entropy_forward
# A simple global dictionary to act as our layer cache mechanism
_fused_loss_cache = {}

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    """
    Compute fused stable softmax probabilities and mean cross-entropy loss.
    
    Args:
        logits: (N, C) array of unnormalized log probabilities.
        y: (N,) array of integer target class indexes.
        
    Returns:
        loss: Python float indicating the mean cross-entropy error value.
    """
    # Reuse upstream stable_softmax (Step 5)
    probs = stable_softmax(logits)
    
    # Reuse upstream cross_entropy_loss (Step 8)
    loss_scalar = cross_entropy_loss(probs, y)
    
    # Cache parameters globally for step 35 to consume
    global _fused_loss_cache
    _fused_loss_cache['probs'] = probs
    _fused_loss_cache['labels'] = y
    
    # Coerce negative zero artifacts to positive zero
    if abs(loss_scalar) < 1e-15:
        loss_scalar = 0.0
        
    # Force the return value to be a standard Python float
    return float(loss_scalar)

# Step 35 - softmax_cross_entropy_backward
# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    """
    Compute the gradient of the fused loss with respect to the input logits.
    
    Args:
        logits: (N, C) array of unnormalized log probabilities.
        y: (N,) array of integer target class indexes.
        
    Returns:
        dlogits: (N, C) gradient matrix.
    """
    N = len(y)
    num_classes = logits.shape[1]
    
    # Reuse upstream stable_softmax (Step 5)
    probs = stable_softmax(logits)
    
    # Reuse upstream one_hot (Step 6)
    y_true = one_hot(y, num_classes)
    
    # Mathematically simplified gradient: (Probs - TrueOneHot) averaged over batch N
    dlogits = (probs - y_true) / N
    
    # Clean up floating-point negative zero artifacts
    dlogits[np.abs(dlogits) < 1e-15] = 0.0
    
    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    """
    Perform a single parameter step update using Stochastic Gradient Descent.
    
    Args:
        param: Current parameter tensor.
        grad: Gradient tensor (same shape as param).
        lr: Scalar learning rate.
        
    Returns:
        updated_param: Updated parameter array.
    """
    return param - lr * grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    """
    Perform one update of Adam's first moment estimate.
    
    Args:
        m: Previous moment array.
        grad: Current gradient array.
        beta_one: First moment decay rate.
        
    Returns:
        new_m: Updated first moment array.
    """
    return beta_one * m + (1.0 - beta_one) * grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    """
    Advance Adam's second moment estimate one step.
    
    Args:
        v: Running second moment estimate array.
        grad: Current gradient array.
        beta_two: Second moment decay rate.
        
    Returns:
        new_v: Updated second moment estimate array of squared gradients.
    """
    return beta_two * v + (1.0 - beta_two) * (grad ** 2)

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    """
    Return the bias-corrected moment estimate for Adam.
    
    Args:
        moment: Accumulated moment array (m or v).
        beta: Corresponding decay rate (beta_one or beta_two).
        t: Current global training step counter index (starting from 1).
        
    Returns:
        corrected_moment: Rescaled moment estimate array.
    """
    return moment / (1.0 - (beta ** t))

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    """
    Apply one Adam parameter update using bias-corrected moments without mutating in-place.
    
    Args:
        param: Current parameter array.
        m_hat: Bias-corrected first moment estimate array.
        v_hat: Bias-corrected second moment estimate array.
        lr: Base scalar learning rate.
        eps: Small smoothing stability denominator offset factor.
        
    Returns:
        new_param: Updated parameter array.
    """
    return param - (lr / (np.sqrt(v_hat) + eps)) * m_hat

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one=0.9, beta_two=0.999, eps=1e-8):
    """
    Perform one full Adam optimizer update by chaining the four upstream Adam helpers.
    
    Args:
        param: Current parameter array.
        grad: Current gradient array.
        m: Running first moment estimate.
        v: Running second moment estimate.
        t: Step counter metric index tracker.
        lr: Active scalar learning rate.
        beta_one: First moment decay rate hyperparameter.
        beta_two: Second moment decay rate hyperparameter.
        eps: Small denominator safety offset variable.
        
    Returns:
        new_param: The adjusted parameter matrix.
        new_m: Updated uncorrected first moment state array.
        new_v: Updated uncorrected second moment state array.
    """
    # Chain the upstream helpers exactly as specified
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)
    
    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)
    
    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)
    
    return new_param, new_m, new_v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    """
    Construct the parameters for a single 2D convolution layer.
    
    The He fan-in accounts for every input element a single output unit sees:
    fan_in = in_channels * kernel_size * kernel_size
    """
    # Calculate fan_in volume
    fan_in = in_channels * kernel_size * kernel_size
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    
    # Reuse upstream primitives
    W = he_init(shape, fan_in, seed)
    b = init_zero_bias(out_channels)
    
    return {'W': W, 'b': b}

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    """
    Build the parameter dictionary for a single fully-connected layer.
    
    For a dense layer, fan_in equals the number of incoming features.
    """
    shape = (in_features, out_features)
    
    # Reuse upstream primitives
    W = he_init(shape, in_features, seed)
    b = init_zero_bias(out_features)
    
    return {'W': W, 'b': b}

# Step 44 - init_lenet
def init_lenet(in_channels=1, num_classes=10, seed=0):
    """
    Build the parameter dictionary for a LeNet-style network.
    
    Args:
        in_channels: Integer indicating the depth of the input volume (e.g., 1 for grayscale, 3 for RGB).
        num_classes: Integer indicating the number of output target classification nodes.
        seed: Base random seed integer forwarded sequentially across layer primitives.
        
    Returns:
        params: Dictionary with keys 'conv1', 'conv2', 'fc1', 'fc2' holding sub-dicts.
    """
    # Layer 1 Conv: adapts dynamically to the specified input volume channels
    conv1 = init_conv_layer(out_channels=6, in_channels=in_channels, kernel_size=5, seed=seed + 1)
    
    # Layer 2 Conv: 6 input channels, 16 filters, 5x5 kernel
    conv2 = init_conv_layer(out_channels=16, in_channels=6, kernel_size=5, seed=seed + 2)
    
    # Layer 3 Fully-Connected: Sized for 28x28 spatial inputs
    fc1 = init_linear_layer(in_features=256, out_features=120, seed=seed + 3)
    
    # Layer 4 Fully-Connected Output Head: adapts dynamically to the target number of classes
    fc2 = init_linear_layer(in_features=120, out_features=num_classes, seed=seed + 4)
    
    return {
        'conv1': conv1,
        'conv2': conv2,
        'fc1': fc1,
        'fc2': fc2
    }

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    """
    Chain a convolution, a ReLU activation, and a 2D max-pool in sequence.
    The pool's stride equals its window size (non-overlapping pooling).
    """
    # 1. Convolution
    conv_out, conv_cache = conv2d_forward(x, W, b, stride, pad)
    
    # 2. Activation
    relu_out, relu_cache = relu_forward(conv_out)
    
    # 3. Max Pooling (pool stride = pool window size)
    pool_out, pool_cache = maxpool2d_forward(relu_out, kernel=pool_size, stride=pool_size)
    
    cache = {
        'conv_cache': conv_cache,
        'relu_cache': relu_cache,
        'pool_cache': pool_cache
    }
    return pool_out, cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    """
    Process the dense head of the LeNet classifier:
    Flatten -> FC1 -> ReLU -> FC2
    """
    # 1. Flatten spatial volumes into dense vectors
    flat_out, flatten_cache = flatten_forward(x)
    
    # 2. First Linear Layer
    fc1_out, fc1_cache = linear_forward(flat_out, fc1['W'], fc1['b'])
    
    # 3. Activation
    relu_out, relu_cache = relu_forward(fc1_out)
    
    # 4. Output Linear Layer (Logits)
    logits, fc2_cache = linear_forward(relu_out, fc2['W'], fc2['b'])
    
    cache = {
        'flatten_cache': flatten_cache,
        'fc1_cache': fc1_cache,
        'relu_cache': relu_cache,
        'fc2_cache': fc2_cache
    }
    return logits, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    """
    Execute the entire LeNet forward pass sequence.
    Each conv block uses stride=1, padding=0, followed by a 2x2 max pool.
    """
    # Block 1
    out_b1, cache_b1 = forward_conv_block(x, params['conv1']['W'], params['conv1']['b'], pool_size=2, stride=1, pad=0)
    
    # Block 2
    out_b2, cache_b2 = forward_conv_block(out_b1, params['conv2']['W'], params['conv2']['b'], pool_size=2, stride=1, pad=0)
    
    # Dense Classifier Head
    logits, cache_cls = forward_classifier_block(out_b2, params['fc1'], params['fc2'])
    
    caches = {
        'block1': cache_b1,
        'block2': cache_b2,
        'classifier': cache_cls
    }
    return logits, caches

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    """
    Unwind the convolutional block in exact reverse order:
    Pool Backprop -> ReLU Backprop -> Conv Backprop
    """
    # 1. Backprop through Max Pool
    drelu_out = maxpool2d_backward(dout, cache['pool_cache'])
    
    # 2. Backprop through ReLU
    dconv_out = relu_backward(drelu_out, cache['relu_cache'])
    
    # 3. Backprop through Conv Layer
    dx, dW, db = conv2d_backward(dconv_out, cache['conv_cache'])
    
    return dx, dW, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    """
    Walk the dense classifier head backward in reverse execution order:
    FC2 Backprop -> ReLU Backprop -> FC1 Backprop -> Flatten Backprop
    """
    # 1. Backprop output linear layer (FC2)
    drelu_out, dW_fc2, db_fc2 = linear_backward(dlogits, cache['fc2_cache'])
    
    # 2. Backprop through ReLU
    dfc1_out = relu_backward(drelu_out, cache['relu_cache'])
    
    # 3. Backprop intermediate linear layer (FC1)
    dflat_out, dW_fc1, db_fc1 = linear_backward(dfc1_out, cache['fc1_cache'])
    
    # 4. Backprop through Flatten to restore target dimensions
    dx = flatten_backward(dflat_out, cache['flatten_cache'])
    
    return {
        'dx': dx,
        'fc1': {'dW': dW_fc1, 'db': db_fc1},
        'fc2': {'dW': dW_fc2, 'db': db_fc2}
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    """
    Stitch every parameter gradient of the model into a single corporate dict.
    Runs the backpropagation sequence in structural reverse.
    """
    # 1. Backward through Classifier Head
    dgrads_cls = backward_classifier_block(dlogits, caches['classifier'])
    
    # 2. Backward through Conv Block 2
    dx_b2, dW_conv2, db_conv2 = backward_conv_block(dgrads_cls['dx'], caches['block2'])
    
    # 3. Backward through Conv Block 1
    _, dW_conv1, db_conv1 = backward_conv_block(dx_b2, caches['block1'])
    
    return {
        'conv1': {'dW': dW_conv1, 'db': db_conv1},
        'conv2': {'dW': dW_conv2, 'db': db_conv2},
        'fc1': dgrads_cls['fc1'],
        'fc2': dgrads_cls['fc2']
    }

# Step 51 - lenet_predict
def lenet_predict(x, params):
    """
    Return the highest-probability class index for each sample in a batch.
    """
    # Run the full forward pipeline
    logits, _ = lenet_forward(x, params)
    
    # Reduce unnormalized logits to row-major integer label indexes
    return argmax_rows(logits).astype(np.int64)

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    """
    Produce a tiny labeled image set for sanity-checking the CNN.
    """
    rng = np.random.default_rng(seed)
    
    # Draw the integer labels uniformly
    y = rng.integers(0, num_classes, size=(num_samples,))
    
    # Draw pixel values using standard normal distribution
    x = rng.standard_normal((num_samples, in_channels, image_size, image_size))
    
    # Shift arrays to make classes linearly separable
    for k in range(num_classes):
        mask = (y == k)
        if np.any(mask):
            shift_scalar = k - (num_classes - 1) / 2.0
            x[mask] += shift_scalar
            
    return x, y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    """
    Return a reproducible random permutation sequence of the integers in [0, n).
    
    Args:
        n: The upper boundary count for generating indices.
        seed: The integer random seed for reproducibility.
        
    Returns:
        indices: Integer ndarray of shape (n,) containing numbers from 0 to n-1.
    """
    # Seed the legacy global random state to guarantee exact permutation matching
    np.random.seed(seed)
    
    # Generate the sequential array and shuffle its index coordinates in place
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    return indices

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    """
    Partition features and labels into training and test sets using a shared shuffled order.
    """
    N = len(y)
    shuffled_idx = shuffle_indices(N, seed=seed)
    
    split_boundary = int(N * test_fraction)
    test_idx = shuffled_idx[:split_boundary]
    train_idx = shuffled_idx[split_boundary:]
    
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    """
    A generator yielding shuffled (xb, yb) pairs covering one full pass over data.
    Drops any trailing partial batches to guarantee exact batch sizing.
    """
    N = len(y)
    shuffled_idx = shuffle_indices(N, seed=seed)
    
    for start_idx in range(0, N - batch_size + 1, batch_size):
        end_idx = start_idx + batch_size
        batch_slice = shuffled_idx[start_idx:end_idx]
        
        yield x[batch_slice], y[batch_slice]

# Step 56 - train_step
def train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step):
    """
    Run one complete forward and backward training iteration of the network.
    """
    # 1. Forward Pass through LeNet
    logits, caches = lenet_forward(xb, params)
    
    # 2. Compute Fused Softmax Cross Entropy Loss
    loss = softmax_cross_entropy_forward(logits, yb)
    
    # 3. Compute Upstream Logits Gradient
    dlogits = softmax_cross_entropy_backward(logits, yb)
    
    # 4. Backward Pass through LeNet
    grads = lenet_backward(dlogits, caches)
    
    # 5. Optimization updates via Adam across structural parameter mirrors
    new_params = {}
    new_opt_state = {}
    
    for layer in params.keys():
        new_params[layer] = {}
        new_opt_state[layer] = {}
        for pname in params[layer].keys():  # 'W' and 'b'
            p = params[layer][pname]
            g = grads[layer][f"d{pname}"]   # maps 'W' to 'dW' and 'b' to 'db'
            m = opt_state[layer][pname]['m']
            v = opt_state[layer][pname]['v']
            
            # Step 41 optimization helper call
            updated_p, updated_m, updated_v = adam_step(
                param=p, grad=g, m=m, v=v, t=step,
                lr=lr, beta_one=beta_one, beta_two=beta_two, eps=eps
            )
            
            new_params[layer][pname] = updated_p
            new_opt_state[layer][pname] = {'m': updated_m, 'v': updated_v}
            
    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    """
    Perform one full pass over the training data using shuffled minibatches.
    """
    loss_list = []
    
    # Step 55 generator call
    batch_generator = iterate_minibatches(x, y, batch_size, seed=seed)
    
    for xb, yb in batch_generator:
        step_counter += 1
        
        # Step 56 update step call
        params, opt_state, loss = train_step(
            params, opt_state, xb, yb, lr, 
            beta_one, beta_two, eps, step=step_counter
        )
        loss_list.append(loss)
        
    return params, opt_state, step_counter, loss_list

# Step 58 - train_loop
def train_loop(params, x_train, y_train, num_epochs, batch_size, lr=1e-3, beta_one=0.9, beta_two=0.999, eps=1e-8, seed=0):
    """
    Drive the full multi-epoch training routine for the LeNet model.
    
    Args:
        params: Initial parameter dictionary tree containing weights and biases.
        x_train: Training feature input image tensor.
        y_train: Training integer class target label vector.
        num_epochs: Total number of complete sweeps through the dataset.
        batch_size: Integer size of each mini-batch chunk.
        lr: Base optimizer scalar learning rate.
        beta_one: Exponential decay tracking rate for Adam first moments.
        beta_two: Exponential decay tracking rate for Adam second moments.
        eps: Denominator stabilization smoothing offset constant.
        seed: Base random seed to compute per-epoch shuffle tracking indexes.
        
    Returns:
        params: The final optimized parameter dictionary.
        all_losses: A flat list of per-step loss values concatenated across all epochs.
    """
    # 1. Build an Adam state that mirrors the parameter tree structurally.
    # Initialize zero first (m) and second (v) moments matching each parameter's shape.
    opt_state = {}
    for layer in params.keys():
        opt_state[layer] = {}
        for pname in params[layer].keys():
            opt_state[layer][pname] = {
                'm': np.zeros_like(params[layer][pname]),
                'v': np.zeros_like(params[layer][pname])
            }
            
    # 2. Maintain a global step counter across all training epochs.
    global_step_counter = 0
    all_losses = []
    
    # 3. Multi-Epoch Training Loop Driver
    for epoch in range(num_epochs):
        # Unique per-epoch shuffle seed matching requirements
        epoch_seed = seed + epoch
        
        # Call train_one_epoch (Step 57)
        params, opt_state, global_step_counter, epoch_losses = train_one_epoch(
            params, opt_state, x_train, y_train, batch_size, lr,
            beta_one, beta_two, eps, global_step_counter, seed=epoch_seed
        )
        
        # Concatenate losses flatly across training iterations
        all_losses.extend(epoch_losses)
        
    return params, all_losses

# Step 59 - evaluate
def evaluate(params, x, y):
    """
    Score the trained model accuracy fraction over a target dataset slice.
    """
    # Step 51 prediction call
    predictions = lenet_predict(x, params)
    
    # Accuracy fraction computation
    accuracy_fraction = np.mean(predictions == y)
    
    return float(accuracy_fraction)

