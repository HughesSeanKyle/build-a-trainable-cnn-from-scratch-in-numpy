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
    Compute linear gradient with respect to weights.
    Handles legacy positional parameters (x, d_out) or dictionary formats safely.
    """
    # If called with a single dictionary cache as the second argument
    if isinstance(dout, dict):
        x = dout['x']
        d_out = x
    else:
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

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

