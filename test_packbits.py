import numpy as np
np.set_printoptions(precision=8)
x = np.array([1.5, -2.25, 3.14159, 1000.5, 0.0001], dtype=np.float32)

x_f16 = x.astype(np.float16)
print("Original float16:", x_f16)

# Encoding logic from utils.py
u8 = x_f16.view(np.uint8).reshape(x_f16.shape + (2,))
bits = np.unpackbits(u8, axis=-1, bitorder='big')
X_bits = bits.reshape(x_f16.shape + (16,)).astype(np.float16)

# Reconstruction logic from evaluate.py
rec_bits = X_bits.astype(np.uint8)
rec_u8 = np.packbits(rec_bits, axis=-1)
X_raw = rec_u8.view(np.float16).reshape(X_bits.shape[0]).astype(np.float32)

print("Reconstructed float32:", X_raw)
print("Difference:", np.abs(x_f16.astype(np.float32) - X_raw))
