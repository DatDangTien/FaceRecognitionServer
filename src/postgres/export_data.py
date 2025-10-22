#!/usr/bin/env python3
"""
Export PyTorch embeddings and NumPy usernames to C++-friendly formats.
This script converts .pth and .npy files to binary and text formats
that can be easily read by the C++ migration tool.
"""

import os
import sys
import struct
import torch
import numpy as np

def export_embeddings(input_path, output_path):
    """
    Export PyTorch embeddings to binary format.
    Format: [num_embeddings (int32)][dimension (int32)][embedding_data (float32)]
    """
    print(f"📖 Loading embeddings from: {input_path}")
    
    # Try to load with CPU if CUDA is not available
    try:
        embeddings = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return False
    
    # Convert to numpy for easier handling
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = np.array(embeddings)
    
    # Ensure it's 2D
    if len(embeddings_np.shape) == 1:
        embeddings_np = embeddings_np.reshape(1, -1)
    
    num_embeddings, dimension = embeddings_np.shape
    print(f"📊 Embeddings shape: {embeddings_np.shape}")
    print(f"   - Number of embeddings: {num_embeddings}")
    print(f"   - Dimension: {dimension}")
    
    # Write to binary file
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('i', num_embeddings))
        f.write(struct.pack('i', dimension))
        
        # Write embeddings
        for embedding in embeddings_np:
            # Ensure float32
            embedding_float32 = embedding.astype(np.float32)
            f.write(embedding_float32.tobytes())
    
    print(f"✅ Exported embeddings to: {output_path}")
    return True

def export_usernames(input_path, output_path):
    """
    Export NumPy usernames array to text format (one username per line).
    """
    print(f"📖 Loading usernames from: {input_path}")
    
    try:
        usernames = np.load(input_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Failed to load usernames: {e}")
        return False
    
    # Convert to list if needed
    if isinstance(usernames, np.ndarray):
        usernames = usernames.tolist()
    
    # Ensure all are strings
    usernames = [str(name) for name in usernames]
    
    print(f"👥 Number of usernames: {len(usernames)}")
    
    # Write to text file
    with open(output_path, 'w') as f:
        for username in usernames:
            f.write(username + '\n')
    
    print(f"✅ Exported usernames to: {output_path}")
    return True

def main():
    # Default paths
    data_path = '../../data'
    
    # Input files
    embeddings_input = os.path.join(data_path, 'faceslist.pth')
    embeddings_input_cpu = os.path.join(data_path, 'faceslistCPU.pth')
    usernames_input = os.path.join(data_path, 'usernames.npy')
    
    # Output files
    embeddings_output = os.path.join(data_path, 'embeddings.bin')
    usernames_output = os.path.join(data_path, 'usernames.txt')
    
    print("🚀 Starting data export...")
    print("=" * 60)
    
    # Check which embeddings file exists
    if os.path.exists(embeddings_input):
        input_file = embeddings_input
    elif os.path.exists(embeddings_input_cpu):
        input_file = embeddings_input_cpu
    else:
        print(f"❌ No embeddings file found!")
        print(f"   Looked for:")
        print(f"   - {embeddings_input}")
        print(f"   - {embeddings_input_cpu}")
        return 1
    
    # Export embeddings
    if not export_embeddings(input_file, embeddings_output):
        return 1
    
    print()
    
    # Export usernames
    if not export_usernames(usernames_input, usernames_output):
        return 1
    
    print()
    print("=" * 60)
    print("✅ Export completed successfully!")
    print(f"\n📁 Output files:")
    print(f"   - {embeddings_output}")
    print(f"   - {usernames_output}")
    print(f"\n💡 Next step: Run the C++ migration tool")
    print(f"   ./migrate_data")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

