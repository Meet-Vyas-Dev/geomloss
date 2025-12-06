"""
Blur Parameter Optimizer - Find Optimal Blur for Your Data

This script helps you find the best blur parameter for your specific
embeddings and distance metric.
"""

import torch
from geomloss import SamplesLoss
import matplotlib.pyplot as plt

def analyze_distances(embeddings_1, embeddings_2, metric="cosine"):
    """
    Analyze the distribution of distances in your data.
    """
    print("=" * 80)
    print(f"DISTANCE ANALYSIS: {metric.upper()}")
    print("=" * 80)
    
    # Sample for efficiency
    # Handle both 2D (N, D) and 3D (B, N, D) tensors
    if embeddings_1.dim() == 3:
        # Flatten batch dimension for sampling
        e1_flat = embeddings_1.reshape(-1, embeddings_1.shape[-1])
        e2_flat = embeddings_2.reshape(-1, embeddings_2.shape[-1])
    else:
        e1_flat = embeddings_1
        e2_flat = embeddings_2
    
    n_samples = min(200, len(e1_flat), len(e2_flat))
    sample_1 = e1_flat[:n_samples]
    sample_2 = e2_flat[:n_samples]
    
    # Compute distances based on metric
    if metric.lower() == "cosine":
        # Normalize for cosine
        s1_norm = torch.nn.functional.normalize(sample_1, dim=-1)
        s2_norm = torch.nn.functional.normalize(sample_2, dim=-1)
        similarity = torch.matmul(s1_norm, s2_norm.T)
        distances = 1 - similarity
        distance_range = "[0, 2]"
    elif metric.lower() == "euclidean":
        distances = torch.cdist(sample_1, sample_2, p=2)
        distance_range = "[0, ∞)"
    elif metric.lower() == "manhattan":
        distances = torch.cdist(sample_1, sample_2, p=1)
        distance_range = "[0, ∞)"
    else:
        # Default to Euclidean
        distances = torch.cdist(sample_1, sample_2, p=2)
        distance_range = "[0, ∞)"
    
    # Flatten for statistics
    distances_flat = distances.flatten()
    
    # Statistics
    min_dist = distances_flat.min().item()
    max_dist = distances_flat.max().item()
    mean_dist = distances_flat.mean().item()
    median_dist = distances_flat.median().item()
    std_dist = distances_flat.std().item()
    q25 = distances_flat.quantile(0.25).item()
    q75 = distances_flat.quantile(0.75).item()
    
    print(f"\nData Statistics:")
    print(f"  Embeddings 1 shape: {embeddings_1.shape}")
    print(f"  Embeddings 2 shape: {embeddings_2.shape}")
    print(f"  Sample size for analysis: {n_samples} × {n_samples}")
    
    print(f"\nDistance Statistics (metric: {metric}):")
    print(f"  Theoretical range: {distance_range}")
    print(f"  Min:               {min_dist:.6f}")
    print(f"  25th percentile:   {q25:.6f}")
    print(f"  Median:            {median_dist:.6f}")
    print(f"  Mean:              {mean_dist:.6f}")
    print(f"  75th percentile:   {q75:.6f}")
    print(f"  Max:               {max_dist:.6f}")
    print(f"  Std deviation:     {std_dist:.6f}")
    
    return {
        'distances': distances_flat,
        'min': min_dist,
        'max': max_dist,
        'mean': mean_dist,
        'median': median_dist,
        'std': std_dist,
        'q25': q25,
        'q75': q75
    }


def recommend_blur(stats, metric="cosine"):
    """
    Recommend blur values based on distance statistics.
    """
    print("\n" + "=" * 80)
    print("RECOMMENDED BLUR VALUES")
    print("=" * 80)
    
    median = stats['median']
    q25 = stats['q25']
    q75 = stats['q75']
    
    # Metric-specific recommendations
    if metric.lower() == "cosine":
        # For cosine, distances are in [0, 2]
        # Typical median ~1.0 for random unit vectors
        blur_very_sharp = 0.05
        blur_sharp = 0.1
        blur_moderate = 0.2
        blur_relaxed = 0.5
        blur_very_relaxed = 1.0
        
        print(f"\nFor COSINE distance with your data:")
        print(f"  (Typical cosine distances: median={median:.4f})")
        
    elif metric.lower() == "euclidean":
        # Scale based on actual distances
        blur_very_sharp = 0.1 * median
        blur_sharp = 0.2 * median
        blur_moderate = 0.3 * median
        blur_relaxed = 0.5 * median
        blur_very_relaxed = 0.7 * median
        
        print(f"\nFor EUCLIDEAN distance with your data:")
        print(f"  (Typical euclidean distances: median={median:.4f})")
        
    else:
        # General scaling
        blur_very_sharp = 0.1 * median
        blur_sharp = 0.2 * median
        blur_moderate = 0.3 * median
        blur_relaxed = 0.5 * median
        blur_very_relaxed = 0.7 * median
        
        print(f"\nFor {metric.upper()} distance with your data:")
        print(f"  (Typical distances: median={median:.4f})")
    
    print(f"\n  1. Very Sharp (strict local matching):")
    print(f"     blur = {blur_very_sharp:.4f}")
    print(f"     → Use for: Contrastive learning, precise discrimination")
    
    print(f"\n  2. Sharp (local matching):")
    print(f"     blur = {blur_sharp:.4f}")
    print(f"     → Use for: Most supervised tasks, metric learning")
    
    print(f"\n  3. Moderate (balanced) ⭐ RECOMMENDED START:")
    print(f"     blur = {blur_moderate:.4f}")
    print(f"     → Use for: General-purpose, stable gradients")
    
    print(f"\n  4. Relaxed (smooth matching):")
    print(f"     blur = {blur_relaxed:.4f}")
    print(f"     → Use for: Distribution matching, alignment")
    
    print(f"\n  5. Very Relaxed (global matching):")
    print(f"     blur = {blur_very_relaxed:.4f}")
    print(f"     → Use for: Generative models, very smooth optimization")
    
    return {
        'very_sharp': blur_very_sharp,
        'sharp': blur_sharp,
        'moderate': blur_moderate,
        'relaxed': blur_relaxed,
        'very_relaxed': blur_very_relaxed
    }


def test_blur_values(embeddings_1, embeddings_2, metric="cosine", blur_values=None):
    """
    Test different blur values and show the resulting losses.
    """
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT BLUR VALUES")
    print("=" * 80)
    
    if blur_values is None:
        if metric.lower() == "cosine":
            blur_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        else:
            # Measure median first
            # Handle 3D tensors
            if embeddings_1.dim() == 3:
                e1_flat = embeddings_1.reshape(-1, embeddings_1.shape[-1])
                e2_flat = embeddings_2.reshape(-1, embeddings_2.shape[-1])
            else:
                e1_flat = embeddings_1
                e2_flat = embeddings_2
            
            sample_1 = e1_flat[:min(100, len(e1_flat))]
            sample_2 = e2_flat[:min(100, len(e2_flat))]
            
            if metric.lower() == "euclidean":
                median = torch.cdist(sample_1, sample_2, p=2).median().item()
            else:
                median = torch.cdist(sample_1, sample_2, p=1).median().item()
            
            blur_values = [
                0.1 * median, 0.2 * median, 0.3 * median,
                0.5 * median, 0.7 * median, 1.0 * median
            ]
    
    print(f"\nMetric: {metric}")
    print(f"Testing {len(blur_values)} blur values:\n")
    
    results = []
    for blur in blur_values:
        try:
            loss_fn = SamplesLoss(metric, blur=blur)
            result = loss_fn(embeddings_1, embeddings_2).mean().item()
            results.append((blur, result))
            print(f"  blur = {blur:8.4f}  →  loss = {result:10.6f}")
        except Exception as e:
            print(f"  blur = {blur:8.4f}  →  ERROR: {str(e)[:40]}")
    
    if results:
        # Find minimum and maximum
        min_blur, min_loss = min(results, key=lambda x: x[1])
        max_blur, max_loss = max(results, key=lambda x: x[1])
        
        print(f"\n  Minimum loss: {min_loss:.6f} at blur={min_blur:.4f}")
        print(f"  Maximum loss: {max_loss:.6f} at blur={max_blur:.4f}")
    
    return results


def plot_blur_sweep(results, metric="cosine"):
    """
    Plot loss vs blur values (optional, requires matplotlib).
    """
    try:
        import matplotlib.pyplot as plt
        
        if not results:
            print("No results to plot!")
            return
        
        blurs = [r[0] for r in results]
        losses = [r[1] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(blurs, losses, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Blur Parameter', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title(f'Loss vs Blur for {metric.upper()} Distance', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Mark minimum
        min_idx = losses.index(min(losses))
        plt.plot(blurs[min_idx], losses[min_idx], 'r*', markersize=15, 
                label=f'Min at blur={blurs[min_idx]:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'blur_analysis_{metric}.png', dpi=150)
        print(f"\n✅ Plot saved as 'blur_analysis_{metric}.png'")
        
    except ImportError:
        print("\n⚠️  Matplotlib not available, skipping plot.")
    except Exception as e:
        print(f"\n⚠️  Could not create plot: {e}")


def main():
    """
    Example usage with sample data.
    """
    print("\n" + "=" * 80)
    print("BLUR PARAMETER OPTIMIZATION TOOL")
    print("=" * 80)
    
    # Generate sample embeddings (replace with your actual data!)
    print("\n📊 Generating sample embeddings...")
    print("   (Replace this with your actual embeddings!)")
    
    batch_size = 32
    num_points = 100
    embedding_dim = 768
    
    # Simulate different embedding types
    print("\n" + "-" * 80)
    print("Example 1: Normalized Embeddings (like BERT, Word2Vec)")
    print("-" * 80)
    
    # Normalized embeddings
    embeddings_1 = torch.randn(batch_size, num_points, embedding_dim)
    embeddings_2 = torch.randn(batch_size, num_points, embedding_dim)
    embeddings_1 = torch.nn.functional.normalize(embeddings_1, dim=-1)
    embeddings_2 = torch.nn.functional.normalize(embeddings_2, dim=-1)
    
    # Analyze
    metric = "cosine"
    stats = analyze_distances(embeddings_1, embeddings_2, metric=metric)
    blur_recommendations = recommend_blur(stats, metric=metric)
    results = test_blur_values(embeddings_1, embeddings_2, metric=metric)
    
    # Try plotting
    plot_blur_sweep(results, metric=metric)
    
    # Second example
    print("\n\n" + "-" * 80)
    print("Example 2: Unnormalized Embeddings (general features)")
    print("-" * 80)
    
    # Unnormalized embeddings
    embeddings_1 = torch.randn(batch_size, num_points, embedding_dim)
    embeddings_2 = torch.randn(batch_size, num_points, embedding_dim)
    
    metric = "euclidean"
    stats = analyze_distances(embeddings_1, embeddings_2, metric=metric)
    blur_recommendations = recommend_blur(stats, metric=metric)
    results = test_blur_values(embeddings_1, embeddings_2, metric=metric)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Replace sample embeddings with your actual data")
    print("2. Run this script to get data-driven blur recommendations")
    print("3. Start with the 'Moderate' blur value")
    print("4. Fine-tune based on validation performance")
    print("\nSee BLUR_PARAMETER_GUIDE.md for detailed explanations!")


if __name__ == "__main__":
    main()


# ============================================================================
# QUICK USAGE TEMPLATE
# ============================================================================
"""
# Copy this code to use with YOUR data:

import torch
from geomloss import SamplesLoss

# Load your embeddings
your_embeddings_1 = ...  # Shape: (batch, num_points, dim)
your_embeddings_2 = ...  # Shape: (batch, num_points, dim)

# Choose your metric
metric = "cosine"  # or "euclidean", "manhattan", etc.

# Run analysis
from blur_optimizer import analyze_distances, recommend_blur, test_blur_values

stats = analyze_distances(your_embeddings_1, your_embeddings_2, metric=metric)
recommendations = recommend_blur(stats, metric=metric)
results = test_blur_values(your_embeddings_1, your_embeddings_2, metric=metric)

# Use the recommended moderate blur
optimal_blur = recommendations['moderate']
print(f"\n🎯 Using recommended blur: {optimal_blur:.4f}")

# Create your loss function
loss_fn = SamplesLoss(metric, blur=optimal_blur)
result = loss_fn(your_embeddings_1, your_embeddings_2)
print(f"Loss: {result.mean().item():.6f}")
"""
