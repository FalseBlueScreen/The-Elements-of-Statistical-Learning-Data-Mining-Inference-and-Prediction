import numpy as np
import matplotlib.pyplot as plt

def main():
    # Model Complexity range
    complexity = np.linspace(0.5, 10, 100)

    # Synthetic Error Curves
    # Training Error: Monotonically decreasing
    # e.g., exponential decay + constant noise floor
    train_error = 2.5 * np.exp(-0.5 * complexity) + 0.5

    # Test Error: U-shaped
    # e.g., train_error + complexity penalty term (Variance)
    # Variance increases with complexity
    variance = 0.05 * (complexity ** 1.8)
    test_error = train_error + variance + 0.3 # Add offset

    # Find minimum test error for vertical line
    min_idx = np.argmin(test_error)
    min_complexity = complexity[min_idx]

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(complexity, train_error, 'b-', linewidth=2, label='Training Error')
    plt.plot(complexity, test_error, 'r-', linewidth=2, label='Test Error')

    # Vertical line at optimum
    plt.axvline(x=min_complexity, color='gray', linestyle='--', alpha=0.7)
    plt.text(min_complexity, plt.ylim()[1]*0.9, 'Optimum', ha='center', va='bottom', fontsize=10)

    # Labels for Bias/Variance regions
    # Left side (Low Complexity)
    plt.text(complexity[10], plt.ylim()[1]*0.8, 'High Bias\nLow Variance', ha='center', fontsize=12)

    # Right side (High Complexity)
    plt.text(complexity[-10], plt.ylim()[1]*0.8, 'Low Bias\nHigh Variance', ha='center', fontsize=12)

    plt.xlabel('Model Complexity', fontsize=14)
    plt.ylabel('Prediction Error', fontsize=14)
    plt.title('Behavior of Test and Training Error (Figure 7.1)', fontsize=16)

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Remove x/y ticks to make it conceptual (like the book usually does for conceptual plots)
    # But keeping them is fine for reproducibility. The book has "Low" "High" labels usually.
    plt.xticks([complexity[0], complexity[-1]], ['Low', 'High'])
    plt.yticks([]) # Remove y ticks values as it's qualitative

    plt.tight_layout()
    output_path = 'src/figures/output/fig_7_1.png'
    plt.savefig(output_path, dpi=300)
    print(f"Figure 7.1 generated in {output_path}")

if __name__ == "__main__":
    main()
