import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime

def generate_graphs():
    # Find the most recent benchmark CSV file
    csv_files = glob.glob("results/benchmark_*.csv")
    if not csv_files:
        print("No benchmark results found. Run run.py first.")
        return
    
    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"Generating graphs from: {latest_csv}")
    
    # Load the data
    df = pd.read_csv(latest_csv)
    
    # Create results directory if it doesn't exist
    os.makedirs("graphs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 1. Tokens per minute by model
    plt.figure(figsize=(12, 6))
    sns.barplot(x="model_id", y="tokens_per_minute", data=df, estimator="mean", errorbar=("ci", 95))
    plt.title("Average Tokens per Minute by Model")
    plt.xlabel("Model")
    plt.ylabel("Tokens per Minute")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"graphs/tokens_per_minute_{timestamp}.png")
    
    # 2. Invocations per minute (calculated from duration)
    df["invocations_per_minute"] = 60 / df["duration"]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="model_id", y="invocations_per_minute", data=df, estimator="mean", errorbar=("ci", 95))
    plt.title("Average Invocations per Minute by Model")
    plt.xlabel("Model")
    plt.ylabel("Invocations per Minute")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"graphs/invocations_per_minute_{timestamp}.png")
    
    # 3. Success rate by model
    plt.figure(figsize=(12, 6))
    success_rate = df.groupby("model_id")["success"].mean() * 100
    success_rate.plot(kind="bar")
    plt.title("Success Rate by Model")
    plt.xlabel("Model")
    plt.ylabel("Success Rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"graphs/success_rate_{timestamp}.png")
    
    # 4. Average response time by model
    plt.figure(figsize=(12, 6))
    sns.barplot(x="model_id", y="duration", data=df, estimator="mean", errorbar=("ci", 95))
    plt.title("Average Response Time by Model")
    plt.xlabel("Model")
    plt.ylabel("Response Time (seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"graphs/response_time_{timestamp}.png")
    
    # 5. Average tokens sent and received by model
    plt.figure(figsize=(14, 8))
    tokens_data = df.groupby("model_id").agg({
        "input_tokens": "mean",
        "output_tokens": "mean",
        "total_tokens": "mean"
    }).reset_index()
    
    # Create a stacked bar chart for input and output tokens
    ax = tokens_data.plot(
        x="model_id", 
        y=["input_tokens", "output_tokens"],
        kind="bar", 
        stacked=True,
        figsize=(14, 8),
        color=["#3498db", "#2ecc71"]
    )
    
    # Add total tokens as text on top of each bar
    for i, row in enumerate(tokens_data.itertuples()):
        ax.text(
            i, 
            row.total_tokens + 100,  # Position text slightly above the bar
            f"Total: {int(row.total_tokens)}",
            ha="center",
            fontweight="bold"
        )
    
    plt.title("Average Tokens by Model (Input vs Output)", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Average Tokens", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Input Tokens", "Output Tokens"])
    plt.tight_layout()
    plt.savefig(f"graphs/tokens_sent_received_{timestamp}.png")
    
    # 5b. Create a pie chart showing the proportion of input vs output tokens for each model
    plt.figure(figsize=(16, 10))
    
    # Create subplots for each model
    fig, axes = plt.subplots(1, len(tokens_data), figsize=(16, 6))
    
    for i, (ax, row) in enumerate(zip(axes, tokens_data.itertuples())):
        model_name = row.model_id.split('.')[-1].split('-')[0]  # Extract short model name
        token_data = [row.input_tokens, row.output_tokens]
        ax.pie(
            token_data, 
            labels=["Input", "Output"],
            autopct='%1.1f%%',
            startangle=90,
            colors=["#3498db", "#2ecc71"]
        )
        ax.set_title(f"{model_name}\n({int(row.total_tokens)} tokens)")
    
    plt.suptitle("Token Distribution by Model (Input vs Output)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"graphs/token_distribution_pie_{timestamp}.png")
    
    # 6. Combined metrics table
    metrics = df.groupby("model_id").agg({
        "duration": "mean",
        "input_tokens": "mean",
        "output_tokens": "mean",
        "total_tokens": "mean",
        "tokens_per_minute": "mean",
        "invocations_per_minute": "mean",
        "success": lambda x: x.mean() * 100
    }).reset_index()
    
    metrics.columns = ["Model", "Avg Response Time (s)", "Avg Tokens Sent", 
                      "Avg Tokens Received", "Avg Total Tokens", "Tokens per Minute", 
                      "Invocations per Minute", "Success Rate (%)"]
    
    # Save metrics to CSV
    metrics.to_csv(f"results/metrics_summary_{timestamp}.csv", index=False)
    
    print(f"Graphs generated in the 'graphs' directory")
    print(f"Metrics summary saved to 'results/metrics_summary_{timestamp}.csv'")
    
    # Display the metrics table
    print("\nMetrics Summary:")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

if __name__ == "__main__":
    generate_graphs()
