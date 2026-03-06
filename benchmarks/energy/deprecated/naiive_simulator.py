# SPDX-License-Identifier: Apache-2.0
# csv format
# TIMESTAMP	ContextTokens	GeneratedTokens
# 2024-05-10 00:00:00.009930+00:00	2162	5

# Get distribution of ContextTokens and GeneratedTokens from entire dataset

import matplotlib.pyplot as plt
import pandas as pd

FILE_NAME = "AzureLLMInferenceTrace_conv_1week.csv"
START_DATE_TIME = "2024-05-13 00:00:00.009930+00:00"
END_DATE_TIME = "2024-05-14 00:00:00.009930+00:00"

# Read the CSV file
df = pd.read_csv(FILE_NAME)

# Convert timestamp to datetime and filter by date range
df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601")
df = df[(df["timestamp"] >= START_DATE_TIME)
        & (df["timestamp"] <= END_DATE_TIME)]

# Get distribution of ContextTokens and GeneratedTokens
print(df["ContextTokens"].describe())
print(df["GeneratedTokens"].describe())

# Mini-simulator
# 0.72s for processing 512 context tokens.
# 0.02s for processing 1 generated token.

# Simulate the number of requests being processed in the system at a given time
# based on processing times: 0.72s for 512 context tokens, 0.02s for 1 generated token


def calculate_processing_time(context_tokens, generated_tokens):
    """Calculate processing time for a request based on token counts"""
    context_time = (context_tokens /
                    512) * 0.72  # 0.72s per 512 context tokens
    generated_time = generated_tokens * 0.02  # 0.02s per generated token
    return context_time + generated_time


# Add processing time to dataframe
df["processing_time"] = df.apply(
    lambda row: calculate_processing_time(row["ContextTokens"], row[
        "GeneratedTokens"]),
    axis=1,
)

# Sort by timestamp to process requests in chronological order
df_sorted = df.sort_values("timestamp").reset_index(drop=True)


# Simulate concurrent processing
def simulate_concurrent_processing(df_sorted):
    """Simulate how many requests are being processed concurrently over time"""
    active_requests = []  # List of (end_time, request_id) tuples
    concurrent_counts = []  # List of (timestamp, count) tuples
    # use tqdm to show progress
    from tqdm import tqdm

    for idx, row in tqdm(df_sorted.iterrows(), total=len(df_sorted)):
        start_time = row["timestamp"]
        end_time = start_time + pd.Timedelta(seconds=row["processing_time"])

        # Remove completed requests
        active_requests = [(end_t, req_id) for end_t, req_id in active_requests
                           if end_t > start_time]

        # Add new request
        active_requests.append((end_time, idx))

        # Record concurrent count
        concurrent_counts.append((start_time, len(active_requests)))

    return concurrent_counts


# Run simulation
concurrent_counts = simulate_concurrent_processing(df_sorted)

# Convert to DataFrame for analysis
concurrent_df = pd.DataFrame(concurrent_counts,
                             columns=["timestamp", "concurrent_requests"])

# Analyze concurrent processing statistics
print("\nConcurrent Processing Statistics:")
print(concurrent_df["concurrent_requests"].describe())

# Plot concurrent requests over time
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.plot(
    concurrent_df["timestamp"],
    concurrent_df["concurrent_requests"],
    alpha=0.7,
    color="blue",
    linewidth=1,
)
ax.set_xlabel("Time")
ax.set_ylabel("Number of Concurrent Requests")
ax.set_title("Concurrent Requests Being Processed Over Time")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FILE_NAME}_concurrent_requests_over_time.png")
plt.close()

# Plot distribution of concurrent requests
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.hist(
    concurrent_df["concurrent_requests"],
    bins=50,
    density=True,
    alpha=0.7,
    color="blue",
    label="Concurrent Requests",
)
ax.set_xlabel("Number of Concurrent Requests")
ax.set_ylabel("Density")
ax.set_title("Distribution of Concurrent Requests")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FILE_NAME}_concurrent_requests_distribution.png")
plt.close()

print(
    f"Maximum concurrent requests: {concurrent_df['concurrent_requests'].max()}"
)
print(
    f"Average concurrent requests: {concurrent_df['concurrent_requests'].mean():.2f}"
)
