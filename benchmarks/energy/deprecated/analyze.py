# SPDX-License-Identifier: Apache-2.0
# csv format
# TIMESTAMP	ContextTokens	GeneratedTokens
# 2024-05-10 00:00:00.009930+00:00	2162	5

# Get distribution of ContextTokens and GeneratedTokens from entire dataset

import os

import pandas as pd

FILE_NAME = "AzureLLMInferenceTrace_conv_1week.csv"
START_DATE_TIME = "2024-05-14 00:00:00.009930+00:00"
END_DATE_TIME = "2024-05-15 00:00:00.009930+00:00"

########### Numerical Modding ###########
MOD = 30


def filter_mod(df, mod):
    """
    Returns a DataFrame containing only every 10th row from the input DataFrame.
    """
    return df.iloc[::mod].reset_index(drop=True)
    # Get distribution of ContextTokens and GeneratedTokens


########### Size Modding ###########
# MOD = 'LS'
# # Filter input size longer than 1024, output shorter than 100
# def filter_mod(df, mod):
#     return df[(df["ContextTokens"] > 1024) & (df["GeneratedTokens"] < 100)]

# If csv file already exists, we can use this instead
CSV_FILE = f"{FILE_NAME}_oneday_{START_DATE_TIME}_{END_DATE_TIME}_MOD{MOD}.csv"

# Make CSV_FILE argument suitable for file name
CSV_FILE = CSV_FILE.replace("/", "_")
# Remove spaces
CSV_FILE = CSV_FILE.replace(" ", "_")

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    # Convert timestamp to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601')
    print(f"{CSV_FILE} already exists")
else:
    # Read the CSV file
    df = pd.read_csv(FILE_NAME)

    # Convert timestamp to datetime and filter by date range
    df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601")
    df = df[(df["timestamp"] >= START_DATE_TIME)
            & (df["timestamp"] <= END_DATE_TIME)]

    # Save csv file, so it may be used later
    df = filter_mod(df, MOD)
    df.to_csv(CSV_FILE, index=False)

    # Filtering function : Use only every 10th request

print(df["ContextTokens"].describe())
print(df["GeneratedTokens"].describe())

# Plot density plot of ContextTokens and GeneratedTokens. Save as png
import matplotlib.pyplot as plt

# Plot density plot of ContextTokens and GeneratedTokens
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Plot ContextTokens distribution
ax1.hist(
    df["ContextTokens"],
    bins=100,
    density=True,
    alpha=0.7,
    color="blue",
    label="Context Tokens",
)
ax1.set_xlabel("Context Tokens")
ax1.set_ylabel("Density")
ax1.set_title("Distribution of Context Tokens")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot GeneratedTokens distribution
ax2.hist(
    df["GeneratedTokens"],
    bins=100,
    density=True,
    alpha=0.7,
    color="red",
    label="Generated Tokens",
)
ax2.set_xlabel("Generated Tokens")
ax2.set_ylabel("Density")
ax2.set_title("Distribution of Generated Tokens")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FILE_NAME}_context_tokens_generated_tokens_density.png")
plt.close()

# Extract hour from timestamp
df["hour"] = df["timestamp"].dt.hour

# Plot hourly distribution
fig, ax = plt.subplots(1, 1, figsize=(15, 8))

# Plot distribution of # requests per minute, per 20s, and per-hour
# Create time bins for per-minute, per 20s, and per-hour analysis
df["minute_bin"] = df["timestamp"].dt.floor("T")  # 'T' for minute
df["10s_bin"] = df["timestamp"].dt.floor("10S")  # '20S' for 20 seconds
df["hour_bin"] = df["timestamp"].dt.floor("H")  # 'H' for hour

# Count requests per minute, per 20s, and per hour
requests_per_minute = df.groupby("minute_bin").size()
requests_per_10s = df.groupby("10s_bin").size()
requests_per_hour = df.groupby("hour_bin").size()

# Create subplots for all three distributions
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

# Plot requests per minute distribution (converted to per second)
ax1.hist(
    requests_per_minute.values / 60,
    bins=50,
    density=True,
    alpha=0.7,
    color="green",
    label="Requests per Second (from minute bins)",
)
ax1.set_xlabel("Number of Requests per Second")
ax1.set_ylabel("Density")
ax1.set_title("Distribution of Requests per Second (from minute bins)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot requests per 10s distribution (converted to per second)
ax2.hist(
    requests_per_10s.values / 10,
    bins=50,
    density=True,
    alpha=0.7,
    color="orange",
    label="Requests per Second (from 10s bins)",
)
ax2.set_xlabel("Number of Requests per Second")
ax2.set_ylabel("Density")
ax2.set_title("Distribution of Requests per Second (from 10s bins)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot requests per hour distribution (converted to per second)
ax3.hist(
    requests_per_hour.values / 3600,
    bins=30,
    density=True,
    alpha=0.7,
    color="purple",
    label="Requests per Second (from hour bins)",
)
ax3.set_xlabel("Number of Requests per Second")
ax3.set_ylabel("Density")
ax3.set_title("Distribution of Requests per Second (from hour bins)")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FILE_NAME}_requests_per_time_density.png")
plt.close()

# Also plot time series of requests over time
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

# Plot requests per minute over time (converted to per second)
ax1.plot(
    requests_per_minute.index,
    requests_per_minute.values / 60,
    alpha=0.7,
    color="green",
    linewidth=1,
)
ax1.set_xlabel("Time")
ax1.set_ylabel("Number of Requests per Second")
ax1.set_title("Requests per Second Over Time (from minute bins)")
ax1.grid(True, alpha=0.3)

# Plot requests per 10s over time (converted to per second)
ax2.plot(
    requests_per_10s.index,
    requests_per_10s.values / 10,
    alpha=0.7,
    color="orange",
    linewidth=1,
)
ax2.set_xlabel("Time")
ax2.set_ylabel("Number of Requests per Second")
ax2.set_title("Requests per Second Over Time (from 10s bins)")
ax2.grid(True, alpha=0.3)

# Plot requests per hour over time (converted to per second)
ax3.plot(
    requests_per_hour.index,
    requests_per_hour.values / 3600,
    alpha=0.7,
    color="purple",
    linewidth=1,
)
ax3.set_xlabel("Time")
ax3.set_ylabel("Number of Requests per Second")
ax3.set_title("Requests per Second Over Time (from hour bins)")
ax3.grid(True, alpha=0.3)

# All y axis 0-100
ax1.set_ylim(0, None)
ax2.set_ylim(0, None)
ax3.set_ylim(0, None)

plt.tight_layout()
plt.savefig(f"{FILE_NAME}_requests_over_time.png")
plt.close()
