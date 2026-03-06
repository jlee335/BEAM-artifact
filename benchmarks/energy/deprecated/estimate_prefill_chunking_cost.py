import pandas as pd
import math
PROMPT_LEN = 2048
DECODE_BATCH_SIZE = 32

chunk_sizes = [64, 128, 256, 512, 1024, 2048]

# Energy modelling

MODEL_NAME = "huggyllama_llama-30b"

GPU_NAME = "NVIDIA GeForce RTX 3090"

# Get the energy consumption of the model on the GPU
df = pd.read_csv(f"dvfs_profile_{GPU_NAME}_{MODEL_NAME}_one.csv")

bounded_df = pd.read_csv(f"dvfs_profile_{GPU_NAME}_{MODEL_NAME}_bounded.csv")
bounded_df = bounded_df[bounded_df['batch_size'] == DECODE_BATCH_SIZE]

batch_sizes = df['batch_size'].unique()
batch_sizes.sort()

def find_smallest_batch_size_greater_than(size):
    for bs in batch_sizes:
        if bs > size:
            return bs
    return batch_sizes[-1]

df['energy_consumption'] = df['energy_consumption'] / 1000 # mJ to J
bounded_df['energy_consumption'] = bounded_df['energy_consumption'] / 1000 # mJ to J

data = []
for chunk_size in chunk_sizes:
    # ceil
    num_chunks = int(PROMPT_LEN / chunk_size)
    
    energy_needed = 0
    
    remainder = PROMPT_LEN % chunk_size
    for i in range(num_chunks):
        energy_needed += df[df['batch_size'] == chunk_size]['energy_consumption'].values[0]

    if remainder > 0:
        # closest-batch-to-remainder
        closest_batch = find_smallest_batch_size_greater_than(remainder)
        energy_needed += df[df['batch_size'] == closest_batch]['energy_consumption'].values[0]

    # format to 2 decimal places
    print(f"Energy needed for {chunk_size} chunk size: {energy_needed:.2f} J")
    data.append(energy_needed)

bounded_data_defaultclock = [] # default clock 1710
bounded_data = []
bounded_clocks = []
for chunk_size in chunk_sizes:
    num_chunks = math.ceil(PROMPT_LEN / chunk_size)
    energy_needed = 0
    energy_needed_defaultclock = 0
    for i in range(3):
        chunk_size_df = bounded_df[bounded_df['bound_size'] == chunk_size]
        # Minimum-energy configuration
        min_energy_row = chunk_size_df[chunk_size_df['energy_consumption'] == chunk_size_df['energy_consumption'].min()]
        energy_consumption = min_energy_row['energy_consumption'].values[0]
        clock = min_energy_row['clock'].values[0]
        print(f"Minimum energy consumption for {chunk_size} chunk size: {energy_consumption:.2f} J at {clock} MHz")
        energy_needed += energy_consumption
        
        print(chunk_size_df)
        
        # Default clock configuration
        default_clock_row = chunk_size_df[chunk_size_df['clock'] == 1830]
        print(default_clock_row)
        default_clock_energy = default_clock_row['energy_consumption'].values[0]
        energy_needed_defaultclock += default_clock_energy
        
        
        
        
        
    bounded_clocks.append(clock)
    bounded_data.append(energy_needed)
    bounded_data_defaultclock.append(energy_needed_defaultclock)

print(bounded_data)
print(bounded_clocks)
print(bounded_data_defaultclock)
# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))



# Plot bounded_data as dashed line  
plt.plot(chunk_sizes, bounded_data, '--', label='Decode')
plt.plot(chunk_sizes, bounded_data_defaultclock, '--', label='Decode (default clock)')
# Start from y = 0

plt.plot(chunk_sizes, data, label='Prefill')
plt.xlabel('Chunk Size')
plt.ylabel('Energy Needed (J)')
plt.title('Energy Needed for Different Chunk Sizes')


sum_data = [x + y for x, y in zip(data, bounded_data)]
sum_data_defaultclock = [x + y for x, y in zip(data, bounded_data_defaultclock)]
# Plot sum_data as solid line
plt.plot(chunk_sizes, sum_data, label='Sum', color='gray')
plt.plot(chunk_sizes, sum_data_defaultclock, label='Sum (default clock)', color='gray', linestyle='-.')

# For Sum, Place a dot in the minimum, min_clock, and max_clock. Write down energy next to them
min_energy_index = sum_data.index(min(sum_data))
min_energy_value = min(sum_data)
min_energy_clock = chunk_sizes[min_energy_index]

smallest_chunk_energy = sum_data[0]
smallest_chunk_clock = chunk_sizes[0]

largest_chunk_energy = sum_data[-1]
largest_chunk_clock = chunk_sizes[-1]

plt.scatter(min_energy_clock, min_energy_value, color='red')
plt.scatter(smallest_chunk_clock, smallest_chunk_energy, color='blue')
plt.scatter(largest_chunk_clock, largest_chunk_energy, color='green')
# Write down energy next to them
plt.text(min_energy_clock, min_energy_value, f'{min_energy_value:.2f}', color='black', fontweight='bold')
plt.text(smallest_chunk_clock, smallest_chunk_energy, f'{smallest_chunk_energy:.2f}', color='black', fontweight='bold')
plt.text(largest_chunk_clock, largest_chunk_energy, f'{largest_chunk_energy:.2f}', color='black', fontweight='bold')

# For the bounded data, write down the clock next to them
for i in range(len(bounded_clocks)):
    plt.text(chunk_sizes[i], bounded_data[i], f'{bounded_clocks[i]}', color='black', fontweight='bold')



# Set labels
plt.legend()

# Size 4 x 2

# save
plt.savefig('energy_needed_for_different_chunk_sizes.png')
