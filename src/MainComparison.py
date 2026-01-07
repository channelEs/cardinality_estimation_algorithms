import matplotlib.pyplot as plt
import numpy as np
import csv
import randomhash
from HyperLogLog import ParallelHyperLogLog
from Recordinality import ParallelRecordinality

NUM_TRIALS = 10
NUM_HASHES = 30

def run_real_data_experiment(txt_files, dat_files):
    print(f"\n--- Real Data Comparison ({NUM_TRIALS} Trials) ---")
    
    for file_idx, txt_file in enumerate(txt_files):
        dat_file = dat_files[file_idx]
        dataset_name = txt_file.split('/')[-1].split('.')[0] # "dracula"

        print(f"\n> Processing Dataset: {dataset_name} [{txt_file}]...")

        # .dat file first
        true_n_dat = 0
        try:
            with open(dat_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip(): true_n_dat += 1
        except FileNotFoundError:
            print(f"  Warning: {dat_file} not found. Relying on txt count.")

        # compute Exact Unique Count
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"  Error: {txt_file} not found. Skipping.")
            continue
        true_n = len(set(words))
        print(f"  Loaded {len(words)} words. True Unique Count (n): {true_n}")
        if true_n_dat > 0:
            print(f"  (Reference from .dat file was: {true_n_dat})")

        b_values = range(6, 13)
        results_data = {b: {'hll_errors': [], 'rec_errors': [], 'bits': 0, 'k': 0} for b in b_values}

        for trial in range(NUM_TRIALS):
            print(f"  - Trial {trial + 1}/{NUM_TRIALS}...")
            
            # Fresh Hash Family for every trial
            hf = randomhash.RandomHashFamily(count=NUM_HASHES)
            
            for b in b_values:
                m = 1 << b
                bits = m * 5
                k_equiv = int(bits / 32)
                if k_equiv < 1: k_equiv = 1
                
                results_data[b]['bits'] = bits
                results_data[b]['k'] = k_equiv

                # 1. HLL
                HLL = ParallelHyperLogLog(b=b, num_hashes=NUM_HASHES, random_hash_functions=hf)
                for w in words: HLL.add(w)
                hll_est = np.mean(HLL.count())
                hll_err = abs(hll_est - true_n) / true_n * 100
                results_data[b]['hll_errors'].append(hll_err)

                # 2. REC
                REC = ParallelRecordinality(k=k_equiv, num_hashes=NUM_HASHES, random_hash_functions=hf)
                for w in words: REC.add(w)
                
                rec_est = np.mean(REC.count())
                rec_err = abs(rec_est - true_n) / true_n * 100
                results_data[b]['rec_errors'].append(rec_err)

        csv_filename = f"{dataset_name}_results.csv"
        print(f"  > Saving results to {csv_filename}...")
        
        plot_data = {
            'x_bits': [],
            'hll_means': [],
            'rec_means': [],
            'hll_scatter_y': [],
            'rec_scatter_y': []
        }

        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            header = ["b", "m", "Memory_Bits", "REC_k", "True_n", "HLL_Mean_Err", "HLL_Std_Err", "REC_Mean_Err", "REC_Std_Err"]
            writer.writerow(header)

            for b in b_values:
                data = results_data[b]
                
                # Stats
                h_mean = np.mean(data['hll_errors'])
                h_std = np.std(data['hll_errors'])
                r_mean = np.mean(data['rec_errors'])
                r_std = np.std(data['rec_errors'])
                
                # Save to CSV
                row = [b, 1<<b, data['bits'], data['k'], true_n,
                       f"{h_mean:.4f}", f"{h_std:.4f}", f"{r_mean:.4f}", f"{r_std:.4f}"]
                writer.writerow(row)
                
                # Prep Plot Data
                plot_data['x_bits'].append(data['bits'])
                plot_data['hll_means'].append(h_mean)
                plot_data['rec_means'].append(r_mean)
                plot_data['hll_scatter_y'].append(data['hll_errors']) # List of trial errors
                plot_data['rec_scatter_y'].append(data['rec_errors']) # List of trial errors

        # Plotting
        plt.figure(figsize=(10, 6))
        
        for i, bits in enumerate(plot_data['x_bits']):
            # HLL Points
            y_vals_h = plot_data['hll_scatter_y'][i]
            x_vals_h = [bits] * len(y_vals_h)
            plt.scatter(x_vals_h, y_vals_h, color='blue', alpha=0.3, s=15)
            
            # REC Points (slightly offset x for visibility if desired, or same x)
            y_vals_r = plot_data['rec_scatter_y'][i]
            x_vals_r = [bits] * len(y_vals_r)
            plt.scatter(x_vals_r, y_vals_r, color='orange', marker='s', alpha=0.3, s=15)

        # Plot the Means
        plt.plot(plot_data['x_bits'], plot_data['hll_means'], 'b-o', linewidth=2, label='HLL Mean')
        plt.plot(plot_data['x_bits'], plot_data['rec_means'], color='orange', marker='s', linestyle='-', linewidth=2, label='REC Mean')

        # Formatting
        plt.title(f"HLL vs Recordinality: Real Data Analysis ({dataset_name})")
        plt.xlabel("Memory Usage (Bits)")
        plt.ylabel("Relative Error (%)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.4)
        
        # Custom Legend to explain scatter vs line
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='orange', lw=2),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.5, label='HLL Trials'),
                        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', alpha=0.5, label='REC Trials')]
        plt.legend(custom_lines, ['HLL Mean', 'REC Mean', 'HLL Trials', 'REC Trials'])

        plot_filename = f"{dataset_name}_robust_plot.png"
        plt.savefig(plot_filename)
        print(f"  > Plot saved to {plot_filename}")
        plt.close() # Close to prevent overlapping plots in loops

def generate_zipfian_data(N, n, alpha):
    # print(f"  > Generating Stream (N={N}, n={n}, alpha={alpha})...")
    ranks = np.arange(1, n + 1)
    if alpha == 0:
        probabilities = np.ones(n) / n
    else:
        weights = 1.0 / np.power(ranks, alpha)
        probabilities = weights / np.sum(weights)
    
    stream = np.random.choice(ranks, size=N, p=probabilities)
    stream_str = [str(x) for x in stream]
    return stream_str

def run_synthetic_data_experiment(N=100000, n=10000, alpha=1.0):
    print(f"\n--- Experiment: Synthetic Data ({NUM_TRIALS} runs) ---")
    
    b_values = range(4, 15)
    
    # Storage for all runs: data[b_index][trial_index]
    hll_all_errors = {b: [] for b in b_values}
    rec_all_errors = {b: [] for b in b_values}
    mem_bits_map = {} # Store bits for X-axis

    for trial in range(NUM_TRIALS):
        print(f"  > Trial {trial+1}/{NUM_TRIALS}...")
        
        # New randomness for every trial
        hf = randomhash.RandomHashFamily(count=NUM_HASHES)
        stream = generate_zipfian_data(N, n, alpha)
        true_n = len(set(stream))

        for b in b_values:
            m = 1 << b
            bits = m * 5
            mem_bits_map[b] = bits
            k = int(bits / 32)
            if k < 1: k = 1

            # HLL
            hll = ParallelHyperLogLog(b=b, num_hashes=NUM_HASHES, random_hash_functions=hf)
            for item in stream: hll.add(item)
            err_hll = abs(np.mean(hll.count()) - true_n) / true_n * 100
            hll_all_errors[b].append(err_hll)

            # REC
            rec = ParallelRecordinality(k=k, num_hashes=NUM_HASHES, random_hash_functions=hf)
            for item in stream: rec.add(item)
            err_rec = abs(np.mean(rec.count()) - true_n) / true_n * 100
            rec_all_errors[b].append(err_rec)

    csv_rows = []
    x_axis = []
    hll_means = []
    rec_means = []

    for b in b_values:
        m = 1 << b
        bits = mem_bits_map[b]
        k = int(bits / 32)
        if k < 1: k = 1
        
        h_mean = np.mean(hll_all_errors[b])
        h_std = np.std(hll_all_errors[b])
        r_mean = np.mean(rec_all_errors[b])
        r_std = np.std(rec_all_errors[b])
        
        x_axis.append(bits)
        hll_means.append(h_mean)
        rec_means.append(r_mean)
        
        csv_rows.append([b, m, k, f"{h_mean:.4f}", f"{h_std:.4f}", f"{r_mean:.4f}", f"{r_std:.4f}"])

    with open("csv_exp_synthetic_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["b", "m", "k", "HLL_Mean", "HLL_Std", "REC_Mean", "REC_Std"])
        writer.writerows(csv_rows)
        print("  > Saved csv_exp_synthetic_data.csv")
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Individual Runs (Faint lines)
    for trial in range(NUM_TRIALS):
        # Extract the y-values for this specific trial
        y_hll = [hll_all_errors[b][trial] for b in b_values]
        y_rec = [rec_all_errors[b][trial] for b in b_values]
        
        plt.plot(x_axis, y_hll, color='blue', alpha=0.15, linewidth=1)
        plt.plot(x_axis, y_rec, color='orange', alpha=0.15, linewidth=1)

    # Plot Means
    plt.plot(x_axis, hll_means, 'b-o', linewidth=2.5, label='HLL Mean')
    plt.plot(x_axis, rec_means, color='orange', marker='s', linestyle='-', linewidth=2.5, label='REC Mean')

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Memory (Bits)')
    plt.ylabel('Relative Error (%)')
    plt.title(f'Synthetic Data Experiment ({NUM_TRIALS} Runs) with N={N}, n={n}, alpha={alpha}')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig("plot_exp_synthetic_data.png")
    plt.show()

def run_alpha_experiment(N=50000, n=5000):
    print(f"\n--- Experiment: Alpha Stress ({NUM_TRIALS} runs) ---")
    
    alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    FIXED_B = 10
    FIXED_K = 160
    hll_results = {a: [] for a in alpha_values}
    rec_results = {a: [] for a in alpha_values}
    
    for trial in range(NUM_TRIALS):
        print(f"  > Trial {trial+1}/{NUM_TRIALS}...")
        hf = randomhash.RandomHashFamily(count=NUM_HASHES)
        
        for alpha in alpha_values:
            stream = generate_zipfian_data(N, n, alpha)
            true_n = len(set(stream))
            
            # HLL
            hll = ParallelHyperLogLog(b=FIXED_B, num_hashes=NUM_HASHES, random_hash_functions=hf)
            for item in stream: hll.add(item)
            err = abs(np.mean(hll.count()) - true_n) / true_n * 100
            hll_results[alpha].append(err)
            
            # REC
            rec = ParallelRecordinality(k=FIXED_K, num_hashes=NUM_HASHES, random_hash_functions=hf)
            for item in stream: rec.add(item)
            err = abs(np.mean(rec.count()) - true_n) / true_n * 100
            rec_results[alpha].append(err)

    # Aggregation
    csv_rows = []
    hll_means = []
    rec_means = []
    
    for alpha in alpha_values:
        h_mean = np.mean(hll_results[alpha])
        h_std = np.std(hll_results[alpha])
        r_mean = np.mean(rec_results[alpha])
        r_std = np.std(rec_results[alpha])
        
        hll_means.append(h_mean)
        rec_means.append(r_mean)
        csv_rows.append([alpha, f"{h_mean:.4f}", f"{h_std:.4f}", f"{r_mean:.4f}", f"{r_std:.4f}"])
        
    with open("csv_exp_alpha.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Alpha", "HLL_Mean", "HLL_Std", "REC_Mean", "REC_Std"])
        writer.writerows(csv_rows)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Individual Runs
    for trial in range(NUM_TRIALS):
        y_h = [hll_results[a][trial] for a in alpha_values]
        y_r = [rec_results[a][trial] for a in alpha_values]
        plt.plot(alpha_values, y_h, color='blue', alpha=0.15)
        plt.plot(alpha_values, y_r, color='orange', alpha=0.15)
        
    # Means
    plt.plot(alpha_values, hll_means, 'b-o', linewidth=2.5, label='HLL Mean')
    plt.plot(alpha_values, rec_means, color='orange', marker='s', linestyle='-', linewidth=2.5, label='REC Mean')
    
    plt.xlabel('Zipfian Skew (Alpha)')
    plt.ylabel('Relative Error (%)')
    plt.title(f'Alpha Experiment ({NUM_TRIALS} Runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_exp_alpha.png")
    plt.show()

def run_scale_impact_experiment():
    print(f"\n--- Experiment: Scale Impact ({NUM_TRIALS} runs) ---")
    
    FIXED_B = 10
    FIXED_K = 160
    ALPHA = 1.0

    # --- Part 1: Varying N ---
    N_values = [5000, 10000, 50000, 100000, 500000, 1000000]
    n_fixed = 5000
    
    hll_N_data = {N: [] for N in N_values}
    rec_N_data = {N: [] for N in N_values}
    
    print("  > Varying N...")
    for trial in range(NUM_TRIALS):
        print(f"  > Trial {trial+1}/{NUM_TRIALS}...")
        hf = randomhash.RandomHashFamily(count=NUM_HASHES)
        for N in N_values:
            stream = generate_zipfian_data(N, n_fixed, ALPHA)
            true_n = len(set(stream))
            
            hll = ParallelHyperLogLog(b=FIXED_B, num_hashes=NUM_HASHES, random_hash_functions=hf)
            for item in stream: hll.add(item)
            hll_N_data[N].append(abs(np.mean(hll.count()) - true_n) / true_n * 100)
            
            rec = ParallelRecordinality(k=FIXED_K, num_hashes=NUM_HASHES, random_hash_functions=hf)
            for item in stream: rec.add(item)
            rec_N_data[N].append(abs(np.mean(rec.count()) - true_n) / true_n * 100)

    # Save & Plot N
    csv_rows = []
    h_means, r_means = [], []
    for N in N_values:
        h_m = np.mean(hll_N_data[N])
        r_m = np.mean(rec_N_data[N])
        h_means.append(h_m)
        r_means.append(r_m)
        csv_rows.append([N, f"{h_m:.4f}", f"{np.std(hll_N_data[N]):.4f}", f"{r_m:.4f}", f"{np.std(rec_N_data[N]):.4f}"])
        
    with open("csv_exp_impact_N.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "HLL_Mean", "HLL_Std", "REC_Mean", "REC_Std"])
        writer.writerows(csv_rows)

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, h_means, 'b-o', label='HLL Mean')
    plt.plot(N_values, r_means, 'tab:orange', marker='s', label='REC Mean')
    plt.xscale('log')
    plt.title(f"Impact of N in Synthetic Data ({NUM_TRIALS} Runs)")
    plt.xlabel("Stream Length (N)")
    plt.ylabel("Relative Error (%)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("plot_exp_impact_N.png")
    plt.show()

    # --- Part 2: Varying n ---
    N_fixed = 1000000
    n_values = [100, 1000, 5000, 10000, 50000, 100000]
    
    hll_n_data = {n: [] for n in n_values}
    rec_n_data = {n: [] for n in n_values}
    true_n_map = {}

    print("  > Varying n...")
    for trial in range(NUM_TRIALS):
        print(f"  > Trial {trial+1}/{NUM_TRIALS}...")
        hf = randomhash.RandomHashFamily(count=NUM_HASHES)
        for n in n_values:
            stream = generate_zipfian_data(N_fixed, n, ALPHA)
            true_n = len(set(stream))
            
            if n not in true_n_map: true_n_map[n] = []
            true_n_map[n].append(true_n)

            hll = ParallelHyperLogLog(b=FIXED_B, num_hashes=NUM_HASHES, random_hash_functions=hf)
            for item in stream: hll.add(item)
            hll_n_data[n].append(abs(np.mean(hll.count()) - true_n) / true_n * 100)
            
            rec = ParallelRecordinality(k=FIXED_K, num_hashes=NUM_HASHES, random_hash_functions=hf)
            for item in stream: rec.add(item)
            rec_n_data[n].append(abs(np.mean(rec.count()) - true_n) / true_n * 100)

    # Save & Plot n
    csv_rows = []
    h_means, r_means, x_axis = [], [], []
    for n in n_values:
        avg_true_n = int(np.mean(true_n_map[n]))
        x_axis.append(avg_true_n)
        
        h_m = np.mean(hll_n_data[n])
        r_m = np.mean(rec_n_data[n])
        h_means.append(h_m)
        r_means.append(r_m)
        csv_rows.append([n, avg_true_n, f"{h_m:.4f}", f"{np.std(hll_n_data[n]):.4f}", f"{r_m:.4f}", f"{np.std(rec_n_data[n]):.4f}"])

    with open("csv_exp_impact_n.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Requested_n", "Avg_True_n", "HLL_Mean", "HLL_Std", "REC_Mean", "REC_Std"])
        writer.writerows(csv_rows)
        
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, h_means, 'b-o', label='HLL Mean')
    plt.plot(x_axis, r_means, 'tab:orange', marker='s', label='REC Mean')
    plt.xscale('log')
    plt.title(f"Impact of Cardinality n in Synthetic Data ({NUM_TRIALS} Runs)")
    plt.xlabel("True Cardinality (n)")
    plt.ylabel("Relative Error (%)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("plot_exp_impact_n.png")
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    data_sets_path = "../datasets/"
    my_txt_files = [f"{data_sets_path}dracula.txt", f"{data_sets_path}crusoe.txt", f"{data_sets_path}iliad.txt"] 
    my_dat_files = [f"{data_sets_path}dracula.dat", f"{data_sets_path}crusoe.dat", f"{data_sets_path}iliad.dat"]
    run_real_data_experiment(my_txt_files, my_dat_files)
    
    run_synthetic_data_experiment(N=100000, n=10000, alpha=1.0)
    run_alpha_experiment(N=50000, n=5000)
    run_scale_impact_experiment()
