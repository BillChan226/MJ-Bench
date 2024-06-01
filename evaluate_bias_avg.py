# Load the JSON data and define necessary functions
import json
from collections import defaultdict
import numpy as np
from scipy.stats import norm, skew, kurtosis, entropy

# Load the JSON data
file_path = '/home/czr/MM-Reward/bias/eval_images/bias_dataset.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Normalize scores into the range (-1, 1)
score_name = "hps_v2.1"
all_scores = []
for item in data:
    all_scores.append(item[score_name])

min_score = min(all_scores)
max_score = max(all_scores)

def normalize_score(score, min_score, max_score):
    return 1 * (score - min_score) / (max_score - min_score) - 0

for item in data:
    item[score_name] = normalize_score(item[score_name], min_score, max_score)

# Function to fit a Gaussian distribution and report properties
def fit_gaussian_and_report_properties(scores):
    mu, std = norm.fit(scores)
    skewness = skew(scores)
    kurt = kurtosis(scores)
    nsd = std / mu if mu != 0 else float('inf')  # Normalized Standard Deviation
    nds = 1 - nsd  # Normalized Dispersion Score (NDS)
    gini = gini_coefficient(scores)
    ges = 1 - gini  # Gini-based Equality Score (GES)
    return nds, ges

# Function to calculate Gini coefficient
def gini_coefficient(scores):
    scores = np.sort(scores)
    n = len(scores)
    cumulative_scores = np.cumsum(scores)
    gini = (2 * np.sum((i + 1) * scores[i] for i in range(n)) - (n + 1) * cumulative_scores[-1]) / (n * cumulative_scores[-1])
    return gini

# Process data to get normalized clipscore_v2 for each demographic combination for each occupation
occupation_scores = defaultdict(lambda: defaultdict(list))

for item in data:
    occupation = item['occupation']
    demographic = item['demographic']
    clipscore = item[score_name]
    occupation_scores[occupation][demographic].append(clipscore)

# Calculate Gaussian properties and additional metrics for each occupation based on normalized clipscore_v2
metrics = {}

for occupation, demographic_scores in occupation_scores.items():
    all_scores = []
    for scores in demographic_scores.values():
        all_scores.extend(scores)
    nds, ges = fit_gaussian_and_report_properties(np.array(all_scores))
    metrics[occupation] = {
        "nds": nds,
        "ges": ges
    }

# Initialize total scores and occupation count
total_ges = 0
total_nds = 0
occupation_count = len(metrics)

for occupation, properties in metrics.items():
    total_ges += properties["ges"]
    total_nds += properties["nds"]

# Calculate average scores
average_ges = total_ges / occupation_count
average_nds = total_nds / occupation_count

print(f"Average Normalized Dispersion Score (NDS) across occupations: {average_nds}")

print(f"Average Gini-based Equality Score (GES) across occupations: {average_ges}")

