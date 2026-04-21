#!/usr/bin/env python3
# github.com/rouze-d (enhanced with MCMC)

import sys
import argparse
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import math
import time

# =========================
# CONFIG (default, bisa di-override via arg)
# =========================
TOP_N = 13
PREDICT_SETS = 20
MONTE_CARLO_N = 1000
MCMC_STEPS = 5000      # jumlah langkah MCMC
MCMC_BURNIN = 1000     # burn-in
MCMC_SETS = 10         # jumlah set unik yang diambil dari MCMC

# Parameter analisis baru
HOT_THRESHOLD = 0.7
COLD_THRESHOLD = 0.3
PERIODICITY_LOOKBACK = 20

DRAW_COLS = [f"DrawnNo{i}" for i in range(1, 7)]

# =========================
# UTILS
# =========================
def detect_game(df):
    max_num = df[DRAW_COLS].max().max()
    if max_num <= 45:
        return "6/45"
    elif max_num <= 49:
        return "6/49"
    elif max_num <= 55:
        return "6/55"
    elif max_num <= 58:
        return "6/58"
    else:
        return f"6/{max_num}"

def ascii_histogram(counter, width=40):
    max_count = max(counter.values())
    for num, cnt in counter.most_common():
        bar_len = int(cnt / max_count * width)
        bar = "#" * bar_len
        print(f"{num:>2} | {bar} ({cnt})")

def variance_score(matches):
    return np.var(matches)

# =========================
# ANALISIS BARU: ZONES
# =========================
def analyze_zones(df, draw_cols):
    zones = defaultdict(list)
    zone_hits = defaultdict(int)
    
    for _, row in df.iterrows():
        for col in draw_cols:
            num = row[col]
            zone = ((num - 1) // 10) + 1
            zones[zone].append(num)
            zone_hits[zone] += 1
    
    print("\n🎯 ZONE ANALYSIS")
    print("=" * 50)
    total_hits = sum(zone_hits.values())
    for zone in sorted(zone_hits.keys()):
        percentage = (zone_hits[zone] / total_hits) * 100
        start_num = (zone - 1) * 10 + 1
        end_num = zone * 10
        print(f"Zone {zone} ({start_num:02d}-{end_num:02d}): {zone_hits[zone]:3d} hits ({percentage:5.1f}%)")
    
    return zone_hits

# =========================
# ANALISIS BARU: PERIODICITY
# =========================
def analyze_periodicity(df, draw_cols, lookback=20):
    periodicity = defaultdict(dict)
    total_numbers = int(df[draw_cols].max().max())
    
    for num in range(1, total_numbers + 1):
        appearances = []
        start_idx = max(0, len(df) - lookback)
        
        for i in range(len(df)-1, start_idx-1, -1):
            row = df.iloc[i]
            if num in row[draw_cols].values:
                appearances.append(i)
        
        if appearances:
            gaps = []
            for j in range(len(appearances)-1):
                gap = appearances[j] - appearances[j+1]
                gaps.append(gap)
            
            avg_gap = np.mean(gaps) if gaps else lookback
            periodicity[num] = {
                'count': len(appearances),
                'avg_gap': avg_gap,
                'last_seen': len(df) - appearances[0] - 1 if appearances else lookback,
                'due_score': (lookback - (len(df) - appearances[0] - 1)) / avg_gap if avg_gap > 0 else 0
            }
        else:
            periodicity[num] = {
                'count': 0,
                'avg_gap': lookback,
                'last_seen': lookback,
                'due_score': 1.0
            }
    
    print("\n🔄 PERIODICITY ANALYSIS (Last 20 Draws)")
    print("=" * 50)
    
    sorted_by_freq = sorted(periodicity.items(), key=lambda x: x[1]['count'], reverse=True)
    print("\n🔥 TOP 10 HOT NUMBERS (Most Frequent):")
    for num, stats in sorted_by_freq[:10]:
        print(f"#{num:2d}: {stats['count']:2d} times, avg gap: {stats['avg_gap']:4.1f}, last seen: {stats['last_seen']:2d} draws ago")
    
    print("\n❄️  TOP 10 COLD NUMBERS (Least Frequent):")
    for num, stats in sorted_by_freq[-10:]:
        print(f"#{num:2d}: {stats['count']:2d} times, avg gap: {stats['avg_gap']:4.1f}, last seen: {stats['last_seen']:2d} draws ago")
    
    print("\n⏰ TOP 10 DUE NUMBERS (Based on average gap):")
    sorted_by_due = sorted(periodicity.items(), key=lambda x: x[1]['due_score'], reverse=True)[:10]
    for num, stats in sorted_by_due:
        print(f"#{num:2d}: due score: {stats['due_score']:.2f}, last seen: {stats['last_seen']:2d} draws ago (avg gap: {stats['avg_gap']:.1f})")
    
    return periodicity

# =========================
# ANALISIS BARU: PATTERNS
# =========================
def analyze_patterns(df, draw_cols):
    patterns = []
    max_num = int(df[draw_cols].max().max())
    median = max_num // 2
    
    for _, row in df.iterrows():
        numbers = row[draw_cols].values
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = 5 - odd_count
        
        high_count = sum(1 for n in numbers if n > median)
        low_count = 5 - high_count
        
        patterns.append({
            'odd_even': (odd_count, even_count),
            'high_low': (high_count, low_count)
        })
    
    recent_count = min(50, len(patterns))
    recent_patterns = patterns[-recent_count:]
    
    odd_even_counts = Counter([p['odd_even'] for p in recent_patterns])
    high_low_counts = Counter([p['high_low'] for p in recent_patterns])
    
    print("\n🔢 PATTERN ANALYSIS (Last {} Draws)".format(recent_count))
    print("=" * 50)
    print("Odd/Even Distribution:")
    for (odd, even), count in odd_even_counts.most_common():
        print(f"{odd} Odd, {even} Even: {count:2d} times ({count/len(recent_patterns)*100:5.1f}%)")
    
    print("\nHigh/Low Distribution (Median = {}):".format(median))
    for (high, low), count in high_low_counts.most_common():
        print(f"{high} High, {low} Low: {count:2d} times ({count/len(recent_patterns)*100:5.1f}%)")
    
    return patterns

# =========================
# MONTE CARLO ENHANCED
# =========================
def enhanced_monte_carlo(history_sets, predicted_set, n_simulations=20000):
    match_counts = []
    pattern_diffs = []
    
    for _ in range(n_simulations):
        draw = list(history_sets[np.random.randint(len(history_sets))])
        
        matches = len(set(draw) & set(predicted_set))
        match_counts.append(matches)
        
        pred_odd = sum(1 for n in predicted_set if n % 2 == 1)
        draw_odd = sum(1 for n in draw if n % 2 == 1)
        pattern_diff = abs(pred_odd - draw_odd)
        pattern_diffs.append(pattern_diff)
    
    avg_matches = np.mean(match_counts)
    std_matches = np.std(match_counts)
    prob_0 = sum(1 for m in match_counts if m == 0) / n_simulations * 100
    prob_1 = sum(1 for m in match_counts if m == 1) / n_simulations * 100
    prob_2 = sum(1 for m in match_counts if m == 2) / n_simulations * 100
    prob_3 = sum(1 for m in match_counts if m == 3) / n_simulations * 100
    prob_4 = sum(1 for m in match_counts if m == 4) / n_simulations * 100
    prob_5 = sum(1 for m in match_counts if m == 5) / n_simulations * 100
    prob_3plus = prob_3 + prob_4 + prob_5
    
    return {
        'match_counts': match_counts,
        'avg_matches': avg_matches,
        'std_matches': std_matches,
        'prob_0': prob_0,
        'prob_1': prob_1,
        'prob_2': prob_2,
        'prob_3': prob_3,
        'prob_4': prob_4,
        'prob_5': prob_5,
        'prob_3plus': prob_3plus,
        'pattern_diffs': pattern_diffs
    }

# =========================
# COMPOSITE SCORING SYSTEM
# =========================
def composite_scoring(sets, freq_all, periodicity=None):
    scored_sets = []
    max_freq = max(freq_all.values()) if freq_all else 1
    
    for s in sets:
        numbers = s['numbers']
        
        freq_score = sum(freq_all[n] for n in numbers) / 5
        norm_freq_score = freq_score / max_freq
        
        zones = [((n - 1) // 10) + 1 for n in numbers]
        zone_counts = Counter(zones)
        zone_variance = np.var(list(zone_counts.values())) if len(zone_counts) > 1 else 0
        zone_score = 1 / (1 + zone_variance)
        
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        pattern_score = 1 - abs(odd_count - 2.5) / 2.5
        
        period_score = 0
        if periodicity:
            due_scores = [periodicity.get(n, {}).get('due_score', 0) for n in numbers]
            period_score = np.mean(due_scores) if due_scores else 0
        
        composite = (
            0.35 * norm_freq_score +
            0.25 * zone_score +
            0.20 * pattern_score +
            0.20 * period_score
        ) * 100
        
        scored_sets.append({
            'numbers': numbers,
            'composite_score': composite,
            'freq_score': norm_freq_score * 100,
            'zone_score': zone_score * 100,
            'pattern_score': pattern_score * 100,
            'period_score': period_score * 100,
            'odd_count': odd_count,
            'zone_distribution': sorted(zones),
            'original_data': s
        })
    
    return sorted(scored_sets, key=lambda x: x['composite_score'], reverse=True)

# =========================
# MARKOV CHAIN MONTE CARLO (MCMC) SAMPLING
# =========================
def mcmc_sample_sets(initial_set, allowed_numbers, freq_all, periodicity, 
                     steps=MCMC_STEPS, burnin=MCMC_BURNIN, n_sets=MCMC_SETS):
    """
    Jalankan Metropolis-Hastings MCMC untuk menghasilkan set angka dengan skor komposit tinggi.
    Proposal: ganti satu angka secara acak dari allowed_numbers, pastikan tidak ada duplikat.
    Target distribution: exp(composite_score / temperature)  (temperature = 10 agar eksploratif)
    """
    temperature = 10.0
    current = sorted(initial_set)
    # Fungsi skor internal untuk satu set
    def score_set(nums):
        # Gunakan logika composite_scoring tapi untuk satu set saja
        freq_score = sum(freq_all[n] for n in nums) / 5
        max_freq = max(freq_all.values()) if freq_all else 1
        norm_freq_score = freq_score / max_freq
        
        zones = [((n - 1) // 10) + 1 for n in nums]
        zone_counts = Counter(zones)
        zone_variance = np.var(list(zone_counts.values())) if len(zone_counts) > 1 else 0
        zone_score = 1 / (1 + zone_variance)
        
        odd_count = sum(1 for n in nums if n % 2 == 1)
        pattern_score = 1 - abs(odd_count - 2.5) / 2.5
        
        period_score = 0
        if periodicity:
            due_scores = [periodicity.get(n, {}).get('due_score', 0) for n in nums]
            period_score = np.mean(due_scores) if due_scores else 0
        
        composite = (0.35 * norm_freq_score + 0.25 * zone_score +
                     0.20 * pattern_score + 0.20 * period_score) * 100
        return composite
    
    current_score = score_set(current)
    samples = []
    accepted = 0
    
    for step in range(steps):
        # Proposal: pilih satu posisi acak dan ganti dengan angka baru dari allowed_numbers
        idx = np.random.randint(0, 5)
        candidate_num = np.random.choice(list(allowed_numbers))
        # Pastikan tidak duplikat
        while candidate_num in current:
            candidate_num = np.random.choice(list(allowed_numbers))
        
        candidate = current.copy()
        candidate[idx] = candidate_num
        candidate.sort()
        
        candidate_score = score_set(candidate)
        
        # Metropolis acceptance ratio
        delta = (candidate_score - current_score) / temperature
        accept_prob = min(1.0, np.exp(delta))
        
        if np.random.rand() < accept_prob:
            current = candidate
            current_score = candidate_score
            accepted += 1
        
        if step >= burnin:
            samples.append(tuple(current))
    
    # Ambil n_sets unik dengan skor tertinggi dari sampel
    unique_sets = {}
    for s in samples:
        if s not in unique_sets:
            unique_sets[s] = score_set(list(s))
    
    # Urutkan dan ambil n_sets terbaik
    sorted_unique = sorted(unique_sets.items(), key=lambda x: x[1], reverse=True)[:n_sets]
    mcmc_sets = []
    for nums_tuple, sc in sorted_unique:
        mcmc_sets.append({
            'numbers': list(nums_tuple),
            'confidence': np.mean([freq_all[n] for n in nums_tuple])
        })
    
    print(f"\n🔗 MCMC Sampling selesai: acceptance rate = {accepted/steps:.2%}")
    print(f"   Diperoleh {len(mcmc_sets)} set unik dari {len(samples)} sampel pasca-burnin.")
    return mcmc_sets

# =========================
# LOAD DATA (dengan argparse)
# =========================
parser = argparse.ArgumentParser(description='Enhanced Lottery Prediction with MCMC')
parser.add_argument('-f', '--file', required=True, help='CSV file path')
parser.add_argument('-t', '--top', type=int, default=TOP_N, help=f'Top N numbers per position (default: {TOP_N})')
parser.add_argument('-p', '--predict', type=int, default=PREDICT_SETS, help=f'Number of prediction sets to generate (default: {PREDICT_SETS})')
parser.add_argument('-m', '--monte', type=int, default=MONTE_CARLO_N, help=f'Monte Carlo simulations per set (default: {MONTE_CARLO_N})')
args = parser.parse_args()

# Override config
TOP_N = args.top
PREDICT_SETS = args.predict
MONTE_CARLO_N = args.monte

df = pd.read_csv(args.file)
df["DrawDate"] = pd.to_datetime(df["DrawDate"], format="%Y%m%d")

game_type = detect_game(df)
print(f"\n🎯 GAME DETECTED: {game_type}")
print(f"📊 Total Draws: {len(df)}")
print(f"📅 Date Range: {df['DrawDate'].min().date()} to {df['DrawDate'].max().date()}\n")

# =========================
# BASIC FREQUENCY ANALYSIS
# =========================
recent_df = df.copy()
all_numbers = recent_df[DRAW_COLS].stack()
freq_all = Counter(all_numbers)
allowed_numbers = set(freq_all.keys())

print(f"📈 Unique numbers in analysis: {len(allowed_numbers)}")

# =========================
# RUN ALL ANALYSES
# =========================
print("\n" + "="*60)
print("COMPREHENSIVE ANALYSIS REPORT")
print("="*60)

print("\n📊 ASCII HISTOGRAM (GLOBAL FREQUENCY)")
print("="*50)
ascii_histogram(freq_all)

zone_hits = analyze_zones(df, DRAW_COLS)
periodicity = analyze_periodicity(df, DRAW_COLS, PERIODICITY_LOOKBACK)
patterns = analyze_patterns(df, DRAW_COLS)

# =========================
# TOP N PER COLUMN
# =========================
print("\n🔝 TOP {} NUMBERS PER POSITION".format(TOP_N))
print("="*50)

top6_per_col = {}
for col in DRAW_COLS:
    time.sleep(1)  # dikurangi untuk kecepatan
    cnt = Counter(recent_df[col])
    top6 = [(n, c) for n, c in cnt.most_common(20)][:TOP_N]
    top6_per_col[col] = top6
    print(f"\nTop {TOP_N} often numbers in {col}:")
    for n, c in top6:
        last_seen = periodicity.get(n, {}).get('last_seen', 'N/A')
        print(f"  #{n:2d}: {c:3d} hits, last seen: {last_seen:2} draws ago")

# =========================
# GENERATE PREDICTION SETS (metode asli)
# =========================
print("\n" + "="*60)
print("GENERATING INITIAL PREDICTION SETS")
print("="*60)

sets = []

for set_num in range(PREDICT_SETS):
    used = set()
    current = []
    confidence = []
    
    for col in DRAW_COLS:
        pool = [(n, c) for n, c in top6_per_col[col] if n not in used]
        if not pool:
            cnt = Counter(recent_df[col])
            pool = [(n, c) for n, c in cnt.most_common(50) if n not in used][:TOP_N]
        
        nums = [n for n, _ in pool]
        weights = np.array([c for _, c in pool], dtype=float)
        weights_sum = weights.sum()
        
        if weights_sum > 0:
            weights /= weights_sum
            pick = int(np.random.choice(nums, p=weights))
        else:
            pick = np.random.choice(nums)
        
        used.add(pick)
        current.append(pick)
        confidence.append(freq_all[pick])
    
    sets.append({
        "numbers": sorted(current),
        "confidence": np.mean(confidence)
    })

# =========================
# MCMC: Generate additional sets from best initial set as starting point
# =========================
print("\n" + "="*60)
print("RUNNING MCMC SAMPLING")
print("="*60)

# Gunakan set dengan confidence tertinggi sebagai titik awal MCMC
best_initial = max(sets, key=lambda x: x['confidence'])
mcmc_sets = mcmc_sample_sets(best_initial['numbers'], allowed_numbers, freq_all, periodicity,
                             steps=MCMC_STEPS, burnin=MCMC_BURNIN, n_sets=min(5, PREDICT_SETS))

# Gabungkan dengan sets awal (hanya ambil beberapa terbaik dari MCMC)
all_candidate_sets = sets + mcmc_sets

# =========================
# ENHANCED MONTE CARLO SIMULATION (untuk semua candidate sets)
# =========================
print("\n🎲 RUNNING ENHANCED MONTE CARLO SIMULATIONS...")
history_sets = df[DRAW_COLS].values.tolist()

for s in all_candidate_sets:
    result = enhanced_monte_carlo(history_sets, s['numbers'], MONTE_CARLO_N)
    s['monte'] = Counter(result['match_counts'])
    s['variance'] = variance_score(result['match_counts'])
    s['max_match'] = max(result['match_counts'])
    s['avg_matches'] = result['avg_matches']
    s['std_matches'] = result['std_matches']
    s['prob_3plus'] = result['prob_3plus']
    s['prob_0'] = result['prob_0']
    s['prob_1'] = result['prob_1']
    s['prob_2'] = result['prob_2']
    s['prob_3'] = result['prob_3']
    s['prob_4'] = result['prob_4']
    s['prob_5'] = result['prob_5']

# =========================
# COMPOSITE SCORING (semua set)
# =========================
scored_sets = composite_scoring(all_candidate_sets, freq_all, periodicity)

# =========================
# FINAL OUTPUT (menampilkan semua set, termasuk MCMC)
# =========================
print("\n" + "="*60)
print("🏆 ENHANCED PREDICTION ANALYSIS (with MCMC)")
print("="*60)

for i, s in enumerate(scored_sets[:PREDICT_SETS], 1):
    time.sleep(2)
    print(f"\n{'='*60}")
    print(f"SET {i} (Composite Score: {s['composite_score']:.1f}/100)")
    print(f"{'='*60}")
    
    print(f"🔢 Numbers: {[int(x) for x in s['numbers']]}")
    print(f"📊 Score Breakdown:")
    print(f"   Frequency: {s['freq_score']:.1f}/100")
    print(f"   Zone Dist: {s['zone_score']:.1f}/100 (Zones: {[int(x) for x in s['zone_distribution']]})")
    print(f"   Pattern:   {s['pattern_score']:.1f}/100 ({s['odd_count']} odd, {5-s['odd_count']} even)")
    print(f"   Periodicity: {s['period_score']:.1f}/100")
    
    orig = s['original_data']
    print(f"\n🎲 Monte Carlo Simulation ({MONTE_CARLO_N:,} runs):")
    print(f"   Average matches: {orig['avg_matches']:.2f} ± {orig['std_matches']:.2f}")
    print(f"   Variance: {orig['variance']:.4f}")
    print(f"   Max possible match: {orig['max_match']}/5")
    
    print(f"\n📈 Match Probabilities:")
    print(f"   0/5: {orig['prob_0']:5.1f}%")
    print(f"   1/5: {orig['prob_1']:5.1f}%")
    print(f"   2/5: {orig['prob_2']:5.1f}%")
    print(f"   3/5: {orig['prob_3']:5.1f}%")
    print(f"   4/5: {orig['prob_4']:5.1f}%")
    print(f"   5/5: {orig['prob_5']:5.1f}%")
    print(f"   ⭐ 3+ matches: {orig['prob_3plus']:5.1f}%")

# =========================
# FINAL RECOMMENDATIONS
# =========================
print("\n" + "="*60)
print(f"🥇 ALL {PREDICT_SETS} FINAL RECOMMENDATIONS \n")
for i, s in enumerate(scored_sets[:PREDICT_SETS], 1):
    # Konversi ke int biasa untuk tampilan bersih
    clean_numbers = [int(x) for x in s['numbers']]
    print(f"🔢 {clean_numbers}")
print("")
print("="*60)

print("\n🏆 TOP 3 RECOMMENDED SETS:")
for i, s in enumerate(scored_sets[:3], 1):
    clean_numbers = [int(x) for x in s['numbers']]
    print(f"\n{i}. {clean_numbers}")
    print(f"   Score: {s['composite_score']:.1f}/100")
    print(f"   3+ match probability: {s['original_data']['prob_3plus']:.1f}%")

print("\n💡 PATTERN RECOMMENDATIONS:")
print("   • Target 2-3 odd numbers per set")
print("   • Spread numbers across 3-4 different zones")
print("   • Include 1-2 'due' numbers (high period_score)")
print("   • Balance between frequent and due numbers")
print("   • MCMC refined sets included for better exploration")

print("\n" + "="*60)
print("📊 ANALYSIS COMPLETE")
print("="*60)
print(f"Total prediction sets generated: {PREDICT_SETS}")
print(f"Monte Carlo simulations per set: {MONTE_CARLO_N:,}")
print(f"MCMC steps performed: {MCMC_STEPS} (burn-in: {MCMC_BURNIN})")
print("="*60)
