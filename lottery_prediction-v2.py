#!/usr/bin/env python3
# github.com/rouze-d

import sys
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import math
import time

# =========================
# CONFIG
# =========================
TOP_N = 13
PREDICT_SETS = 20
MONTE_CARLO_N = 1000000

# Parameter analisis baru
HOT_THRESHOLD = 0.7  # 70% percentile ke atas = HOT
COLD_THRESHOLD = 0.3 # 30% percentile ke bawah = COLD
PERIODICITY_LOOKBACK = 20  # Bilangan draw untuk analisis periodik

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
    """Analyze number zones (1-10, 11-20, etc.)"""
    zones = defaultdict(list)
    zone_hits = defaultdict(int)
    
    for _, row in df.iterrows():
        for col in draw_cols:
            num = row[col]
            zone = ((num - 1) // 10) + 1  # Zone 1: 1-10, Zone 2: 11-20, etc.
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
    """Analyze how often numbers reappear"""
    periodicity = defaultdict(dict)
    total_numbers = int(df[draw_cols].max().max())
    
    # Check last N draws for each number
    for num in range(1, total_numbers + 1):
        appearances = []
        start_idx = max(0, len(df) - lookback)
        
        for i in range(len(df)-1, start_idx-1, -1):
            row = df.iloc[i]
            if num in row[draw_cols].values:
                appearances.append(i)
        
        if appearances:
            # Calculate average gaps between appearances
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
                'due_score': 1.0  # Due to appear
            }
    
    print("\n🔄 PERIODICITY ANALYSIS (Last 20 Draws)")
    print("=" * 50)
    
    # Sort by frequency
    sorted_by_freq = sorted(periodicity.items(), key=lambda x: x[1]['count'], reverse=True)
    print("\n🔥 TOP 10 HOT NUMBERS (Most Frequent):")
    for num, stats in sorted_by_freq[:10]:
        print(f"#{num:2d}: {stats['count']:2d} times, avg gap: {stats['avg_gap']:4.1f}, last seen: {stats['last_seen']:2d} draws ago")
    
    print("\n❄️  TOP 10 COLD NUMBERS (Least Frequent):")
    for num, stats in sorted_by_freq[-10:]:
        print(f"#{num:2d}: {stats['count']:2d} times, avg gap: {stats['avg_gap']:4.1f}, last seen: {stats['last_seen']:2d} draws ago")
    
    # Numbers due to appear
    print("\n⏰ TOP 10 DUE NUMBERS (Based on average gap):")
    sorted_by_due = sorted(periodicity.items(), key=lambda x: x[1]['due_score'], reverse=True)[:10]
    for num, stats in sorted_by_due:
        print(f"#{num:2d}: due score: {stats['due_score']:.2f}, last seen: {stats['last_seen']:2d} draws ago (avg gap: {stats['avg_gap']:.1f})")
    
    return periodicity

# =========================
# ANALISIS BARU: PATTERNS
# =========================
def analyze_patterns(df, draw_cols):
    """Analyze odd/even and high/low patterns"""
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
    
    # Analyze last 50 draws
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
    """Enhanced Monte Carlo simulation with pattern matching"""
    match_counts = []
    pattern_diffs = []
    
    for _ in range(n_simulations):
        # Pick random historical draw
        draw = list(history_sets[np.random.randint(len(history_sets))])
        
        # Basic match count
        matches = len(set(draw) & set(predicted_set))
        match_counts.append(matches)
        
        # Pattern analysis (odd/even ratio)
        pred_odd = sum(1 for n in predicted_set if n % 2 == 1)
        draw_odd = sum(1 for n in draw if n % 2 == 1)
        pattern_diff = abs(pred_odd - draw_odd)
        pattern_diffs.append(pattern_diff)
    
    # Calculate additional statistics
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
    """Calculate composite score for each set"""
    scored_sets = []
    max_freq = max(freq_all.values()) if freq_all else 1
    
    for s in sets:
        numbers = s['numbers']
        
        # 1. Frequency Score (35%)
        freq_score = sum(freq_all[n] for n in numbers) / 5
        norm_freq_score = freq_score / max_freq
        
        # 2. Zone Distribution Score (25%)
        zones = [((n - 1) // 10) + 1 for n in numbers]
        zone_counts = Counter(zones)
        zone_variance = np.var(list(zone_counts.values())) if len(zone_counts) > 1 else 0
        zone_score = 1 / (1 + zone_variance)  # Higher if zones are balanced
        
        # 3. Pattern Score (20%)
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        pattern_score = 1 - abs(odd_count - 2.5) / 2.5  # Ideal: 2-3 odd numbers
        
        # 4. Periodicity Score (20%) - if available
        period_score = 0
        if periodicity:
            due_scores = [periodicity.get(n, {}).get('due_score', 0) for n in numbers]
            period_score = np.mean(due_scores) if due_scores else 0
        
        # Composite Score (weighted)
        composite = (
            0.35 * norm_freq_score +
            0.25 * zone_score +
            0.20 * pattern_score +
            0.20 * period_score
        ) * 100  # Scale to 0-100
        
        scored_sets.append({
            'numbers': numbers,
            'composite_score': composite,
            'freq_score': norm_freq_score * 100,
            'zone_score': zone_score * 100,
            'pattern_score': pattern_score * 100,
            'period_score': period_score * 100,
            'odd_count': odd_count,
            'zone_distribution': sorted(zones),
            'original_data': s  # Keep original data
        })
    
    return sorted(scored_sets, key=lambda x: x['composite_score'], reverse=True)

# =========================
# LOAD DATA
# =========================
if len(sys.argv) < 2:
    print("Usage: python3 analyze_draw.py data.csv")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
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

# ASCII Histogram
print("\n📊 ASCII HISTOGRAM (GLOBAL FREQUENCY)")
print("="*50)
ascii_histogram(freq_all)

# Run all new analyses
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
    time.sleep(3)
    cnt = Counter(recent_df[col])
    top6 = [(n, c) for n, c in cnt.most_common(20)][:TOP_N]
    top6_per_col[col] = top6
    print(f"\nTop {TOP_N} often numbers in {col}:")
    for n, c in top6:
        last_seen = periodicity.get(n, {}).get('last_seen', 'N/A')
        print(f"  #{n:2d}: {c:3d} hits, last seen: {last_seen:2} draws ago")

# =========================
# GENERATE PREDICTION SETS
# =========================
print("\n" + "="*60)
print("GENERATING PREDICTION SETS")
print("="*60)

sets = []

for set_num in range(PREDICT_SETS):
    used = set()
    current = []
    confidence = []
    
    for col in DRAW_COLS:
        pool = [(n, c) for n, c in top6_per_col[col] if n not in used]
        if not pool:  # If all numbers used, expand pool
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
# ENHANCED MONTE CARLO SIMULATION
# =========================
print("\n🎲 RUNNING ENHANCED MONTE CARLO SIMULATIONS...")
history_sets = df[DRAW_COLS].values.tolist()

for s in sets:
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
# COMPOSITE SCORING
# =========================
scored_sets = composite_scoring(sets, freq_all, periodicity)

# =========================
# FINAL OUTPUT
# =========================

print("\n" + "="*60)
print("🏆 ENHANCED PREDICTION ANALYSIS")
print("="*60)

for i, s in enumerate(scored_sets, 1):
    time.sleep(3)
    print(f"\n{'='*60}")
    print(f"SET {i} (Composite Score: {s['composite_score']:.1f}/100)")
    print(f"{'='*60}")
    
    print(f"🔢 Numbers: {s['numbers']}")
    print(f"📊 Score Breakdown:")
    print(f"   Frequency: {s['freq_score']:.1f}/100")
    print(f"   Zone Dist: {s['zone_score']:.1f}/100 (Zones: {s['zone_distribution']})")
    print(f"   Pattern:   {s['pattern_score']:.1f}/100 ({s['odd_count']} odd, {5-s['odd_count']} even)")
    print(f"   Periodicity: {s['period_score']:.1f}/100")
    
    # Original Monte Carlo results
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
for i, s in enumerate(scored_sets, 1):
    print(f"🔢 {s['numbers']}")
print("")
print("="*60)

print("\n🏆 TOP 3 RECOMMENDED SETS:")
for i, s in enumerate(scored_sets[:3], 1):
    print(f"\n{i}. {s['numbers']}")
    print(f"   Score: {s['composite_score']:.1f}/100")
    print(f"   3+ match probability: {s['original_data']['prob_3plus']:.1f}%")

print("\n💡 PATTERN RECOMMENDATIONS:")
print("   • Target 2-3 odd numbers per set")
print("   • Spread numbers across 3-4 different zones")
print("   • Include 1-2 'due' numbers (high period_score)")
print("   • Balance between frequent and due numbers")

print("\n" + "="*60)
print("📊 ANALYSIS COMPLETE")
print("="*60)
print(f"Total prediction sets generated: {PREDICT_SETS}")
print(f"Monte Carlo simulations per set: {MONTE_CARLO_N:,}")
print("="*60)
