#!/usr/bin/env python3
# github.com/rouze-d

import pandas as pd
import numpy as np
import requests
import os
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import random
import itertools
import argparse
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

# Optional TensorFlow (LSTM). Wrapped in try/except.
#try:
#    import tensorflow as tf
#    from tensorflow.keras.models import Sequential
#    from tensorflow.keras.layers import LSTM, Dense
#    TF_AVAILABLE = True
#except Exception:
#    TF_AVAILABLE = False

# =============================
# Game Configuration
# =============================
class GameConfig:
    """Configuration for different lottery games"""
    
    GAMES = {
        "1": {
            "name": "Supreme 6/58",
            "max_num": 58,
            "k": 6
        },
        "2": {
            "name": "Power 6/55",
            "max_num": 55,
            "k": 6
        },
        "3": {
            "name": "Star 6/50",
            "max_num": 50,
            "k": 6
        }
    }
    
    @classmethod
    def get_game(cls, choice):
        """Get game configuration based on choice"""
        return cls.GAMES.get(choice)
    
    @classmethod
    def display_menu(cls):
        """Display game selection menu"""
        print("\n" + "="*60)
        print("🎰 SELECT LOTTERY GAME")
        print("="*60)
        for key, game in cls.GAMES.items():
            print(f"  {key}.  📊 {game['name']}")
        print("="*60)
    
    @classmethod
    def get_user_choice(cls):
        """Get user choice for game"""
        while True:
            choice = input("\nSelect game (1-3): ").strip()
            if choice in cls.GAMES:
                return choice
            print("❌ Invalid choice. Please select 1, 2, or 3.")

# =============================
# Configuration
# =============================
class Config:
    """Configuration loaded based on game selection"""
    def __init__(self, game_choice):
        game = GameConfig.get_game(game_choice)
        self.GAME_NAME = game["name"]
        self.N_NUMBERS = game["max_num"]
        self.K = game["k"]
        self.DATE_COL = "DrawDate"
        self.MIN_DRAWS = 50
        print(f"\n🎯 Selected: {self.GAME_NAME}")
        print(f"📊 Numbers: 1-{self.N_NUMBERS}")
        print(f"🎰 Pick: {self.K} numbers")

# =============================
# Data Loading and Processing
# =============================
def load_data(file_path):
    if file_path:
        print(f"📂 Loading data from file: {file_path}")
        try:
            # Read the file
            df = pd.read_csv(file_path)
            print(f"✅ Successfully loaded {len(df)} rows")
            
            # Show column names for debugging
            print(f"📋 Columns found: {list(df.columns)}")
            print(f"...")
            print(f"📋 First few rows:")
            print(df.head())
            print(f"...")
            print(f"📋 End few rows:")
            print(df.tail())
            print(f"...")

            
            return df
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ Successfully loaded with {encoding} encoding")
                print(f"📊 Total rows: {len(df):,}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"⚠️ Error with {encoding}: {e}")
                continue
        
        # If all encodings fail, try without specifying encoding
        try:
            df = pd.read_csv(file_path)
            print("✅ Loaded with default encoding")
            return df
        except Exception as e:
            print(f"❌ Failed to load file: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def validate_data(df, config):
    """Validate the loaded data"""
    print("\n🔍 Validating data...")
    
    # Check for required columns
    required_cols = ["DrawDate", "DrawnNo1", "DrawnNo2", "DrawnNo3", "DrawnNo4", "DrawnNo5", "DrawnNo6"]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    print("✅ All required columns present")

    
    # Check for valid numbers (1-N_NUMBERS) in DrawnNo columns
    #print("\n📊 Validating number ranges...")
    nan_counts = {}
    range_violations = {}
    
    for i in range(1, 7):
        col_name = f"DrawnNo{i}"
        # Convert to numeric, coerce errors
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # Count NaN values
        nan_counts[col_name] = df[col_name].isna().sum()
        
        # Count values outside valid range
        valid_mask = (df[col_name] >= 1) & (df[col_name] <= config.N_NUMBERS)
        range_violations[col_name] = (~valid_mask).sum()
    
    # Print summary
    #print("\n📊 NaN values per column:")
    total_nan = 0
    for col_name, count in nan_counts.items():
    #    print(f"  {col_name}: {count:,} NaN values")
        total_nan += count
    
    #print("\n📊 Values outside range per column:")
    total_out_of_range = 0
    for col_name, count in range_violations.items():
    #    print(f"  {col_name}: {count:,} values outside 1-{config.N_NUMBERS}")
        total_out_of_range += count
    
    # Convert date column
    print("📅 Processing dates...")
    time.sleep(6)
    df["DrawDate"] = pd.to_datetime(df["DrawDate"], errors='coerce')
    invalid_dates = df["DrawDate"].isna().sum()
    if invalid_dates > 0:
        print(f"⚠️ Found {invalid_dates:,} rows with invalid dates")
    
    # Calculate valid rows
    valid_rows = len(df) - total_nan - total_out_of_range
    validity_percentage = (valid_rows / len(df)) * 100 if len(df) > 0 else 0
    
    print(f"\n📊 Data Quality Summary:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Valid rows (no NaN, within range): {valid_rows:,} ({validity_percentage:.1f}%)")
    print(f"  Rows with issues: {len(df) - valid_rows:,}")
    
    if valid_rows < config.MIN_DRAWS:
        print(f"\n⚠️ Warning: Only {valid_rows:,} valid rows (minimum recommended: {config.MIN_DRAWS:,})")
    
    print("✅ Data validation complete")
    return True

def preprocess_draws(df, config):
    """Prepare draws list and frequency"""
    print("\n📊 Processing draws data...")
    
    # Get DrawnNo columns
    drawn_cols = ["DrawnNo1", "DrawnNo2", "DrawnNo3", "DrawnNo4", "DrawnNo5", "DrawnNo6"]
    
    print(f"Using columns: {drawn_cols}")
    print(f"Original rows: {len(df):,}")
    
    # Make a copy of the relevant columns
    draws_df = df[drawn_cols].copy()
    
    # Convert all columns to numeric, forcing errors to NaN
    for col in drawn_cols:
        draws_df[col] = pd.to_numeric(draws_df[col], errors='coerce')
    
    # Drop rows with ANY NaN values
    before_drop = len(draws_df)
    draws_df_clean = draws_df.dropna()
    after_drop = len(draws_df_clean)
    
    print(f"\n📊 Cleaning results:")
    print(f"  Rows before dropping NaN: {before_drop:,}")
    print(f"  Rows after dropping NaN: {after_drop:,}")
    print(f"  Rows dropped due to NaN: {before_drop - after_drop:,}")
    
    if len(draws_df_clean) == 0:
        print("❌ No valid draws after dropping NaN values")
        return [], pd.Series(dtype=float)
    
    # Convert to integers
    for col in drawn_cols:
        draws_df_clean[col] = draws_df_clean[col].astype(int)
    
    # Check if numbers are within valid range
    valid_mask = draws_df_clean.apply(lambda row: all(1 <= x <= config.N_NUMBERS for x in row), axis=1)
    draws_df_valid = draws_df_clean[valid_mask]
    
    invalid_rows_count = len(draws_df_clean) - len(draws_df_valid)
    print(f"\n📊 Range validation:")
    print(f"  Rows after NaN removal: {len(draws_df_clean):,}")
    print(f"  Rows with all values in 1-{config.N_NUMBERS} range: {len(draws_df_valid):,}")
    print(f"  Rows with values outside range: {invalid_rows_count:,}")
    
    if len(draws_df_valid) == 0:
        print(f"❌ No draws with all values in valid range (1-{config.N_NUMBERS})")
        return [], pd.Series(dtype=float)
    
    # Convert each row to a sorted tuple of ints
    print("\n🔄 Converting draws to tuples...")
    sorted_draws = []
    for idx, row in tqdm(draws_df_valid.iterrows(), total=len(draws_df_valid), desc="Processing draws"):
        try:
            numbers = [int(x) for x in row]
            sorted_draws.append(tuple(sorted(numbers)))
        except Exception as e:
            print(f"⚠️ Error processing row {idx}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(sorted_draws):,} draws")
    
    if len(sorted_draws) == 0:
        print("❌ No valid draws found")
        return [], pd.Series(dtype=float)
    
    # Calculate frequencies
    print("📈 Calculating frequencies...")
    all_numbers = []
    for draw in tqdm(sorted_draws, desc="Counting numbers"):
        all_numbers.extend(draw)
    
    counter = Counter(all_numbers)
    
    # Create frequency series
    freq = pd.Series([0] * config.N_NUMBERS, index=range(1, config.N_NUMBERS + 1))
    for num, count in counter.items():
        if 1 <= num <= config.N_NUMBERS:
            freq.loc[num] = count
    
    print(f"\n📊 Frequency calculation complete.")
    print(f"  Most frequent numbers: {freq.sort_values(ascending=False).head(5).index.tolist()}")
    print(f"  Least frequent numbers: {freq.sort_values(ascending=True).head(5).index.tolist()}")
    print(f"  Total number appearances: {sum(freq.values):,}")
    
    return sorted_draws, freq

# =============================
# Helper Functions
# =============================
def weighted_sample_no_replace(items, scores, k, temperature=1.0):
    """Sample without replacement from weighted scores"""
    items = list(items)
    scores = np.array(scores, dtype=float)
    out = []
    available = items.copy()
    sc = scores.copy()
    for _ in range(min(k, len(items))):
        exp = np.exp(sc / max(1e-6, temperature))
        probs = exp / exp.sum()
        choice = np.random.choice(len(available), p=probs)
        out.append(available.pop(choice))
        sc = np.delete(sc, choice)
    return out

def gen_hot_cold(freq, pick=6, method="hot", variation=0):
    """Generate hot, cold, or random-weighted numbers"""
    if method == "hot":
        topK = max(pick, pick + variation)
        top = list(map(int, freq.sort_values(ascending=False).index[:topK].tolist()))
        if variation == 0:
            return tuple(sorted(top[:pick]))
        else:
            s = sorted(random.sample(top, pick))
            return tuple(s)
    elif method == "cold":
        bottomK = max(pick, pick + variation)
        bottom = list(map(int, freq.sort_values(ascending=True).index[:bottomK].tolist()))
        if variation == 0:
            return tuple(sorted(bottom[:pick]))
        else:
            s = sorted(random.sample(bottom, pick))
            return tuple(s)
    elif method == "random-weighted":
        weights = freq.values.astype(float) + 1e-6
        nums = list(freq.index)
        chosen = np.random.choice(nums, size=pick, replace=False, p=weights/weights.sum())
        return tuple(sorted(int(x) for x in chosen))
    else:
        nums = list(freq.index)
        return tuple(sorted(int(x) for x in random.sample(nums, pick)))

def monte_carlo_suggest(freq, pick=6, n_sim=5000, top_k=5):
    """Monte Carlo simulation for suggestions"""
    nums = freq.index.tolist()
    probs = freq.values.astype(float)
    if probs.sum() == 0:
        probs = np.ones_like(probs)
    p = probs / probs.sum()
    counter = {}
    for _ in range(n_sim):
        s = tuple(sorted(np.random.choice(nums, size=pick, replace=False, p=p)))
        counter[s] = counter.get(s, 0) + 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]

# =============================
# ML Logistic Model
# =============================
def build_ml_model(draws_list, max_num):
    """Build ML Logistic Regression models"""
    rows = []
    for i in range(len(draws_list) - 1):
        cur = np.zeros(max_num, dtype=int)
        nxt = np.zeros(max_num, dtype=int)
        for n in draws_list[i]:
            cur[n-1] = 1
        for n in draws_list[i+1]:
            nxt[n-1] = 1
        rows.append((cur, nxt))
    if not rows:
        return None
    X = np.vstack([r[0] for r in rows])
    Y = np.vstack([r[1] for r in rows])
    models = {}
    for num_idx in range(max_num):
        y = Y[:, num_idx]
        if y.sum() < 3:
            models[num_idx+1] = None
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        models[num_idx+1] = (clf, acc)
    return models

def ml_score_list(models, last_draw):
    """Get ML scores for all numbers"""
    if models is None:
        return []
    max_num = len(models)
    x = np.zeros(max_num)
    for n in last_draw:
        if n-1 < max_num:
            x[n-1] = 1
    scores = []
    for num in range(1, max_num+1):
        mdl = models.get(num)
        if mdl is None:
            scores.append((num, 0.0))
        else:
            clf, acc = mdl
            prob = clf.predict_proba(x.reshape(1,-1))[0][1]
            score = prob * (0.5 + 0.5*acc)
            scores.append((num, score))
    return scores

def ml_predict(models, last_draw, pick=6, variation=0, temperature=1.0):
    """Predict using ML Logistic model"""
    if models is None:
        return ()
    scores = ml_score_list(models, last_draw)
    nums = [s[0] for s in scores]
    vals = np.array([s[1] for s in scores], dtype=float)
    if variation == 0:
        chosen = [s[0] for s in sorted(scores, key=lambda x:x[1], reverse=True)[:pick]]
        return tuple(sorted(int(x) for x in chosen))
    else:
        chosen = weighted_sample_no_replace(nums, vals, pick, temperature=temperature)
        return tuple(sorted(int(x) for x in chosen))

# =============================
# Markov Chain Model
# =============================
def build_markov_transitions(draws_list, order=1):
    """Build Markov chain transitions"""
    transitions = defaultdict(Counter)
    for i in range(len(draws_list)-1):
        cur = draws_list[i]
        nxt = draws_list[i+1]
        for state in itertools.combinations(cur, order):
            state = tuple(sorted(state))
            for next_num in nxt:
                transitions[state][next_num] += 1
    return transitions

def markov_chain_predict_from_transitions(transitions, last_draw, max_num, pick=6, variation=0, temperature=1.0):
    """Predict using Markov chain transitions"""
    # Aggregate counts for next numbers from all states that are subset of last_draw
    score_acc = Counter()
    for state, counter in transitions.items():
        if set(state).issubset(set(last_draw)):
            for num, cnt in counter.items():
                score_acc[num] += cnt
    if not score_acc:
        # fallback to top aggregated next counts or random
        all_counts = Counter()
        for c in transitions.values():
            all_counts.update(c)
        if all_counts:
            top = [n for n, _ in all_counts.most_common(pick)]
            return tuple(sorted(int(x) for x in top))
        return tuple(sorted(random.sample(range(1, max_num+1), pick)))
    
    # Deterministic top pick
    items = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)
    nums = [n for n, v in items]
    vals = np.array([v for n, v in items], dtype=float)
    if variation == 0:
        chosen = nums[:pick]
        # fill if needed
        if len(chosen) < pick:
            for n in range(1, max_num+1):
                if n not in chosen:
                    chosen.append(n)
                if len(chosen) >= pick:
                    break
        return tuple(sorted(int(x) for x in chosen[:pick]))
    else:
        # stochastic sampling from the available scored numbers
        pool = nums.copy()
        pool_scores = vals.copy()
        # include remaining numbers with small epsilon score to allow sampling
        rest = [n for n in range(1, max_num+1) if n not in pool]
        if rest:
            pool += rest
            pool_scores = np.concatenate([pool_scores, np.ones(len(rest))*1e-6])
        sampled = weighted_sample_no_replace(pool, pool_scores, pick, temperature=temperature)
        return tuple(sorted(int(x) for x in sampled))

def markov_chain_predict(draws_list, max_num, order=1, pick=6, variation=0, temperature=1.0):
    """Predict using Markov chain"""
    if len(draws_list) < 2:
        return ()
    if order < 1:
        order = 1
    # if order > size of draw, fallback to 1
    last = draws_list[-1]
    if order > len(last):
        order = 1
    transitions = build_markov_transitions(draws_list, order=order)
    return markov_chain_predict_from_transitions(transitions, last, max_num, pick=pick, variation=variation, temperature=temperature)

# =============================
# LSTM Model
# =============================
#def build_lstm_data(draws_list, max_num, seq_len=10):
#    """Build LSTM training data"""
#    X, y = [], []
#    for i in range(len(draws_list) - seq_len):
#        seq = np.zeros((seq_len, max_num), dtype=float)
#        for j in range(seq_len):
#            for n in draws_list[i+j]:
#                seq[j, n-1] = 1.0
#        target = np.zeros(max_num, dtype=float)
#        for n in draws_list[i+seq_len]:
#            target[n-1] = 1.0
#        X.append(seq); y.append(target)
#    if not X:
#        return None, None
#    return np.array(X), np.array(y)

#def build_lstm_model(seq_len, max_num, latent=64):
#    """Build LSTM model"""
#    model = Sequential([
#        LSTM(latent, input_shape=(seq_len, max_num)),
#        Dense(max_num, activation="sigmoid")
#    ])
#    model.compile(optimizer="adam", loss="binary_crossentropy")
#    return model

#def lstm_predict(draws_list, max_num, variation=0, temperature=1.0):
#    """Predict using LSTM"""
#    if not TF_AVAILABLE:
#        print("⚠️ TensorFlow not available for LSTM")
#        return ()
    
#    X, y = build_lstm_data(draws_list, max_num, seq_len=10)
#    if X is None:
#        print("⚠️ Not enough data for LSTM")
#        return ()
    
#    model = build_lstm_model(X.shape[1], max_num, latent=64)
#    print("🤖 Training LSTM...")
#    model.fit(X, y, epochs=3, batch_size=16, verbose=0)
    
#    probs = model.predict(X[-1].reshape(1, X.shape[1], max_num))[0]
#    top_idx = np.argsort(probs)[::-1]
#    pool = (top_idx[:12] + 1).tolist()
    
#    if variation == 0:
#        chosen = sorted([int(x) for x in pool[:6]])
#    else:
#        chosen = sorted(np.random.choice(pool, size=6, replace=False).tolist())
#    
#    return tuple(chosen)

# =============================
# Hybrid (Freq+ML) Model
# =============================
def hybrid_freq_ml_predict(draws_list, freq, max_num, hybrid_ml_weight=0.5, variation=0, temperature=1.0):
    """Hybrid prediction combining frequency and ML"""
    models = build_ml_model(draws_list, max_num)
    hot_rank = list(freq.sort_values(ascending=False).index)  # most -> least
    
    if models:
        ml_scores_dict = dict(ml_score_list(models, draws_list[-1]))
        ml_vals = np.array([ml_scores_dict.get(n, 0.0) for n in hot_rank], dtype=float)
        if ml_vals.max() > 0:
            ml_norm = (ml_vals - ml_vals.min()) / (ml_vals.max() - ml_vals.min() + 1e-9)
        else:
            ml_norm = np.zeros_like(ml_vals)
        
        freq_vals = np.array([freq.loc[n] for n in hot_rank], dtype=float)
        if freq_vals.max() > 0:
            freq_norm = (freq_vals - freq_vals.min()) / (freq_vals.max() - freq_vals.min() + 1e-9)
        else:
            freq_norm = np.zeros_like(freq_vals)
        
        combined = hybrid_ml_weight * ml_norm + (1.0 - hybrid_ml_weight) * freq_norm
        
        pool = hot_rank[:20]
        pool_scores = combined[:20]
        
        if variation == 0:
            chosen = sorted([int(x) for x in pool[:6]])
        else:
            sampled = weighted_sample_no_replace(pool, pool_scores, 6, temperature=temperature)
            chosen = sorted(int(x) for x in sampled)
    else:
        # Fallback to hot numbers
        chosen = gen_hot_cold(freq, pick=6, method="hot", variation=variation)
    
    return tuple(chosen)

# =============================
# Original Markov Prediction Functions (for backward compatibility)
# =============================
def markov_prediction(numbers, method="pairwise"):
    """Markov chain prediction (original version)"""
    if method == "pairwise":
        transitions = defaultdict(Counter)
        
        print(f"🔄 Building Markov transitions ({method})...")
        for draw in tqdm(numbers, desc="Processing draws"):
            for i in range(len(draw)-1):
                transitions[draw[i]][draw[i+1]] += 1
        
        if not transitions:
            return random.sample(range(1, 55), 6)
        
        # Start with most common first number
        first_numbers = [draw[0] for draw in numbers]
        start_num = Counter(first_numbers).most_common(1)[0][0]
        
        result = [start_num]
        for _ in range(5):  # We already have 1 number, need 5 more
            next_states = transitions[result[-1]]
            if next_states:
                next_num = max(next_states, key=next_states.get)
                attempts = 0
                while next_num in result and attempts < 10:
                    next_states[next_num] = 0
                    if next_states:
                        next_num = max(next_states, key=next_states.get)
                    else:
                        next_num = random.randint(1, 55)
                    attempts += 1
            else:
                next_num = random.randint(1, 55)
                while next_num in result:
                    next_num = random.randint(1, 55)
            
            result.append(next_num)
        
        return sorted(result)
    
    else:  # Frequency method
        print(f"📊 Calculating frequencies ({method})...")
        flat_numbers = [num for draw in tqdm(numbers, desc="Flattening draws") for num in draw]
        counter = Counter(flat_numbers)
        
        # Get most common numbers
        most_common = [num for num, _ in counter.most_common(55)]
        
        result = []
        for num in tqdm(most_common, desc="Selecting numbers"):
            if num not in result:
                result.append(num)
            if len(result) == 6:
                break
        
        # Pad if needed
        while len(result) < 6:
            new_num = random.randint(1, 55)
            if new_num not in result:
                result.append(new_num)
        
        return sorted(result)

# =============================
# Prediction Models
# =============================
def prepare_features_targets(numbers):
    """Prepare features and targets for ML models"""
    print("🔄 Preparing features and targets...")
    
    X = []
    y = []
    
    # Create sliding window of sequences
    for i in tqdm(range(len(numbers) - 1), desc="Creating sequences"):
        # Features: current draw
        # Target: next draw (represented as index)
        X.append(numbers[i])
        y.append(i + 1)  # Next draw index
    
    if not X:
        # If we can't create sequences, use simple approach
        X = numbers[:-1]
        y = list(range(1, len(numbers)))
    
    return np.array(X), np.array(y)

def train_simple_models(X, y, config):
    """Train simple machine learning models with progress bar"""
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "NaiveBayes": GaussianNB()
    }
    
    preds = {}
    print("🤖 Training models...")
    
    for name, model in tqdm(models.items(), desc="Training models"):
        try:
            model.fit(X, y)
            
            # Prepare sample for prediction (most recent draw)
            if len(X) > 0:
                # Create a synthetic sample based on most recent draw
                sample = X[-1].reshape(1, -1)
                
                # Predict next number probabilities
                probs = np.zeros(config.N_NUMBERS)
                
                # Get all possible next number indices
                if hasattr(model, 'predict_proba'):
                    all_probs = model.predict_proba(sample)[0]
                    
                    # Map indices back to numbers
                    for idx, prob in enumerate(all_probs):
                        if idx < config.N_NUMBERS:
                            probs[idx] = prob
                
                preds[name] = probs
            else:
                preds[name] = np.zeros(config.N_NUMBERS)
                
        except Exception as e:
            print(f"⚠️ {name} model failed: {str(e)[:100]}")
            preds[name] = np.zeros(config.N_NUMBERS)
    
    return preds

def ensemble_predictions(preds_dict, config, strategy="vote"):
    """Ensemble predictions from multiple models"""
    if strategy == "vote":
        all_preds = []
        
        print("🗳️ Generating ensemble predictions (voting)...")
        for name, probs in tqdm(preds_dict.items(), desc="Collecting model votes"):
            # Get top K numbers with highest probabilities
            if len(probs) >= config.K:
                top_indices = np.argsort(probs)[-config.K:]
                all_preds.extend(top_indices + 1)  # Convert indices to numbers (1-N_NUMBERS)
        
        if not all_preds:
            return random.sample(range(1, config.N_NUMBERS + 1), config.K)
        
        counter = Counter(all_preds)
        final = [num for num, _ in counter.most_common(config.K)]
        
        while len(final) < config.K:
            new_num = random.randint(1, config.N_NUMBERS)
            if new_num not in final:
                final.append(new_num)
        
        return sorted(final)
    
    else:  # Probability weighted
        print("⚖️ Generating ensemble predictions (probability weighted)...")
        all_probs = []
        for probs in preds_dict.values():
            if len(probs) > 0:
                all_probs.append(probs)
        
        if not all_probs:
            return random.sample(range(1, config.N_NUMBERS + 1), config.K)
        
        avg_probs = np.mean(all_probs, axis=0)
        top_indices = np.argsort(avg_probs)[-config.K:]
        final = [int(idx) + 1 for idx in top_indices]
        return sorted(final)

def hybrid_ensemble_weighted(preds_dict, numbers, config, weights):
    """Hybrid ensemble with weighted voting"""
    print("🧬 Generating hybrid ensemble predictions...")
    weighted_counter = Counter()
    
    # ML Models
    if "RandomForest" in preds_dict:
        rf_pred = ensemble_predictions({"rf": preds_dict["RandomForest"]}, config, "vote")
        for num in tqdm(rf_pred, desc="Random Forest votes"):
            weighted_counter[num] += weights.get("Random Forest", 1.0)
    
    if "Logistic" in preds_dict:
        log_pred = ensemble_predictions({"log": preds_dict["Logistic"]}, config, "vote")
        for num in tqdm(log_pred, desc="Logistic Regression votes"):
            weighted_counter[num] += weights.get("Logistic", 1.0)
    
    if "NaiveBayes" in preds_dict:
        nb_pred = ensemble_predictions({"nb": preds_dict["NaiveBayes"]}, config, "vote")
        for num in tqdm(nb_pred, desc="Naive Bayes votes"):
            weighted_counter[num] += weights.get("Naive Bayes", 1.0)
    
    # Original Markov predictions
    mp_pred = markov_prediction(numbers, "pairwise")
    for num in tqdm(mp_pred, desc="Markov Pairwise votes"):
        weighted_counter[num] += weights.get("Markov (Pairwise)", 1.0)
    
    mf_pred = markov_prediction(numbers, "freq")
    for num in tqdm(mf_pred, desc="Markov Frequency votes"):
        weighted_counter[num] += weights.get("Markov (Frequency)", 1.0)
    
    # Get top k by weight
    final = []
    for num, _ in tqdm(weighted_counter.most_common(config.N_NUMBERS), desc="Selecting final numbers"):
        if num not in final and 1 <= num <= config.N_NUMBERS:
            final.append(num)
        if len(final) == config.K:
            break
    
    while len(final) < config.K:
        new_num = random.randint(1, config.N_NUMBERS)
        if new_num not in final:
            final.append(new_num)
    
    return sorted(final)

# =============================
# Statistics and Analysis
# =============================
def calculate_statistics(numbers, config):
    """Calculate number statistics"""
    print("📈 Calculating statistics...")
    
    all_numbers = [num for draw in tqdm(numbers, desc="Processing draws") for num in draw]
    counter = Counter(all_numbers)
    
    # Most frequent numbers
    most_common = counter.most_common(10)
    least_common = counter.most_common()[-10:]
    
    stats = {
        "most_common": most_common,
        "least_common": least_common,
        "total_draws": len(numbers),
        "unique_numbers": len(counter),
        "avg_frequency": len(all_numbers) / config.N_NUMBERS
    }
    
    return stats

def display_predictions(predictions, config, stats=None):
    """Display predictions in a formatted way"""
    print("\n" + "="*60)
    print(f"🎯 {config.GAME_NAME.upper()} PREDICTION RESULTS")
    print("="*60)
    
    for model_name, pred in predictions.items():
        if isinstance(pred, tuple):
            pred = list(pred)
        elif pred is None:
            pred = []
        pred_str = " ".join([f"{num:02d}" for num in pred])
        print(f"📋 {model_name:<30}: {pred_str}")
    
    print("\n" + "="*60)
    print("🌟 RECOMMENDED PREDICTION")
    print("="*60)
    if "Hybrid Ensemble (Weighted)" in predictions:
        recommended = predictions["Hybrid Ensemble (Weighted)"]
        if recommended:
            pred_str = "  ".join([f"{num:02d}" for num in recommended])
            print(f"\n🎲 Numbers: {pred_str}")
    print(f"📅 Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if stats:
        print("\n" + "="*60)
        print("📊 NUMBER STATISTICS")
        print("="*60)
        
        print("\n🏆 Top 10 Most Frequent Numbers:")
        for num, freq in stats["most_common"]:
            percentage = (freq / stats["total_draws"]) * 100
            print(f"  {num:02d}: {freq:3d} times ({percentage:5.1f}%)")
        
        print("\n📉 Top 10 Least Frequent Numbers:")
        for num, freq in stats["least_common"]:
            percentage = (freq / stats["total_draws"]) * 100
            print(f"  {num:02d}: {freq:3d} times ({percentage:5.1f}%)")
        
        print(f"\n📊 Total draws analyzed: {stats['total_draws']}")
        print(f"🎰 Unique numbers appeared: {stats['unique_numbers']}/{config.N_NUMBERS}")
        print(f"📊 Average frequency per number: {stats['avg_frequency']:.1f}")

# =============================
# Main Function
# =============================
def main():
    """Main function"""
    print("\n" + "="*60)
    print("🎰 LOTTERY PREDICTOR v2.0")
    print("="*60)
    
    # Display game selection menu
    GameConfig.display_menu()
    game_choice = GameConfig.get_user_choice()
    
    # Load configuration based on game choice
    config = Config(game_choice)
    
    # Get file path from user
    print("\n" + "="*60)
    print("📂 LOAD DATA (REQUIRED)")
    print("="*60)
    
    while True:
        file_path = input("\nPath to data file: ").strip()
        
        if not file_path:
            print("❌ Please enter a file path")
            continue
            
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
            
        break
    
    # Load data
    df = load_data(file_path)
    
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Validate data
    if not validate_data(df, config):
        print("❌ Data validation failed.")
        return
    
    # Prepare draws and frequency
    draws_list, freq = preprocess_draws(df, config)
    
    if not draws_list:
        print("❌ No draws available for analysis.")
        return
    
    print(f"\n📊 Loaded {len(draws_list):,} draws for analysis")
    print(f"🎰 Most recent draw: {list(map(int, draws_list[-1]))}")
    
    # Display frequency information
    print("\n📊 Frequency Analysis:")
    top_6 = list(freq.sort_values(ascending=False).head(6).index)
    bottom_6 = list(freq.sort_values(ascending=True).head(6).index)
    print(f"Hot numbers (most frequent): {top_6}")
    print(f"Cold numbers (least frequent): {bottom_6}")
    
    # Automatically run all methods (no choice selection)
    print("\n" + "="*60)
    print("🔮 RUNNING ALL PREDICTION METHODS")
    print("="*60)
    
    # Get variation level
    try:
        variation = int(input(f"Variation level (0=deterministic, 1-10=randomness, default 2): ").strip() or "2")
        if variation < 0 or variation > 10:
            print("⚠️ Setting to default (2)")
            variation = 2
    except:
        variation = 2
        print("⚠️ Using default (variation=2)")
    
    # Default model weights
    weights = {
        "Random Forest": 1.0,
        "Logistic": 1.0,
        "Naive Bayes": 1.0,
        "Markov (Pairwise)": 1.0,
        "Markov (Frequency)": 1.0
    }
    
    # Train original models for ensemble
    numbers_array = np.array([list(d) for d in draws_list])
    X, y = prepare_features_targets(numbers_array)
    
    if len(X) > 0 and len(y) > 0:
        preds_dict = train_simple_models(X, y, config)
    else:
        print("⚠️ Not enough data for ML models")
        preds_dict = {}
    
    # Generate predictions from ALL methods
    print("\n" + "="*60)
    print("🔮 GENERATING PREDICTIONS FROM ALL METHODS")
    print("="*60)
    
    raw_predictions = {}
    
    # Hot Numbers
    print("🔥 Generating Hot numbers...")
    hot_prediction = gen_hot_cold(freq, pick=config.K, method="hot", variation=variation)
    if hot_prediction:
        raw_predictions["Hot Numbers"] = hot_prediction
    
    # Cold Numbers
    print("❄️ Generating Cold numbers...")
    cold_prediction = gen_hot_cold(freq, pick=config.K, method="cold", variation=variation)
    if cold_prediction:
        raw_predictions["Cold Numbers"] = cold_prediction
    
    # Random Weighted
    print("🎲 Generating Random Weighted numbers...")
    random_weighted_prediction = gen_hot_cold(freq, pick=config.K, method="random-weighted", variation=variation)
    if random_weighted_prediction:
        raw_predictions["Random Weighted"] = random_weighted_prediction
    
    # Monte Carlo
    print("🎯 Generating Monte Carlo suggestions...")
    mc_results = monte_carlo_suggest(freq, pick=config.K, n_sim=5000, top_k=1)
    if mc_results and mc_results[0][0]:
        raw_predictions["Monte Carlo"] = mc_results[0][0]
    
    # ML Logistic
    print("🤖 Generating ML Logistic predictions...")
    ml_models = build_ml_model(draws_list, config.N_NUMBERS)
    if ml_models:
        ml_prediction = ml_predict(ml_models, draws_list[-1], pick=config.K, 
                                  variation=variation, temperature=1.0)
        if ml_prediction:
            raw_predictions["ML Logistic"] = ml_prediction
    else:
        print("⚠️ Could not build ML model")
    
    # Markov Chain
    print("🔄 Generating Markov Chain predictions...")
    markov_prediction_result = markov_chain_predict(draws_list, config.N_NUMBERS, order=1,
                                                    pick=config.K, variation=variation, 
                                                    temperature=1.0)
    if markov_prediction_result:
        raw_predictions["Markov Chain"] = markov_prediction_result
    
    # LSTM
    #if TF_AVAILABLE:
    #    print("🧠 Generating LSTM predictions...")
    #    lstm_prediction = lstm_predict(draws_list, config.N_NUMBERS, variation=variation, 
    #                                  temperature=1.0)
    #    if lstm_prediction:
    #        raw_predictions["LSTM"] = lstm_prediction
    #else:
    #    print("⚠️ TensorFlow not available for LSTM")
    
    # Hybrid (Freq+ML)
    print("🧬 Generating Hybrid (Freq+ML) predictions...")
    hybrid_prediction = hybrid_freq_ml_predict(draws_list, freq, config.N_NUMBERS, 
                                              hybrid_ml_weight=0.5,
                                              variation=variation, 
                                              temperature=1.0)
    if hybrid_prediction:
        raw_predictions["Hybrid (Freq+ML)"] = hybrid_prediction
    
    # Ensemble methods (if we have ML models)
    print("🤝 Generating Ensemble predictions...")
    if preds_dict:
        raw_predictions.update({
            "Random Forest": ensemble_predictions({"rf": preds_dict["RandomForest"]}, config, "vote"),
            "Logistic Regression": ensemble_predictions({"log": preds_dict["Logistic"]}, config, "vote"),
            "Naive Bayes": ensemble_predictions({"nb": preds_dict["NaiveBayes"]}, config, "vote"),
            "Ensemble (Voting)": ensemble_predictions(preds_dict, config, "vote"),
            "Ensemble (Prob Weighted)": ensemble_predictions(preds_dict, config, "prob"),
            "Hybrid Ensemble (Weighted)": hybrid_ensemble_weighted(preds_dict, numbers_array.tolist(), config, weights)
        })
    
    # Calculate statistics
    stats = calculate_statistics(draws_list, config)
    
    # Display results
    display_predictions(raw_predictions, config, stats)
    
    # Export to TXT file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Create export data
    game_name_short = config.GAME_NAME.replace(" ", "_").replace("/", "_")
    txt_filename = f"{game_name_short}_predictions_{timestamp}.txt"
    
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"{config.GAME_NAME.upper()} PREDICTION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total draws analyzed: {len(draws_list):,}\n")
        f.write(f"Prediction method: ALL METHODS\n")
        f.write(f"Variation level: {variation}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("PREDICTIONS\n")
        f.write("="*60 + "\n\n")
        
        for model_name, pred in raw_predictions.items():
            if isinstance(pred, tuple):
                pred = list(pred)
            elif pred is None:
                pred = []
            pred_str = " ".join([f"{num:02d}" for num in pred])
            f.write(f"{model_name:<30}: {pred_str}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Top 10 Most Frequent Numbers:\n")
        for num, freq in stats["most_common"]:
            percentage = (freq / stats["total_draws"]) * 100
            f.write(f"  {num:02d}: {freq:3d} times ({percentage:5.1f}%)\n")
        
        f.write("\nTop 10 Least Frequent Numbers:\n")
        for num, freq in stats["least_common"]:
            percentage = (freq / stats["total_draws"]) * 100
            f.write(f"  {num:02d}: {freq:3d} times ({percentage:5.1f}%)\n")
        
        f.write(f"\nTotal draws analyzed: {stats['total_draws']}\n")
        f.write(f"Unique numbers appeared: {stats['unique_numbers']}/{config.N_NUMBERS}\n")
        f.write(f"Average frequency per number: {stats['avg_frequency']:.1f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("DISCLAIMER\n")
        f.write("="*60 + "\n")
        f.write("For entertainment and analytical purposes only.\n")
        f.write("Lottery predictions are inherently uncertain.\n")
    
    print(f"\n💾 Predictions saved to: {txt_filename}")
    
    
    print("\n" + "="*60)
    print("✅ ALL PREDICTION METHODS COMPLETED")
    print("="*60)
    print("⚠️  DISCLAIMER: For entertainment and analytical purposes only.")
    print("   Lottery predictions are inherently uncertain.")
    print("="*60)

if __name__ == "__main__":
    main()
