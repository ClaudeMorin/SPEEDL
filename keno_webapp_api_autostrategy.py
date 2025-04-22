from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from collections import Counter
import os

app = Flask(__name__)
CORS(app)

COMBINATION_SIZE = 10
TOP_N_COMBOS = 3
FULL_SET_SIZE = 22

def extract_hot_cold(recent_sets):
    flat = [n for s in recent_sets for n in s]
    counter = Counter(flat)
    hot = [n for n, _ in counter.most_common(10)]
    cold = [n for n, _ in counter.most_common()][-10:]
    recent2plus = [n for n, c in counter.items() if c >= 2]
    return hot, cold, recent2plus

def generate_top_combinations(filtered_pool):
    combos = set()
    attempts = 0
    while len(combos) < TOP_N_COMBOS and attempts < 1000:
        combo = sorted(random.sample(filtered_pool, min(COMBINATION_SIZE, len(filtered_pool))))
        combos.add(tuple(combo))
        attempts += 1
    return [list(c) for c in combos]

def predict_full_sum(pool):
    pool = list(set(pool))
    while len(pool) < FULL_SET_SIZE:
        pool.append(random.randint(1, 80))
    return sum(random.sample(pool, FULL_SET_SIZE))

@app.route("/", methods=["GET"])
def home():
    return "✅ Auto-strategy API ready. Send POST /generate with recent_10_sets"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        recent_10_sets = data["recent_10_sets"]
        if len(recent_10_sets) != 10 or any(len(s) != 22 for s in recent_10_sets):
            return jsonify({"error": "Input must be 10 sets of 22 numbers each."})

        hot, cold, recent2plus = extract_hot_cold(recent_10_sets)
        a1_pool = set(hot + cold + recent_10_sets[-1])  # 마지막 세트 = 직전 회차
        final_pool = list(a1_pool & set(recent2plus))

        if len(final_pool) < 10:
            return jsonify({"error": "Not enough overlapping numbers to generate combo."})

        top_combos = generate_top_combinations(final_pool)
        full_sum = predict_full_sum(final_pool)

        return jsonify({
            "final_pool": sorted(final_pool),
            "top_combinations": top_combos,
            "predicted_full_sum": full_sum
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
