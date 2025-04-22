from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from collections import Counter

app = Flask(__name__)
CORS(app)

COMBINATION_SIZE = 10
TOP_N_COMBOS = 3
FULL_SET_SIZE = 22

def generate_top_combinations(filtered_pool):
    combos = set()
    attempts = 0
    while len(combos) < TOP_N_COMBOS and attempts < 1000:
        combo = sorted(random.sample(filtered_pool, min(COMBINATION_SIZE, len(filtered_pool))))
        combos.add(tuple(combo))
        attempts += 1
    return [list(c) for c in combos]

def predict_full_sum(filtered_pool):
    full_pool = list(set(filtered_pool))
    while len(full_pool) < FULL_SET_SIZE:
        full_pool.append(random.randint(1, 80))
    return sum(random.sample(full_pool, FULL_SET_SIZE))

@app.route("/", methods=["GET"])
def home():
    return "✅ Keno WebApp API is live. Use POST /generate"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        last_round = data["last_round"]
        hot = data["hot"]
        cold = data["cold"]
        recent_10_sets = data["recent_10_sets"]  # 10개 회차 번호 (22개씩)

        flat = [n for s in recent_10_sets for n in s]
        counter = Counter(flat)
        recent2plus_pool = set(n for n, c in counter.items() if c >= 2)
        a1_pool = set(last_round + hot + cold)
        filtered_pool = list(a1_pool & recent2plus_pool)

        if len(filtered_pool) < 10:
            return jsonify({"error": "Not enough filtered numbers for combo."})

        top_combos = generate_top_combinations(filtered_pool)
        full_sum = predict_full_sum(filtered_pool)

        return jsonify({
            "filtered_pool": sorted(filtered_pool),
            "top_combinations": top_combos,
            "predicted_full_sum": full_sum
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
