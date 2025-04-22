
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from collections import Counter
import os

app = Flask(__name__)
CORS(app)

COMBINATION_SIZE = 10
COMBO_COUNT = 4
FULL_SET_SIZE = 22

def extract_filtered_pool(recent_sets):
    flat = [n for s in recent_sets for n in s]
    counter = Counter(flat)
    hot = [n for n, c in counter.items() if c >= 3]
    cold = [n for n, c in counter.items() if c == 1]
    recent2plus = [n for n, c in counter.items() if c >= 2]
    last_round = recent_sets[0]  # 최신 회차를 첫 번째 세트로 간주

    a1_pool = set(hot + cold + last_round)
    final_pool = list(a1_pool & set(recent2plus))
    return final_pool

def generate_combinations(pool, n=COMBO_COUNT):
    combos = set()
    attempts = 0
    while len(combos) < n and len(pool) >= COMBINATION_SIZE and attempts < 1000:
        combo = tuple(sorted(random.sample(pool, COMBINATION_SIZE)))
        combos.add(combo)
        attempts += 1
    return [list(c) for c in combos]

def predict_full_sum(pool):
    while len(pool) < FULL_SET_SIZE:
        pool.append(random.randint(1, 80))
    return sum(random.sample(pool, FULL_SET_SIZE))

@app.route("/", methods=["GET"])
def home():
    return "✅ 4조합 혼합전략 API (Render Ready). POST /generate"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        recent_10_sets = data["recent_10_sets"]
        if len(recent_10_sets) < 7 or any(len(s) != 22 for s in recent_10_sets[:7]):
            return jsonify({"error": "입력은 최근 7회차 22개씩 번호여야 합니다."})

        final_pool = extract_filtered_pool(recent_10_sets[:7])
        if len(final_pool) < 10:
            return jsonify({"error": "조합 생성에 충분한 번호가 없습니다."})

        combos = generate_combinations(final_pool)
        full_sum = predict_full_sum(final_pool)

        return jsonify({
            "final_pool": sorted(final_pool),
            "top_combinations": combos,
            "predicted_full_sum": full_sum
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
