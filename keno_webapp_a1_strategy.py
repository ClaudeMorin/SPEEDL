from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import os

app = Flask(__name__)
CORS(app)

COMBINATION_SIZE = 10
TOP_N_COMBOS = 3

def generate_top_combinations(last_round, hot, cold):
    pool = list(set(last_round + hot + cold))
    combos = set()
    attempts = 0
    while len(combos) < TOP_N_COMBOS and attempts < 1000:
        combo = sorted(random.sample(pool, min(COMBINATION_SIZE, len(pool))))
        combos.add(tuple(combo))
        attempts += 1
    return [list(c) for c in combos]

@app.route("/", methods=["GET"])
def home():
    return "✅ a1 전략 API (Render Ready). Use POST /generate"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        last_round = data["last_round"]
        hot = data["hot"]
        cold = data["cold"]

        top_combos = generate_top_combinations(last_round, hot, cold)

        return jsonify({
            "top_combinations": top_combos
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
