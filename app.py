import os
DATA_PATH = 'kenohistory7.csv'
MODEL_PATH = 'model.pkl'

# 모델 파일이 없을 때 최초 학습
if not os.path.exists(MODEL_PATH):
    data = pd.read_csv(DATA_PATH)
    X, y = prepare_features(data)
    model = MultiOutputClassifier(XGBClassifier(n_estimators=100, max_depth=7))
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

from flask import Flask, render_template, request, jsonify
import pandas as pd  # 반드시 추가!
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os
import gdown

app = Flask(__name__)

DATA_PATH = 'kenohistory7.csv'
MODEL_PATH = 'model.pkl'

# 모델 최초 1회 로딩 필수!
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

# 여기에 반드시 `/predict` route 추가 필수!
@app.route('/predict', methods=['POST'])
def predict():
    date = request.form.get('date')
    round_num = request.form.get('round_num')

    if not date or not round_num:
        return jsonify({'error': '날짜와 회차를 입력해주세요.'})

    round_num = int(round_num)

    data = pd.read_csv(DATA_PATH)
    daily_data = data[data['Date'] == date].reset_index(drop=True)

    try:
        idx = daily_data.index[daily_data['GlobalRound'] == round_num][0]
    except IndexError:
        return jsonify({'error': '데이터에 해당 회차가 없습니다.'})

    if idx == 0 or pd.isna(daily_data.iloc[idx - 1]['Numbers']):
        return jsonify({'error': '이전 회차 데이터가 없습니다.'})

    prev_numbers = daily_data.iloc[idx - 1]['Numbers']
    hour = pd.to_datetime(daily_data.iloc[idx]['Time']).hour
    minute = pd.to_datetime(daily_data.iloc[idx]['Time']).minute
    global_round = daily_data.iloc[idx]['GlobalRound']

    features = [global_round, hour, minute] + \
               [1 if str(num).zfill(2) in prev_numbers else 0 for num in range(1, 71)]
    X_input = np.array(features).reshape(1, -1)

    pred_proba = np.array([est.predict_proba(X_input)[0][1] for est in model.estimators_])

    sets = []
    for _ in range(3):
        nums = np.argsort(pred_proba + np.random.rand(70) * 0.01)[-22:][::-1] + 1
        sets.append(sorted(nums.tolist()))

    return jsonify({'sets': sets})

# update route 추가 필수!
@app.route('/update', methods=['POST'])
def update():
    date, round_num, nums = request.form['date'], int(request.form['round_num']), request.form['numbers']
    actual_numbers = [int(n) for n in nums.split(',') if n.strip().isdigit()]
    numbers_str = ','.join([str(n).zfill(2) for n in actual_numbers])

    data = pd.read_csv(DATA_PATH)
    data.loc[(data['Date'] == date) & (data['GlobalRound'] == round_num), 'Numbers'] = numbers_str
    data.to_csv(DATA_PATH, index=False)

    X, y = prepare_features(data)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    return jsonify({'status': '모델 업데이트 완료!'})

# prepare_features 함수는 기존 그대로 유지하세요 (업로드된 내용 유지)
