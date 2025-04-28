# Flask 및 필수 라이브러리 import
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os
import gdown

app = Flask(__name__)

DATA_PATH = 'kenohistory7.csv'
MODEL_PATH = 'model.pkl'

# Google Drive에서 최신 모델과 데이터 파일 자동 다운로드
MODEL_URL = 'https://drive.google.com/uc?id=1-0aDTDnPgESOmGaSOHGN--_incCrVH6O'
DATA_URL = 'https://drive.google.com/uc?id=1-49bfcTDao0RZMNqo0klgf45-2k6StwK'

# 모델 파일 강제 최신화
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# 데이터 파일 강제 최신화
if os.path.exists(DATA_PATH):
    os.remove(DATA_PATH)
gdown.download(DATA_URL, DATA_PATH, quiet=False)

# 데이터 전처리 함수
def prepare_features(df):
    df = df.copy()
    df['Hour'] = pd.to_datetime(df['Time'], errors='coerce', format='%I:%M:%S %p').dt.hour
    df['Minute'] = pd.to_datetime(df['Time'], errors='coerce', format='%I:%M:%S %p').dt.minute
    df['Prev_Numbers'] = df['Numbers'].shift(1)
    for num in range(1, 71):
        num_str = str(num).zfill(2)
        df[f'Prev_Num_{num}'] = df['Prev_Numbers'].apply(
            lambda x: 1 if pd.notna(x) and num_str in x.split(',') else 0
        )
    df.dropna(subset=['Numbers', 'Prev_Numbers', 'Hour', 'Minute'], inplace=True)
    X_columns = ['GlobalRound', 'Hour', 'Minute'] + [f'Prev_Num_{num}' for num in range(1, 71)]
    X = df[X_columns].to_numpy()

    def numbers_to_binary(nums):
        binary_array = np.zeros(70, dtype=int)
        if pd.isna(nums):
            return binary_array
        for n in nums.split(','):
            if n.strip().isdigit():
                idx = int(n.strip()) - 1
                if 0 <= idx < 70:
                    binary_array[idx] = 1
        return binary_array.tolist()

    y = df['Numbers'].apply(numbers_to_binary).tolist()
    return X, np.array(y)

# 모델 초기 로드
model = joblib.load(MODEL_PATH)

# 메인 페이지 설정
@app.route('/')
def home():
    return render_template('index.html')

# CSV 데이터 확인용 라우트 추가
@app.route('/recent_data')
def recent_data():
    data = pd.read_csv(DATA_PATH)
    recent = data.tail(5).to_dict(orient='records')
    return jsonify(recent)

# 최근 5회차로만 학습 및 예측 (빈 셀 무시)
@app.route('/predict', methods=['POST'])
def predict():
    date = request.form.get('date')
    round_num = request.form.get('round_num')

    if not date or not round_num:
        return jsonify({'error': '날짜와 회차를 입력해주세요.'})

    round_num = int(round_num)
    data = pd.read_csv(DATA_PATH)

    # 빈 셀 무시, 최근 5회차만 선택
    valid_data = data.dropna(subset=['Numbers']).tail(5).copy()

    if len(valid_data) < 5:
        return jsonify({'error': '최소 5회차의 유효 데이터가 필요합니다.'})

    X_recent, y_recent = prepare_features(valid_data)
    model.fit(X_recent, y_recent)

    try:
        current_idx = data[(data['Date'] == date) & (data['GlobalRound'] == round_num)].index[0]
        prev_row = data.iloc[current_idx - 1]
    except (IndexError, KeyError):
        return jsonify({'error': '입력한 날짜와 회차를 찾을 수 없습니다.'})

    recent_valid_row = data.iloc[:current_idx].dropna(subset=['Numbers']).iloc[-1]
    prev_numbers = recent_valid_row['Numbers']
    hour = pd.to_datetime(prev_row['Time']).hour
    minute = pd.to_datetime(prev_row['Time']).minute
    global_round = prev_row['GlobalRound']

    features = [global_round, hour, minute] + [
        1 if str(num).zfill(2) in prev_numbers else 0 for num in range(1, 71)
    ]
    X_input = np.array(features).reshape(1, -1)

    pred_proba = np.array([est.predict_proba(X_input)[0][1] for est in model.estimators_])

    top22_mean_proba = np.mean(np.sort(pred_proba)[-22:])
    threshold = 0.5

    sets = []
    if top22_mean_proba >= threshold:
        for _ in range(3):
            nums = np.argsort(pred_proba + np.random.rand(70) * 0.01)[-22:][::-1] + 1
            sets.append(sorted(nums.tolist()))
        cold_jump = False
    else:
        nums = np.argsort(pred_proba)[-22:][::-1] + 1
        sets.append(sorted(nums.tolist()))
        cold_jump = True

    return jsonify({'sets': sets, 'cold_jump': cold_jump})

# 실제 번호 입력 후 모델 업데이트
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

if __name__ == '__main__':
    app.run()
