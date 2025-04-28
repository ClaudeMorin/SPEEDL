from flask import Flask, render_template, request, jsonify
import pandas as pd, numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib, os, random

app = Flask(__name__)
DATA_PATH = 'kenohistory7.csv'
MODEL_PATH = 'model.pkl'

def prepare_features(df):
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    df['Minute'] = pd.to_datetime(df['Time']).dt.minute
    df['Prev_Numbers'] = df['Numbers'].shift(1)
    for num in range(1,71):
        df[f'Prev_Num_{num}'] = df['Prev_Numbers'].str.contains(str(num).zfill(2)).fillna(0).astype(int)
    df.dropna(subset=['Prev_Numbers'], inplace=True)
    X = df[['GlobalRound','Hour','Minute']+[f'Prev_Num_{num}' for num in range(1,71)]].values
    y = df['Numbers'].apply(lambda nums: np.isin(range(1,71),[int(n) for n in nums.split(',')]).astype(int)).tolist()
    return np.array(X), np.array(y)

# 최초 1회 모델 학습
if not os.path.exists(MODEL_PATH):
    data = pd.read_csv(DATA_PATH)
    X,y = prepare_features(data)
    model = MultiOutputClassifier(XGBClassifier(n_estimators=100,max_depth=7))
    model.fit(X,y)
    joblib.dump(model,MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

def clean_numbers(input_str):
    nums=input_str.replace(' ','').split(',')
    return [int(n.lstrip('0')) for n in nums if n]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date, round_num = request.form['date'], int(request.form['round_num'])
    data=pd.read_csv(DATA_PATH)
    daily=data[data['Date']==date].reset_index(drop=True)
    try:
        idx=daily.index[daily['GlobalRound']==round_num][0]
        prev_numbers=daily.iloc[idx-1]['Numbers']
    except:
        return jsonify({'error':'이전 데이터가 없습니다.'})

    hour=pd.to_datetime(daily.iloc[idx]['Time']).hour
    minute=pd.to_datetime(daily.iloc[idx]['Time']).minute
    global_round=daily.iloc[idx]['GlobalRound']
    features=[global_round,hour,minute]+[1 if str(num).zfill(2) in prev_numbers else 0 for num in range(1,71)]
    X_input=np.array(features).reshape(1,-1)

    pred_proba=np.array([est.predict_proba(X_input)[0][1] for est in model.estimators_])

    sets = []
    for _ in range(3):
        nums = np.argsort(pred_proba+np.random.rand(70)*0.01)[-22:][::-1]+1
        sets.append(sorted(nums.tolist()))

    return jsonify({'sets':sets})

@app.route('/update', methods=['POST'])
def update():
    date, round_num, nums = request.form['date'], int(request.form['round_num']), request.form['numbers']
    actual_numbers=clean_numbers(nums)
    numbers_str=','.join([str(n).zfill(2) for n in actual_numbers])
    data=pd.read_csv(DATA_PATH)
    data.loc[(data['Date']==date)&(data['GlobalRound']==round_num),'Numbers']=numbers_str
    data.to_csv(DATA_PATH,index=False)
    X,y=prepare_features(data)
    model.fit(X,y)
    joblib.dump(model,MODEL_PATH)
    return jsonify({'status':'모델 업데이트 완료!'})

if __name__=='__main__':
    app.run()
