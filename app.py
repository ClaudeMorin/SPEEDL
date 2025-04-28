from flask import Flask
app = Flask(__name__)  # 반드시 이렇게 작성되어 있어야 합니다.

def prepare_features(df):
    df = df.copy()

    # 시간정보를 명확한 포맷으로 안전하게 처리 (에러 발생 방지)
    df['Hour'] = pd.to_datetime(df['Time'], errors='coerce', format='%I:%M:%S %p').dt.hour
    df['Minute'] = pd.to_datetime(df['Time'], errors='coerce', format='%I:%M:%S %p').dt.minute

    # 이전 회차의 번호(Prev_Numbers) 생성
    df['Prev_Numbers'] = df['Numbers'].shift(1)

    # 숫자 (1~70)의 이전 회차 등장 여부를 명확히 체크
    for num in range(1, 71):
        num_str = str(num).zfill(2)
        df[f'Prev_Num_{num}'] = df['Prev_Numbers'].apply(
            lambda x: 1 if pd.notna(x) and num_str in x.split(',') else 0
        )

    # 필수 데이터 누락된 행을 명확히 제거 (결측값 방지)
    df.dropna(subset=['Numbers', 'Prev_Numbers', 'Hour', 'Minute'], inplace=True)

    # 입력 특성(X) 명확히 정의 및 생성
    X_columns = ['GlobalRound', 'Hour', 'Minute'] + [f'Prev_Num_{num}' for num in range(1, 71)]
    X = df[X_columns].to_numpy()

    # 예측할 정답(y) 명확히 생성 (NaN 완벽 방지)
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
