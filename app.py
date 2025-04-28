def prepare_features(df):
    df = df.copy()
    
    # 시간 정보 처리 (안전하게 에러 처리)
    df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
    df['Minute'] = pd.to_datetime(df['Time'], errors='coerce').dt.minute
    
    # 이전 회차 숫자 정보 생성
    df['Prev_Numbers'] = df['Numbers'].shift(1)

    # 숫자(1~70) 존재 여부를 체크하여 특성 생성 (빈칸 안전처리 포함)
    for num in range(1, 71):
        df[f'Prev_Num_{num}'] = df['Prev_Numbers'].apply(
            lambda x: 1 if isinstance(x, str) and str(num).zfill(2) in x else 0
        )

    # 결측값 제거 (중요 컬럼에 결측 존재 시 해당 row 삭제)
    df.dropna(subset=['Numbers', 'Prev_Numbers', 'Hour', 'Minute'], inplace=True)

    # 특성 데이터(X) 생성
    X_columns = ['GlobalRound', 'Hour', 'Minute'] + [f'Prev_Num_{num}' for num in range(1, 71)]
    X = df[X_columns].values

    # 정답 데이터(y) 생성 (NaN 완벽 방지)
    def numbers_to_binary(nums):
        if not isinstance(nums, str):
            nums = ''
        nums_set = set(int(n) for n in nums.split(',') if n.strip().isdigit())
        return [1 if num in nums_set else 0 for num in range(1, 71)]

    y = df['Numbers'].apply(numbers_to_binary).tolist()

    return np.array(X), np.array(y)
