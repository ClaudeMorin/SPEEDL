def prepare_features(df):
    df = df.copy()
    df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
    df['Minute'] = pd.to_datetime(df['Time'], errors='coerce').dt.minute
    df['Prev_Numbers'] = df['Numbers'].shift(1)

    for num in range(1,71):
        df[f'Prev_Num_{num}'] = df['Prev_Numbers'].apply(lambda x: str(num).zfill(2) in str(x)).astype(int)

    df.dropna(subset=['Numbers', 'Prev_Numbers', 'Hour', 'Minute'], inplace=True)

    X = df[['GlobalRound','Hour','Minute']+[f'Prev_Num_{num}' for num in range(1,71)]].values
    y = df['Numbers'].apply(
        lambda nums: np.isin(range(1,71),[int(n) for n in str(nums).split(',')]).astype(int)
    ).tolist()

    return np.array(X), np.array(y)
