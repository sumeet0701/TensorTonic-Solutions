def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    import pandas as pd
    df = pd.DataFrame(models)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # sort by rules
    df_sorted = df.sort_values(
        by=['accuracy', 'latency', 'timestamp'],
        ascending=[False, True, False]
    )
    return df_sorted.iloc[0]['name']