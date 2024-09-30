


df = pd.read_csv('', index_col=0, parse_dates=True)
freq = infer_frequency(df)
results = forecast_and_evaluate_random_forest(df_arg = df, exog=exog)


print(results['mae'])