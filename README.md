# ml-for-forecasting
This repository checks for non-deep learning machine learning we can use in conjunction with skforecast library to 
make a time seriesforecasting model.

To evaluate the performance of each ml model in forecasting task, we use skforecast package to transform them into forecaster. This involves
1) Instantiating an object of ForecastAutoreg or ForecasterAutoregMultiVariate with the ML model from scikit-learn library.
2) Use random_search to search for optimal parameter.
3) Evaluate the model using the best parameter found in step no. 3. by computing evaluation metrics (mae, mape, mse, rmse).

We automate the process to tests these models in different time series data. 

## Reminder
-I updated the default branch to separate-datasets-for-uni-multi-ts
