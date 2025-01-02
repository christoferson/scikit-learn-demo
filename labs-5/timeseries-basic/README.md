### Random Forest

Loading data...
Creating features...
Preparing features...
Dataset shape: (6570, 18)

Features created: ['store_id', 'product_id', 'price', 'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end', 'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30', 'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_30']

Performing cross-validation...

Cross-validation results:
MAE: 4.15 (+/- 0.54) 
RMSE: 5.28 (+/- 0.78)
R2: 0.86 (+/- 0.05)

Tuning hyperparameters...
<site>sklearn\model_selection\_validation.py:425: FitFailedWarning: 
185 fits failed out of a total of 500.
The score on these train-test partitions for these parameters will be set to nan.
If these failures are not expected, you can try to debug them by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
64 fits failed with the following error:
Traceback (most recent call last):
  File "<site>sklearn\model_selection\_validation.py", line 729, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "<site>sklearn\base.py", line 1145, in wrapper
    estimator._validate_params()
  File "<site>sklearn\base.py", line 638, in _validate_params
    validate_parameter_constraints(
  File "<site>sklearn\utils\_param_validation.py", line 96, in validate_parameter_constraints
    raise InvalidParameterError(
sklearn.utils._param_validation.InvalidParameterError: The 'max_features' parameter of RandomForestRegressor must be an int in the range [1, inf), a float in the range (0.0, 1.0], a str among {'sqrt', 'log2'} or None. Got 'auto' instead.

--------------------------------------------------------------------------------
121 fits failed with the following error:
Traceback (most recent call last):
  File "<site>sklearn\model_selection\_validation.py", line 729, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "<site>sklearn\base.py", line 1145, in wrapper
    estimator._validate_params()
  File "<site>sklearn\base.py", line 638, in _validate_params
    validate_parameter_constraints(
  File "<site>sklearn\utils\_param_validation.py", line 96, in validate_parameter_constraints
    raise InvalidParameterError(
sklearn.utils._param_validation.InvalidParameterError: The 'max_features' parameter of RandomForestRegressor must be an int in the range [1, inf), a float in the range (0.0, 1.0], a str among {'log2', 'sqrt'} or None. Got 'auto' instead.

  warnings.warn(some_fits_failed_message, FitFailedWarning)
<site>sklearn\model_selection\_search.py:979: UserWarning: One or more of the test scores are non-finite: [        nan -4.01661537 -3.97972742 -3.97733796         nan         nan
         nan         nan         nan -3.95350111 -3.92769515 -3.97939971
 -4.00326804 -4.06661669 -4.01215758 -3.97025931         nan         nan
 -3.92758939         nan         nan -3.99729227 -4.09446062 -3.98768501
 -3.99753721 -4.03307854         nan -4.0132821  -3.94200311 -4.24694013
 -4.02975849 -3.91396505         nan -4.02436723 -4.01387736         nan
 -3.95526643         nan -3.92827435         nan         nan -4.06672956
         nan -3.93143828         nan -4.07823982 -3.96238375 -4.01345581
         nan -4.14260165 -3.93010086         nan -3.95842533         nan
 -4.01302757 -4.13601087 -3.92171007         nan         nan -3.94387215
 -4.41300665 -3.97870647 -4.01303605 -4.41628232 -3.95578048         nan
         nan -3.94441446 -3.92210556         nan -4.13345973 -4.0003867
 -4.03096307         nan -3.947224           nan -3.94247256 -4.23677237
 -3.97857275 -3.98619162         nan         nan         nan -3.93930457
 -4.0035062          nan -4.23069073         nan -4.03113895         nan
 -3.95268924 -4.03355177         nan -3.96364975 -4.03000997         nan
 -3.91734591 -3.9471473          nan -3.91790278]
  warnings.warn(
Best parameters found:  {'max_depth': 29, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 476}
Best MAE found:  3.9139650501495047

Training final model with best hyperparameters...
Final Model Performance:
MAE: 1.44
RMSE: 1.82
R2 Score: 0.99

Generating forecast...

Plotting forecast...

Forecast for the next 30 days:
         date  predicted_sales
0  2023-01-01       151.415966
1  2023-01-02       170.592437
2  2023-01-03       170.430672
3  2023-01-04       170.497899
4  2023-01-05       170.338235
5  2023-01-06       169.962185
6  2023-01-07       153.756303
7  2023-01-08       153.726891
8  2023-01-09       170.834034
9  2023-01-10       170.712185
10 2023-01-11       170.785714
11 2023-01-12       170.726891
12 2023-01-13       170.378151
13 2023-01-14       154.142857
14 2023-01-15       154.119748
15 2023-01-16       171.592437
16 2023-01-17       171.453782
17 2023-01-18       171.523109
18 2023-01-19       171.409664
19 2023-01-20       171.096639
20 2023-01-21       154.720588
21 2023-01-22       154.699580
22 2023-01-23       172.025210
23 2023-01-24       171.867647
24 2023-01-25       171.911765
25 2023-01-26       171.676471
26 2023-01-27       171.552521
27 2023-01-28       155.271008
28 2023-01-29       155.228992
29 2023-01-30       172.537815

Feature Importance:
                  feature  importance
12            sales_lag_7    0.196301
15   sales_rolling_mean_7    0.157416
13           sales_lag_14    0.147310
16  sales_rolling_mean_14    0.121971
8              is_weekend    0.091167
5             day_of_week    0.085748
17  sales_rolling_mean_30    0.079297
11            sales_lag_1    0.038143
2                   price    0.017555
7            week_of_year    0.016829
4                   month    0.012664
14           sales_lag_30    0.012598
6            day_of_month    0.008995
1              product_id    0.007798
0                store_id    0.003255
3                    year    0.001712
9          is_month_start    0.000786
10           is_month_end    0.000453


RandomForest V1 (Updated)
-----------------------------------------------------------------------------------------

Loading data...
Creating features...
Preparing features...
Dataset shape: (6570, 18)

Features created: ['store_id', 'product_id', 'price', 'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end', 'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30', 'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_30']

Performing cross-validation...

Cross-validation results:
MAE: 4.15 (+/- 0.54)
RMSE: 5.28 (+/- 0.78)
R2: 0.86 (+/- 0.05)

Tuning hyperparameters...
Best parameters found:  {'max_depth': 13, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 338}
Best MAE found:  3.9075480408023475

Training final model with best hyperparameters...
Final Model Performance:
MAE: 2.27
RMSE: 2.83
R2 Score: 0.97

Generating forecast...

Plotting forecast...

Forecast for the next 30 days:
         date  predicted_sales
0  2023-01-01       149.678498
1  2023-01-02       169.898472
2  2023-01-03       169.959025
3  2023-01-04       170.045810
4  2023-01-05       169.778500
5  2023-01-06       169.565841
6  2023-01-07       152.989612
7  2023-01-08       152.793014
8  2023-01-09       170.020373
9  2023-01-10       169.974775
10 2023-01-11       170.083174
11 2023-01-12       170.032884
12 2023-01-13       169.827375
13 2023-01-14       153.578862
14 2023-01-15       153.334927
15 2023-01-16       171.050058
16 2023-01-17       170.940590
17 2023-01-18       170.971474
18 2023-01-19       170.857217
19 2023-01-20       170.729603
20 2023-01-21       153.936160
21 2023-01-22       153.865943
22 2023-01-23       171.506750
23 2023-01-24       171.406695
24 2023-01-25       171.405955
25 2023-01-26       171.287800
26 2023-01-27       171.053332
27 2023-01-28       154.812661
28 2023-01-29       154.598066
29 2023-01-30       172.054187

Feature Importance:
                  feature  importance
12            sales_lag_7    0.192768
15   sales_rolling_mean_7    0.162832
13           sales_lag_14    0.155041
16  sales_rolling_mean_14    0.118807
5             day_of_week    0.090988
8              is_weekend    0.088242
17  sales_rolling_mean_30    0.087269
11            sales_lag_1    0.035082
7            week_of_year    0.015082
4                   month    0.010585
14           sales_lag_30    0.010527
6            day_of_month    0.006894
1              product_id    0.006688
0                store_id    0.002489
3                    year    0.001133
9          is_month_start    0.000813
10           is_month_end    0.000384


RandomForest (V2)
-----------------------------

Loading data...
Creating features...
Preparing features...
Dataset shape: (6570, 18)

Features created: ['store_id', 'product_id', 'price', 'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end', 'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30', 'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_30']

Splitting data into train and test sets...
Training set shape: (5256, 18), Test set shape: (1314, 18)

Performing cross-validation on training set...

Cross-validation results:
MAE: 4.10 (+/- 0.28)
RMSE: 5.20 (+/- 0.39)
R2: 0.86 (+/- 0.02)

Tuning hyperparameters...
Best parameters found:  {'max_depth': 16, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 463}
Best MAE found:  3.9062776179416856

Training final model with best hyperparameters...

Evaluating model on training set:
Training MAE: 1.88
Training RMSE: 2.37
Training R2 Score: 0.98

Evaluating model on test set:
Test MAE: 3.96
Test RMSE: 5.01
Test R2 Score: 0.88

Generating forecast...

Plotting forecast...

Forecast for the next 30 days:
         date  predicted_sales
0  2023-01-01       147.301440
1  2023-01-02       166.889815
2  2023-01-03       167.081103
3  2023-01-04       167.389869
4  2023-01-05       167.246654
5  2023-01-06       167.146678
6  2023-01-07       149.802356
7  2023-01-08       149.778535
8  2023-01-09       166.971065
9  2023-01-10       167.016612
10 2023-01-11       167.429116
11 2023-01-12       167.392489
12 2023-01-13       167.355934
13 2023-01-14       150.183596
14 2023-01-15       150.181738
15 2023-01-16       168.682529
16 2023-01-17       168.720107
17 2023-01-18       169.100687
18 2023-01-19       168.951965
19 2023-01-20       169.010880
20 2023-01-21       150.595658
21 2023-01-22       150.551138
22 2023-01-23       169.050578
23 2023-01-24       169.072803
24 2023-01-25       169.204643
25 2023-01-26       168.969852
26 2023-01-27       168.907554
27 2023-01-28       150.538447
28 2023-01-29       150.407822
29 2023-01-30       169.386184

Feature Importance:
                  feature  importance
12            sales_lag_7    0.194131
15   sales_rolling_mean_7    0.162929
13           sales_lag_14    0.148560
16  sales_rolling_mean_14    0.109767
8              is_weekend    0.101403
5             day_of_week    0.100538
17  sales_rolling_mean_30    0.064020
11            sales_lag_1    0.037830
7            week_of_year    0.018347
2                   price    0.016483
4                   month    0.014267
14           sales_lag_30    0.011885
6            day_of_month    0.008577
1              product_id    0.006111
0                store_id    0.002467
3                    year    0.001384
9          is_month_start    0.000796
10           is_month_end    0.000503


RandomForest (V3)
--------------------------------------------------------------------------------

Loading data...
Creating features...
Preparing features...
Dataset shape: (6570, 31)

Features created: ['store_id', 'product_id', 'price', 'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end', 'quarter', 'is_quarter_start', 'is_quarter_end', 'days_in_month', 'sine_day_of_year', 'cosine_day_of_year', 'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30', 'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_30', 'sales_rolling_mean_60', 'sales_rolling_mean_90', 'sales_lag_365', 'sales_rolling_mean_365', 'sales_pct_change_1', 'sales_pct_change_7', 'sales_pct_change_365']

Splitting data into train and test sets...
Training set shape: (5256, 31), Test set shape: (1314, 31)

Performing cross-validation on training set...

Cross-validation results:
MAE: 1.16 (+/- 0.81)
RMSE: 2.26 (+/- 0.79)
R2: 0.97 (+/- 0.02)

Tuning hyperparameters...
Best parameters found:  {'max_depth': 29, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 
2, 'n_estimators': 476}
Best MAE found:  0.724317705523889

Training final model with best hyperparameters...

Evaluating model on training set:
Training MAE: 0.21
Training RMSE: 0.52
Training R2 Score: 1.00

Evaluating model on test set:
Test MAE: 0.83
Test RMSE: 1.66
Test R2 Score: 0.99

Generating forecast...

Plotting forecast...

Forecast for the next 30 days:
         date  predicted_sales
0  2023-01-01       165.754202
1  2023-01-02       170.773109
2  2023-01-03       170.745798
3  2023-01-04       170.743697
4  2023-01-05       170.760504
5  2023-01-06       170.792017
6  2023-01-07       165.659664
7  2023-01-08       165.649160
8  2023-01-09       170.794118
9  2023-01-10       170.789916
10 2023-01-11       170.836134
11 2023-01-12       170.859244
12 2023-01-13       170.810924
13 2023-01-14       165.500000
14 2023-01-15       165.497899
15 2023-01-16       170.867647
16 2023-01-17       170.771008
17 2023-01-18       170.764706
18 2023-01-19       170.785714
19 2023-01-20       170.777311
20 2023-01-21       165.489496
21 2023-01-22       165.487395
22 2023-01-23       170.762605
23 2023-01-24       170.756303
24 2023-01-25       170.750000
25 2023-01-26       170.777311
26 2023-01-27       170.768908
27 2023-01-28       165.575630
28 2023-01-29       165.579832
29 2023-01-30       170.758403

Feature Importance:
                   feature  importance
18             sales_lag_7    0.703348
29      sales_pct_change_7    0.168863
21    sales_rolling_mean_7    0.082289
8               is_weekend    0.013231
5              day_of_week    0.012128
28      sales_pct_change_1    0.004582
22   sales_rolling_mean_14    0.002584
7             week_of_year    0.002178
16      cosine_day_of_year    0.002175
19            sales_lag_14    0.001628
17             sales_lag_1    0.001114
27  sales_rolling_mean_365    0.000914
25   sales_rolling_mean_90    0.000768
23   sales_rolling_mean_30    0.000757
24   sales_rolling_mean_60    0.000656
2                    price    0.000438
6             day_of_month    0.000437
15        sine_day_of_year    0.000325
30    sales_pct_change_365    0.000325
26           sales_lag_365    0.000308
20            sales_lag_30    0.000291
4                    month    0.000278
0                 store_id    0.000085
12        is_quarter_start    0.000059
14           days_in_month    0.000051
11                 quarter    0.000051
1               product_id    0.000044
9           is_month_start    0.000043
3                     year    0.000034
10            is_month_end    0.000009
13          is_quarter_end    0.000005