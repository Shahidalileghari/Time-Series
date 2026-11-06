import numpy as np
import pandas as pd
import pytest
from ts_intel.data.validation import validate_series_frame
from ts_intel.features.lag import make_lag_features
from ts_intel.models.baselines import persistence, seasonal_naive
from ts_intel.metrics import forecast_metrics
from ts_intel.anomaly.detection import residual_anomaly_scores, flag_anomalies
from ts_intel.anomaly.change_points import rolling_mean_shift
from ts_intel.evaluation.backtest import rolling_origin_backtest

def test_validate_sorts_and_rejects_duplicate_timestamps():
    frame=pd.DataFrame({"t":["2024-01-02","2024-01-01"],"y":[2,1]})
    out=validate_series_frame(frame,"t","y")
    assert out.y.tolist()==[1,2]
    frame.loc[1,"t"]="2024-01-01"
    with pytest.raises(ValueError): validate_series_frame(frame,"t","y")

def test_lag_features_are_shifted_to_avoid_current_target_leakage():
    s=pd.Series([1.,2.,3.,4.])
    f=make_lag_features(s,lags=(1,),windows=(2,))
    assert f.loc[2,"lag_1"]==2
    assert f.loc[2,"rolling_mean_2"]==1.5

def test_baselines():
    s=pd.Series([2.,3.,4.])
    assert persistence(s,2).tolist()==[4.,4.]
    assert seasonal_naive(pd.Series([1,2,3,4]),3,2).tolist()==[3.,4.,3.]

def test_metrics():
    m=forecast_metrics([1,2],[1,4])
    assert m["mae"]==1.0 and m["rmse"]==pytest.approx(2**0.5)

def test_anomaly_scoring_flags_outlier():
    scores=residual_anomaly_scores([0,0,0,0,0,10],[0,0,0,0,0,0])
    assert flag_anomalies(scores,3.5)[-1]

def test_change_point_detects_mean_shift():
    values=np.r_[np.zeros(30),np.ones(30)*8]
    assert rolling_mean_shift(values,window=8,threshold=2)

def test_backtest_generates_folds():
    s=pd.Series(np.arange(40,dtype=float))
    result=rolling_origin_backtest(s,horizon=3,initial_train_size=20,model="persistence")
    assert len(result)>0 and {"mae","rmse","smape"} <= set(result.columns)
