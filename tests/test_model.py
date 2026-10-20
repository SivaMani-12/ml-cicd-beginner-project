from src.data import make_dataset,FEATURES,TARGET
from src.model import build_model,train
def test_model_predicts():
    d=make_dataset(120); m=build_model(); m.fit(d[FEATURES],d[TARGET]); assert len(m.predict(d[FEATURES].head()))==5
def test_pipeline_writes_artifacts(tmp_path):
    report=train(str(tmp_path)); assert report["quality_gate_passed"]
    assert (tmp_path/"model.joblib").exists(); assert (tmp_path/"metrics.json").exists()
