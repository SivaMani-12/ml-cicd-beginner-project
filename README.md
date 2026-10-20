# Customer Churn ML CI/CD — Beginner Lab

This project uses synthetic data, scikit-learn, pytest, GitHub Actions, model evaluation and an optional Docker API. It is educational, not a production churn model.

## Prerequisites
Install Python 3.11, Git, a GitHub account, and optionally Docker Desktop. Open PowerShell in the project folder.

## 1. Environment
```powershell
python --version
git --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again.

## 2. Test and run ML pipeline
```powershell
pytest -v
python -m src.pipeline
Get-Content artifacts\metrics.json
Get-Content artifacts\evaluation_report.json
```
The pipeline validates synthetic data, trains a model, evaluates F1 and ROC-AUC, fails if F1 is below the quality threshold, and saves artifacts.

## 3. Run prediction API (after pipeline succeeds)
```powershell
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
Visit http://127.0.0.1:8000/docs. Try POST `/predict` with:
```json
{"tenure_months":4,"monthly_charges":85.5,"support_tickets":3,"contract_type":"month_to_month","uses_auto_pay":false}
```

## 4. Git CRUD practice
CRUD is not a formal CI/CD category; it is a useful way to learn repository maintenance.

Create repository history:
```powershell
git init
git add .
git commit -m "Create ML CI/CD project"
git branch -M main
```
Create an empty GitHub repository named `ml-cicd-beginner-project`, then connect (replace YOUR_USERNAME):
```powershell
git remote add origin https://github.com/YOUR_USERNAME/ml-cicd-beginner-project.git
git push -u origin main
```
Read state/history/differences:
```powershell
git status
git log --oneline
git diff
```
Update code or workflow, then test and push:
```powershell
pytest -v
python -m src.pipeline
git add .
git commit -m "Update model pipeline"
git push
```
Delete a tracked file safely:
```powershell
git rm path/to/file
git commit -m "Remove obsolete file"
git push
```

## 5. Observe GitHub Actions
Push the repo. Open GitHub repository > Actions > `ML CI - Test Validate Train`. Open the latest run and inspect each step. Download the `ml-artifacts` artifact. Workflow runs on pushes and pull requests targeting `main`, and can be run manually with `workflow_dispatch`.

CI includes dependency install, tests, data validation, model training/evaluation, and a quality gate. Artifact upload makes outputs downloadable. This is not yet production deployment; CD needs a real target environment, secrets/permissions, approvals, and rollback design.

## 6. Test a failing quality gate
In `src/model.py`, temporarily change `MIN_F1=0.60` to `MIN_F1=0.99`, then run `python -m src.pipeline`. It should fail if the model cannot meet the threshold. Restore 0.60 afterward. Do not lower quality requirements without a reason. You can also deliberately fail a test on a branch and observe the Actions run fail.

## 7. Optional Docker packaging
```powershell
docker build -t churn-api:1.0 .
docker run --rm -p 8000:8000 churn-api:1.0
```
Open http://localhost:8000/docs. The Dockerfile trains the model during image build for simplicity. Production systems commonly promote a tested, versioned model artifact explicitly.

## Troubleshooting
- Missing module: activate `.venv`, then `pip install -r requirements.txt`.
- Model missing in API: run `python -m src.pipeline` first.
- Quality gate fails: inspect `artifacts/evaluation_report.json`; investigate before changing the threshold.
- GitHub Actions fails: inspect the failed step's log and compare local Python/dependencies.
- Port 8000 busy: use `docker run --rm -p 8001:8000 churn-api:1.0` and open port 8001.

## Cleanup
Stop API with Ctrl+C. Optional remove artifacts: `Remove-Item artifacts\model.joblib, artifacts\metrics.json, artifacts\evaluation_report.json -ErrorAction SilentlyContinue`. Remove venv: `deactivate` then `Remove-Item -Recurse -Force .venv`. This does not delete GitHub history.
