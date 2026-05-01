.PHONY: install build test dashboard notebook clean

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

build: test dashboard

test:
	$(PYTHON) -m pytest -q

dashboard:
	$(PYTHON) -m src.make_dashboard --input Churn_Modelling.csv --output visualizations/churn_dashboard.html

notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace "churn pipeline.ipynb"

clean:
	rm -rf .pytest_cache .jupyter_runtime __pycache__ src/__pycache__ tests/__pycache__ visualizations/churn_dashboard.html
