.PHONY: install build test dashboard notebook clean

PYTHON ?= python3
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
	$(PYTHON) -m jupyter nbconvert --to notebook --execute "churn pipeline.ipynb" --output "churn pipeline.executed.ipynb"

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__ visualizations/churn_dashboard.html
