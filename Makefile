install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	python src/test_model.py
