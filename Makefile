.PHONY: install test dry-run benchmark train-gpu docker-build

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=src pytest -v

dry-run:
	PYTHONPATH=src python3 -m bio_peft.train_lora --config configs/lora_sft_config.yaml --dry-run

benchmark:
	python3 src/bio_peft/benchmark_strategies.py

train-gpu:
	bash scripts/run_gpu_train.sh configs/lora_sft_config.yaml

docker-build:
	docker build -t bio-peft-ontology-linker:latest .
