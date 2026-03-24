.PHONY: build up down restart logs shell gpu-check

# Собрать образ
build:
	docker compose build

# Запустить контейнер в фоне
up:
	docker compose up -d
	@echo ""
	@echo "  JupyterLab запущен: http://localhost:8888"
	@echo ""

# Остановить контейнер
down:
	docker compose down

# Пересобрать и перезапустить
restart: down build up

# Следить за логами
logs:
	docker compose logs -f

# Открыть bash внутри контейнера
shell:
	docker compose exec finetuning bash

# Проверить доступность GPU внутри контейнера
gpu-check:
	docker compose exec finetuning nvidia-smi

# Проверить PyTorch видит ли GPU
torch-check:
	docker compose exec finetuning python -c \
		"import torch; print('CUDA available:', torch.cuda.is_available()); \
		 print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
