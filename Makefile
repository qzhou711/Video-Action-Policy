.PHONY: login install precompute train-stage1 train-stage2 evaluate train-all clean help

DEVICE ?= cuda
PRECOMPUTED_DIR ?= precomputed
STAGE1_CKPT ?= checkpoints/stage1/final
STAGE2_CKPT ?= checkpoints/stage2/final
HF_TOKEN ?= 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

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

login: ## Authenticate with HuggingFace
	huggingface-cli login --token $(HF_TOKEN)

install: ## Install all dependencies and log in to HuggingFace
	pip install -r requirements.txt
	$(MAKE) login

precompute: ## Precompute T5 embeddings (and optionally VAE latents with LATENTS=1)
ifdef LATENTS
	python scripts/precompute_embeddings.py --latents --output_dir $(PRECOMPUTED_DIR) --device $(DEVICE)
else
	python scripts/precompute_embeddings.py --output_dir $(PRECOMPUTED_DIR) --device $(DEVICE)
endif

train-stage1: ## Stage 1: LoRA finetuning of video backbone
	python scripts/train_stage1.py --precomputed_dir $(PRECOMPUTED_DIR) --device $(DEVICE)

train-stage2: ## Stage 2: Action decoder training (frozen backbone)
	python scripts/train_stage2.py --precomputed_dir $(PRECOMPUTED_DIR) --device $(DEVICE)

evaluate: ## Evaluate trained model on held-out episodes
	python scripts/evaluate.py \
		--stage1_checkpoint $(STAGE1_CKPT) \
		--stage2_checkpoint $(STAGE2_CKPT) \
		--precomputed_dir $(PRECOMPUTED_DIR) \
		--device $(DEVICE)

train-all: precompute train-stage1 train-stage2 evaluate ## Run full pipeline: precompute → stage1 → stage2 → evaluate

clean: ## Remove checkpoints and precomputed data
	rm -rf checkpoints/ precomputed/ wandb/
