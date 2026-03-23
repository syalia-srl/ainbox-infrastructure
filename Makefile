# ==============================================================================
# SuperBot Infrastructure Operations
# ==============================================================================
.PHONY: image run stop

# --- Build Phase ---
image:
ifndef RECIPE
	$(error RECIPE is not set. Usage: make image RECIPE=recipes/my_recipe.json)
endif
	@chmod +x build.sh
	@./build.sh $(RECIPE)

# --- Deployment Phase ---
MODE ?= gpu

run:
ifndef CONFIG
	$(error CONFIG is not set. Usage: make run CONFIG=deploy/your_config.json TAG=your_image_tag)
endif
ifndef TAG
	$(error TAG is not set. Usage: make run CONFIG=... TAG=your_image_tag)
endif
	@if [ ! -f $(CONFIG) ]; then echo "[ERROR] Config file $(CONFIG) not found"; exit 1; fi
	@echo "[DEPLOY] Launching superbot:$(TAG) with config $(CONFIG) in $(MODE) profile..."
	@IMAGE_TAG=superbot:$(TAG) CONFIG_PATH=$(CONFIG) docker compose --profile $(MODE) up -d
	@echo "[DEPLOY] Containers initialized. Verify status with: docker logs -f superbot_engine"

stop:
	@docker compose --profile gpu --profile cpu down