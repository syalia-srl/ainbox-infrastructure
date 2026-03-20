# Usage: make image RECIPE=recipes/rtx4060_v1.json
.PHONY: image

# Check if RECIPE is provided
ifndef RECIPE
$(error RECIPE is not set. Usage: make image RECIPE=recipes/my_recipe.json)
endif

image:
	@chmod +x build.sh
	@./build.sh $(RECIPE)