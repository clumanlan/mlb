export AWS_PAGER=

MODULES_DIR=src/data/modules
BUILD_DIR=dist
REGION=us-east-2

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	pylint src/ --fail-under=7.0

# --- Lambda build + deploy ---

zip-mlb-fetch:
	mkdir -p $(BUILD_DIR)
	cp src/lambdas/daily_mlb_fetch/handler.py $(BUILD_DIR)/
	cp $(MODULES_DIR)/*.py $(BUILD_DIR)/
	cp src/shared/*.py $(BUILD_DIR)/
	cd $(BUILD_DIR) && zip daily_mlb_fetch.zip *.py
	rm $(BUILD_DIR)/*.py

zip-dkslate-fetch:
	mkdir -p $(BUILD_DIR)
	cp src/lambdas/daily_dkslate_fetch/handler.py $(BUILD_DIR)/
	cp $(MODULES_DIR)/*.py $(BUILD_DIR)/
	cd $(BUILD_DIR) && zip daily_dkslate_fetch.zip *.py
	rm $(BUILD_DIR)/*.py

zip-process-data:
	mkdir -p $(BUILD_DIR)
	cp src/lambdas/daily_process_data/handler.py $(BUILD_DIR)/
	cp $(MODULES_DIR)/*.py $(BUILD_DIR)/
	cp src/shared/*.py $(BUILD_DIR)/
	cd $(BUILD_DIR) && zip daily_process_data.zip *.py
	rm $(BUILD_DIR)/*.py

deploy-mlb-fetch: zip-mlb-fetch
	aws lambda update-function-code \
	  --region $(REGION) \
	  --function-name daily_mlb_fetch \
	  --zip-file fileb://$(BUILD_DIR)/daily_mlb_fetch.zip > /dev/null
	@echo "✓ daily_mlb_fetch deployed"

deploy-dkslate-fetch: zip-dkslate-fetch
	aws lambda update-function-code \
	  --region $(REGION) \
	  --function-name daily_dkslate_fetch \
	  --zip-file fileb://$(BUILD_DIR)/daily_dkslate_fetch.zip > /dev/null
	@echo "✓ daily_dkslate_fetch deployed"

zip-odds-fetch:
	mkdir -p $(BUILD_DIR)
	cp src/lambdas/daily_odds_fetch/handler.py $(BUILD_DIR)/
	cp $(MODULES_DIR)/*.py $(BUILD_DIR)/
	cp src/shared/*.py $(BUILD_DIR)/
	cd $(BUILD_DIR) && zip daily_odds_fetch.zip *.py
	rm $(BUILD_DIR)/*.py

zip-schedule-fetch:
	mkdir -p $(BUILD_DIR)
	cp src/lambdas/daily_schedule_fetch/handler.py $(BUILD_DIR)/
	cp src/shared/*.py $(BUILD_DIR)/
	cd $(BUILD_DIR) && zip daily_schedule_fetch.zip *.py
	rm $(BUILD_DIR)/*.py

deploy-schedule-fetch: zip-schedule-fetch
	aws lambda update-function-code \
	  --region $(REGION) \
	  --function-name daily_schedule_fetch \
	  --zip-file fileb://$(BUILD_DIR)/daily_schedule_fetch.zip > /dev/null
	@echo "✓ daily_schedule_fetch deployed"

deploy-odds-fetch: zip-odds-fetch
	aws lambda update-function-code \
	  --region $(REGION) \
	  --function-name daily_odds_fetch \
	  --zip-file fileb://$(BUILD_DIR)/daily_odds_fetch.zip > /dev/null
	@echo "✓ daily_odds_fetch deployed"

deploy-process-data: zip-process-data
	aws lambda update-function-code \
	  --region $(REGION) \
	  --function-name daily_process_data \
	  --zip-file fileb://$(BUILD_DIR)/daily_process_data.zip > /dev/null
	@echo "✓ daily_process_data deployed"

build-layer:
	rm -rf $(BUILD_DIR)/python
	mkdir -p $(BUILD_DIR)/python
	pip install MLB-StatsAPI requests \
	  --target $(BUILD_DIR)/python \
	  --quiet
	cd $(BUILD_DIR) && zip -r mlb-custom-layer.zip python/
	rm -rf $(BUILD_DIR)/python

publish-layer: build-layer
	aws lambda publish-layer-version \
	  --region $(REGION) \
	  --layer-name mlb-custom \
	  --zip-file fileb://$(BUILD_DIR)/mlb-custom-layer.zip \
	  --compatible-runtimes python3.11

clean:
	rm -rf $(BUILD_DIR)
	rm -rf src/lambdas/layers

zip-lineup-fetch:
	mkdir -p $(BUILD_DIR)
	cp src/lambdas/daily_lineup_fetch/handler.py $(BUILD_DIR)/
	cp src/shared/*.py $(BUILD_DIR)/
	cd $(BUILD_DIR) && zip daily_lineup_fetch.zip *.py
	rm $(BUILD_DIR)/*.py

deploy-lineup-fetch: zip-lineup-fetch
	aws lambda update-function-code \
	  --region $(REGION) \
	  --function-name daily_lineup_fetch \
	  --zip-file fileb://$(BUILD_DIR)/daily_lineup_fetch.zip > /dev/null
	@echo "✓ daily_lineup_fetch deployed"

# daily_feature_create bundles a nested-package closure (models.*, data.modules),
# not a flat sibling-module list like the other Lambdas — see
# src/lambdas/daily_feature_create/implementation_plan.md. No deploy target:
# this Lambda's IaC ownership is an unresolved, separate decision.
zip-feature-create:
	rm -rf $(BUILD_DIR)/feature_create
	mkdir -p $(BUILD_DIR)/feature_create
	cp src/lambdas/daily_feature_create/handler.py $(BUILD_DIR)/feature_create/
	cp src/features/transforms/batter_lineup.py $(BUILD_DIR)/feature_create/
	cp src/features/transforms/batter_pa_volume.py $(BUILD_DIR)/feature_create/
	cp src/features/transforms/data_readers.py $(BUILD_DIR)/feature_create/
	cp src/features/feature_store.yaml $(BUILD_DIR)/feature_create/
	mkdir -p $(BUILD_DIR)/feature_create/data/modules
	cp src/data/__init__.py $(BUILD_DIR)/feature_create/data/
	cp src/data/modules/__init__.py src/data/modules/preprocessing.py $(BUILD_DIR)/feature_create/data/modules/
	mkdir -p $(BUILD_DIR)/feature_create/models/hit_predictor/processing/features
	mkdir -p $(BUILD_DIR)/feature_create/models/n_pa_predictor/processing/features
	cp src/models/hit_predictor/processing/__init__.py src/models/hit_predictor/processing/schema.py $(BUILD_DIR)/feature_create/models/hit_predictor/processing/
	cp src/models/hit_predictor/processing/features/rolling_stats.py \
	   src/models/hit_predictor/processing/features/season_stats.py \
	   src/models/hit_predictor/processing/features/interaction_feats.py \
	   src/models/hit_predictor/processing/features/tier1_feats.py \
	   $(BUILD_DIR)/feature_create/models/hit_predictor/processing/features/
	cp src/models/n_pa_predictor/processing/__init__.py src/models/n_pa_predictor/processing/pipeline.py src/models/n_pa_predictor/processing/schema.py $(BUILD_DIR)/feature_create/models/n_pa_predictor/processing/
	cp src/models/n_pa_predictor/processing/features/batter_playing_time.py $(BUILD_DIR)/feature_create/models/n_pa_predictor/processing/features/
	cd $(BUILD_DIR)/feature_create && zip -r ../daily_feature_create.zip . -x "*.pyc" -x "__pycache__/*"
	rm -rf $(BUILD_DIR)/feature_create

# daily_feature_materialize is a container-image Lambda, not zip+layers — feast's
# real dependency closure (~470MB full install) doesn't fit Lambda's 250MB
# zip+layers cap. Builds the image locally only; no push (that's a separate,
# explicitly-confirmed step against a real ECR repo).
build-feature-materialize-image:
	rm -rf $(BUILD_DIR)/feature_materialize
	mkdir -p $(BUILD_DIR)/feature_materialize
	cp src/lambdas/daily_feature_materialize/handler.py $(BUILD_DIR)/feature_materialize/
	cp src/lambdas/daily_feature_materialize/requirements.txt $(BUILD_DIR)/feature_materialize/
	cp src/lambdas/daily_feature_materialize/Dockerfile $(BUILD_DIR)/feature_materialize/
	cp src/features/feature_store.yaml $(BUILD_DIR)/feature_materialize/
	cd $(BUILD_DIR)/feature_materialize && docker build -t daily_feature_materialize:local .
	rm -rf $(BUILD_DIR)/feature_materialize

deploy-all:
	$(MAKE) clean
	$(MAKE) deploy-mlb-fetch
	$(MAKE) deploy-dkslate-fetch
	$(MAKE) deploy-process-data
	$(MAKE) deploy-schedule-fetch
	$(MAKE) deploy-odds-fetch
	$(MAKE) deploy-lineup-fetch

# --- Dashboard ---

DASHBOARD_BACKEND  := $(abspath dashboard/backend)
DASHBOARD_VENV     := $(abspath .venv/bin)
DASHBOARD_PYTHON   := PYTHONPATH=$(DASHBOARD_BACKEND) $(DASHBOARD_VENV)/python
DASHBOARD_UVICORN  := PYTHONPATH=$(DASHBOARD_BACKEND) $(DASHBOARD_VENV)/uvicorn
DASHBOARD_PYTEST   := $(DASHBOARD_PYTHON) -m pytest

dashboard-dev:
	$(DASHBOARD_UVICORN) main:app --app-dir $(DASHBOARD_BACKEND) --port 8000 --reload &
	cd dashboard/frontend && npm run dev &
	sleep 3 && open http://localhost:5173

dashboard-backend:
	$(DASHBOARD_UVICORN) main:app --app-dir $(DASHBOARD_BACKEND) --port 8000 --reload

dashboard-frontend:
	cd dashboard/frontend && npm run dev

dashboard-stop:
	@pkill -f "uvicorn main:app" 2>/dev/null && echo "backend stopped" || echo "backend was not running"
	@pkill -f "vite" 2>/dev/null && echo "frontend stopped" || echo "frontend was not running"

dashboard-open:
	open http://localhost:5173

dashboard-test:
	cd $(DASHBOARD_BACKEND) && $(DASHBOARD_PYTEST) tests/ --ignore=tests/test_integration.py -v

dashboard-test-int:
	cd $(DASHBOARD_BACKEND) && $(DASHBOARD_PYTEST) tests/test_integration.py -m integration -v \
		$(if $(DATE),--date $(DATE),)
