# SuperBot Infrastructure Engine

This repository contains the infrastructure as code to deploy the **SuperBot AI Engine**, a universal orchestrator optimized for running large language models (LLMs via **llama.cpp**) and speech-to-text services (STT via **faster-whisper**) in local or cloud environments.

The architecture is divided into three fundamental phases:
1. **Build:** Downloading base models and baking the Docker image (based on "recipes").
2. **Deploy:** Launching services with specific hardware parameters (based on "configurations").
3. **Adapters (LoRAs):** Dynamic injection of styles or fine-tunes on top of the base models.

---

## Phase 1: Image Build

In this phase, a JSON file (a "recipe" located in **recipes/**) is used to define **which models** will be downloaded and "baked" into the Docker image.

The build process is triggered using the **make image** command. The system will automatically assign the image the same name (**TAG**) as the recipe's JSON file.

**Example 1: Full GPU Image (RTX 4060)**
This recipe downloads heavy models like Qwen 3.5 (9B, 4B, 2B), embeddings, and STT models (tiny, small).
* **Run:** make image RECIPE=recipes/rtx4060_v1.json
* **Result:** Creates the Docker image **superbot:rtx4060_v1**.

**Example 2: Lightweight Image for VPS (CPU)**
This recipe only downloads the lightest models, making it ideal for resource-constrained environments.
* **Run:** make image RECIPE=recipes/vps_light_test.json
* **Result:** Creates the Docker image **superbot:vps_light_test**.

**Example 3: CPU Test Image**
* **Run:** make image RECIPE=recipes/vps_cpu_test.json
* **Result:** Creates the Docker image **superbot:vps_cpu_test**.

---

## Phase 2: Deploy and Execution

Once the image is built, a configuration JSON file (located in **deploy/**) is used to define **how** those models will be executed: ports, RAM/VRAM usage (**n_ctx**, **n_gpu_layers**), hardware profiles, and engine flags.

The deployment is triggered via **make run**, where you must specify the configuration (**CONFIG**), the image tag to use (**TAG**), and, optionally, the hardware environment (**MODE**).

**Example 1: Full GPU Deployment (Maximum Performance)**
Launches all models utilizing the graphics card's acceleration.
* **Run:** make run CONFIG=deploy/rtx4060_gpu.json TAG=rtx4060_v1 MODE=gpu

**Example 2: Hybrid Deployment (GPU + CPU)**
Ideal if GPU VRAM is limited. Configures parameters like **"n_gpu_layers": 0** in the JSON to delegate specific models to the CPU.
* **Run:** make run CONFIG=deploy/rtx4060_hybrid.json TAG=rtx4060_v1 MODE=gpu

**Example 3: Minimalist VPS Deployment (CPU Only)**
Launches the orchestrator in a non-GPU environment, using the Docker profile to bypass CUDA requirements.
* **Run:** make run CONFIG=deploy/vps_minimal.json TAG=vps_light_test MODE=cpu

> **Note on Logs:** When executing **make run**, your terminal will automatically attach to the container's logs. You can press **Ctrl+C** to exit the log view without shutting down the server.

---

## Phase 3: Loading Adapters (LoRAs)

The system allows you to load LoRA (Low-Rank Adaptation) adapters dynamically at deployment time, without needing to rebuild the Docker image. The **entrypoint.sh** automatically verifies the file's existence before injecting it, preventing the server from crashing due to typos in filenames.

### How to configure it?

1.  **Local Directory:** Place your **.gguf** LoRA files in a directory on your host machine (defaults to **./loras**).
2.  **Deployment File (deploy/*.json):** Add the **loras** array to the corresponding LLM node, defining the file, alias, and scale.

**JSON Configuration Example:**
* **alias:** "qwen3.5-2b"
* **port:** 9000
* **loras:**
    * { "file": "qwen35_voice_adapter.gguf", "alias": "voice_original", "scale": 1.0 }
    * { "file": "pirate_style.gguf", "alias": "voice_pirate", "scale": 0.8 }

### Examples of Execution with LoRAs

**Example 1: Using the default directory (./loras)**
If your LoRA files are already in the **loras** folder at the project root, just point to your config.
* **Run:** make run CONFIG=deploy/rtx4060_gpu-small.json TAG=rtx4060_v1 MODE=gpu

**Example 2: Using a custom external directory**
If you have your adapters in another path on your system (e.g., **/mnt/data/my_loras**), you can pass the **LORAS** variable to the Makefile to mount it into the container:
* **Run:** make run CONFIG=deploy/rtx4060_gpu-small.json TAG=rtx4060_v1 LORAS=/mnt/data/my_loras MODE=gpu

> **Fail-Safe Security:** If the JSON specifies a LoRA file that does not exist in the mounted directory, the system will log a **[WARNING]** and proceed to launch the base model without interrupting the execution.

---

## Service Management (Useful Commands)

**Stop all services (CPU and GPU profiles):**
* **Run:** make stop

**View logs manually:**
* **If launched in GPU mode:** docker logs -f superbot_engine_gpu
* **If launched in CPU mode:** docker logs -f superbot_engine_cpu

