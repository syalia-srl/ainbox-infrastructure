import json
import os

import uvicorn

from .app import create_app
from .spec import load_spec
from .supervisor import LlamaSupervisor

_SPEC_PATH = os.environ.get("AINBOX_SPEC", "/app/config/superbot_config.json")


def main() -> None:
    with open(_SPEC_PATH, encoding="utf-8") as f:
        spec = load_spec(json.load(f))
    app = create_app(spec, LlamaSupervisor())
    uvicorn.run(app, host="0.0.0.0", port=spec.gateway_port)


if __name__ == "__main__":
    main()
