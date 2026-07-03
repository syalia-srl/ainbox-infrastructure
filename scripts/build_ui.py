"""Launch the ainbox-builder UI on the local Docker host."""
import os
import uvicorn
from ainbox_builder.app import create_app


def main():
    repo_root = os.environ.get("BUILDER_REPO_ROOT", os.getcwd())
    app = create_app(repo_root=repo_root)
    uvicorn.run(app, host=os.environ.get("BUILDER_HOST", "0.0.0.0"),
                port=int(os.environ.get("BUILDER_PORT", "8090")))


if __name__ == "__main__":
    main()
