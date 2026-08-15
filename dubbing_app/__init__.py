"""MamboDubb server the local HTTP sidecar between the UI and `dubbing/`.

Layering (docs/APP_ARCHITECTURE.md):

    web UI  →  local HTTP  →  dubbing_app (this)  →  dubbing/ pipeline

This package owns jobs, HTTP and file serving and nothing else. Every call into
the pipeline goes through `dubbing_app.ops`, the single seam, so the pipeline
stays runnable headless and the app stays a second front end rather than a fork.
"""

__all__ = ["__version__"]

__version__ = "0.1.2"
