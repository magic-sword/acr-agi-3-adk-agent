# Offline Python wheels

These unmodified `py3-none-any` wheels were downloaded from the named PyPI releases. They let the Kaggle GPU image install the local ARC runner even when Docker cannot resolve PyPI. `pip install --no-deps` uses the common scientific dependencies from the Kaggle base image. The project-specific packages are checked with imports during the build.

| Wheel | PyPI release | SHA-256 | License |
| --- | --- | --- | --- |
| arc_agi-0.9.9 | https://pypi.org/project/arc-agi/0.9.9/ | `a0536df47b5ab93af16ba708083f74261cd1b7801bb2e0802824623c04d59e50` | MIT; see `licenses/arc-agi.LICENSE` |
| arcengine-0.9.3 | https://pypi.org/project/arcengine/0.9.3/ | `5f9739d6d0055780a4581fd6fe09066bb08775c4c8212c9adcca2eb008aef59c` | MIT; see `licenses/arcengine.LICENSE` |
| flask-3.1.2 | https://pypi.org/project/Flask/3.1.2/ | `ca1d8112ec8a6158cc29ea4858963350011b5c846a414cdb7a954aa9e967d03c` | BSD-3-Clause; notice included in wheel |
| python_dotenv-1.1.1 | https://pypi.org/project/python-dotenv/1.1.1/ | `31f23644fe2602f88ff55e1f5c79ba497e01224ee7737937930c448e4d0e24dc` | BSD-3-Clause; notice included in wheel |
