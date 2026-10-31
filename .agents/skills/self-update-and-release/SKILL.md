---
name: self-update-and-release
description: Constraints for changing vLLM Hub's image publishing, version tags, and in-place self-update (updates.py, updater.py, compose.yml, cd.yml). Use before touching release workflows, the Dockerfile entrypoint, healthcheck, or update logic.
---

# Releases and self-update

## Publishing

- `cd.yml` runs on push to `main` and on `v*.*.*` tags. It re-runs `ci.yml` via `workflow_call`, then pushes `ghcr.io/mitch-b/vllm-hub`.
  - Tags: `latest`, `YYYYMMDD-HHMM` (US Central build time) and `sha-*`, plus semver on version tags.
- The image embeds its version and revision as OCI labels. `/api/version` compares the local image with the registry's `latest`.

## Self-update contract

- The running hub launches a helper container from the **new** image and runs `python -m app.updater <container-id> <image> [old-image-config-json]`.
  - The helper starts the replacement, waits for `/healthz`, swaps, and rolls back on failure (about 1 s of downtime).
  - Old hubs call new updaters, so that CLI signature must stay backward compatible. Only add optional arguments.
- `/healthz` must stay fast and dependency-free; the swap and rollback depend on it.
- `HUB_AUTO_UPDATE=true` applies updates automatically, so every merge to `main` ships to those users.

## Changing this area

- Test with fakes in `src/tests/test_updates.py`. Never require a real Docker daemon or registry.
- Keep `compose.yml` env names stable (`HUB_BIND`, `HUB_AUTO_UPDATE`, `HUB_UPDATE_CHECK`, `HF_CACHE_SOURCE`, `VLLM_IMAGE`, `VLLM_PORT`). Document new ones in `.env.example` and the README configuration table.
- The `vllm-hub-state:/state` volume holds learned capacity. Don't change its path without a migration.
