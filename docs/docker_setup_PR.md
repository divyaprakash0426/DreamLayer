## Description
Add a minimal Docker-based development environment.  
Running

```bash
docker compose -f docker-compose.dev.yml up --build
```

starts the Flask backend (port 5002) and the Vite/React frontend (port 3000) with live-reload, satisfying **task-2** CI check (`curl :3000` returns HTML).

## Changes Made
- [x] **docker-compose.dev.yml** – backend + frontend services  
- [x] **docker/Dockerfile.backend.dev** – Python 3.11 slim image for backend  
- [x] **docker/Dockerfile.frontend.dev** – Node 20 alpine image for frontend  
- [x] **.dockerignore** – exclude unnecessary files from image context  
- [x] `dream_layer_frontend/package.json` – expose dev server on 0.0.0.0:3000  
- [x] **docs/docker_setup_PR.md** – this description

## Evidence Required ✅

### UI Screenshot
<!-- Paste a screenshot of the UI changes here -->
![UI Screenshot]()

### Generated Image
<!-- Paste an image generated with your changes here -->
![Generated Image]()

### Logs
<!-- Paste relevant logs that verify your changes work -->
```text
# Paste logs here
```

### Tests (Optional)
<!-- If you added tests, paste the test results here -->
```text
# Test results
```

## Future Considerations
Current Compose file only runs the main Flask API and the frontend.  
For full parity with `start_dream_layer.sh` we should:

1. Create additional services (or a single supervisor container) for  
   - `txt2img_server.py` → 5001  
   - `extras.py` → 5003  
   - `img2img_server.py` → 5004  
   - ComfyUI → 8188  
2. Re-use the same backend image (`docker/Dockerfile.backend.dev`) and mount the project source (`.:/app`) to enable live-reload across all services.  
3. Define `depends_on`/health-checks so the frontend waits until the APIs are ready.  
4. Optionally build a production-ready `docker-compose.yml` with optimized images, no host-mounts, and proper reverse-proxying.

## Checklist
- [ ] UI screenshot provided
- [ ] Generated image provided  
- [ ] Logs provided
- [ ] Tests added (optional)
- [x] Code follows project style
- [x] Self-review completed
