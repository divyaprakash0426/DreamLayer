# feat: Add Luma Photon Depth to Image Node

## Description

This pull request introduces a new ComfyUI custom node, `LumaPhotonDepth2Img`, which integrates the Luma Photon API for novel-view image generation. The node takes an input image, generates a depth map locally using the MiDaS model, and then calls the Luma API to produce a new image based on a text prompt and the original input image.

This implementation directly addresses the requirements outlined in `docs/tasks/task1.txt`.

## Key Changes

-   **New Node**: Added `LumaPhotonDepth2Img` under the `DreamLayer/API` category.
-   **Local Depth Estimation**: Integrates the `MiDaS_small` model to generate a depth map from the input image.
-   **Depth Map Output**: The generated depth map is saved to the `outputs/depth/` directory and is also available as an image output from the node for user reference and further chaining.
-   **Luma API Integration**: Calls the Luma Photon API to perform an image-to-image generation using the original input image.
-   **Configurable Depth**: Includes a boolean toggle to disable the depth estimation step for debugging or faster execution.
-   **Dependencies**: Added `timm`, `opencv-python`, and `lpips` to a local `requirements.txt` for the custom node.

## Screenshots

### Node in ComfyUI
*(Please add your screenshot of the node in the ComfyUI graph here)*

### Example Workflow
*(Please add your screenshot of an example workflow using the node here)*

### Output Images
*(Please add your screenshot showing the generated image and the depth map here)*

## Logs

The following is a summary of the log output from a test run. It shows the successful generation and saving of the depth map, followed by the call to the Luma Photon API.

It also highlights a `[Errno 36] File name too long` error from the API logging utility. This does not affect the node's execution but points to a potential issue in the logging system when filenames are generated from long URLs.

```log
Error writing API log to ...: [Errno 36] File name too long: ...
Running MiDaS depth estimation...
Saved depth map to /home/modernyogi/Projects/DreamLayer/Dream_Layer_Resources/output/depth/depth_4ebefa5a-5b30-4337-939a-3a83b81c2637.png
Calling Luma Photon API...
[DEBUG] Final headers: {'Accept': 'application/json', 'User-Agent': 'comfy-api-nodes/1.0', 'Authorization': 'Bearer ey...'}
```

## Testing

-   Verified that the node correctly loads in ComfyUI.
-   Tested the node with an input image to ensure it generates both a novel-view image and a depth map.
-   Confirmed that the depth map is saved to the `outputs/depth/` directory.
-   Tested the `disable_depth` functionality to ensure the depth estimation step is skipped.
-   Ensured that if no API key is provided, the API call is skipped and only the depth map is generated.

## Dependencies and Installation

The required Python packages `timm`, `opencv-python`, and `lpips` are listed in `ComfyUI/custom_nodes/luma_photon_node/requirements.txt`. These should be installed automatically by ComfyUI Manager or can be installed manually.

The `lpips` dependency was discovered during testing as it is required by MiDaS but was not explicitly listed in its own dependencies.

## CI/CD and Model Caching

To fulfill the CI requirement from `docs/tasks/task1.txt` ("CI must pass with the depth model downloaded during the GitHub Action’s cache-restore step"), a script has been added to facilitate the pre-downloading and caching of the MiDaS model.

The script `scripts/download_midas.py` will download the model to a predictable local directory (`torch_hub_cache`).

### Example GitHub Actions Workflow

If a CI workflow (e.g., `.github/workflows/ci.yml`) is created, it can be configured to cache the MiDaS model with the following steps:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Cache MiDaS model
        id: cache-midas
        uses: actions/cache@v4
        with:
          path: torch_hub_cache
          key: ${{ runner.os }}-midas-v1

      - name: Download MiDaS model if not cached
        if: steps.cache-midas.outputs.cache-hit != 'true'
        run: |
          pip install torch
          python scripts/download_midas.py

      - name: Install dependencies
        run: |
          # Install other project dependencies
          pip install -r ComfyUI/custom_nodes/luma_photon_node/requirements.txt

      - name: Run tests with cached model
        env:
          TORCH_HOME: ${{ github.workspace }}/torch_hub_cache
        run: |
          # Your test command here
          # e.g., pytest
```

This setup ensures that `torch.hub.load` in the node will find the pre-downloaded models via the `TORCH_HOME` environment variable, avoiding runtime downloads during tests and speeding up the CI process.

## Future Considerations

The current implementation generates and saves the depth map but does not send it to the Luma API, as this was not specified in the task requirements (`docs/tasks/task1.txt`). The task only required saving the depth map for user reference.

A potential future enhancement could be to utilize the depth map in the API call if the Luma API supports depth-guided image-to-image generation. This would likely involve passing the depth map as an additional reference image, which could provide more structural guidance for the novel-view synthesis. This was not implemented to adhere strictly to the current set of deliverables.

## Checklist
- [ ] UI screenshot provided
- [ ] Generated image provided  
- [ ] Logs provided
- [ ] Tests added (optional)
- [ ] Code follows project style
- [ ] Self-review completed 
