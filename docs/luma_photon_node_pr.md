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

## Testing

-   Verified that the node correctly loads in ComfyUI.
-   Tested the node with an input image to ensure it generates both a novel-view image and a depth map.
-   Confirmed that the depth map is saved to the `outputs/depth/` directory.
-   Tested the `disable_depth` functionality to ensure the depth estimation step is skipped.
-   Ensured that if no API key is provided, the API call is skipped and only the depth map is generated.

## Dependencies and Installation

The required Python packages `timm`, `opencv-python`, and `lpips` are listed in `ComfyUI/custom_nodes/luma_photon_node/requirements.txt`. These should be installed automatically by ComfyUI Manager or can be installed manually.

The `lpips` dependency was discovered during testing as it is required by MiDaS but was not explicitly listed in its own dependencies.

## Future Considerations

The current implementation generates and saves the depth map but does not send it to the Luma API, as this was not specified in the task requirements (`docs/tasks/task1.txt`). The task only required saving the depth map for user reference.

A potential future enhancement could be to utilize the depth map in the API call if the Luma API supports depth-guided image-to-image generation. This would likely involve passing the depth map as an additional reference image, which could provide more structural guidance for the novel-view synthesis. This was not implemented to adhere strictly to the current set of deliverables.
