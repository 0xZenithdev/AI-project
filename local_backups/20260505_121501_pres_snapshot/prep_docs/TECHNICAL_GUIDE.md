# Technical Guide

This guide is for team preparation only. It stays outside `submission_project` so the presentation copy remains clean.

## 1. Project Summary

The project takes an input image, extracts drawable structure from it, converts that structure into ordered paths, and then exports robot-ready actions for an `mBot2` with a pen attachment.

High-level flow:

`image -> perception -> path extraction -> cleanup -> path ordering -> plot commands -> CyberPi Python / .mcode export`

## 2. Main Technologies

- `Python`: main implementation language.
- `OpenCV (cv2)`: image loading, thresholding, edge detection, connected components, contours, morphology, preview rendering.
- `NumPy`: array operations, geometry calculations, masks, resizing support logic.
- `PyTorch`: optional ML perception model (`TinyUNet`) for line-mask prediction.
- `HTML + CSS + JavaScript`: local browser UI.
- `http.server`: built-in Python web server for the local UI, so no Flask or Django is needed.
- `zipfile`: packages the generated robot Python file as `.mcode`.
- `mBlock / CyberPi / mBot2`: robot-side execution environment.

## 3. What Is AI In This Project

There are 2 AI-related parts in the course framing:

1. `Perception`
   Classical mode uses image-processing rules.
   ML mode uses a trained segmentation model to predict a binary line mask.

2. `Planning`
   The project compares `nearest_neighbor` against `two_opt` for path ordering.
   This is the search / optimization part.

Everything after that is execution logic:

- path cleanup
- path stitching
- coordinate scaling
- plot command generation
- robot export

## 4. Runtime Modes

### Classical mode

Used for the recommended live workflow.

What it does:

- prepares the image
- tries color-region detection for filled icons/logos
- tries foreground-mask extraction
- tries outline edge extraction
- converts the chosen binary result into drawable paths

### ML mode

Optional mode used for comparison and testing.

What it does:

- loads a trained checkpoint
- predicts a binary line mask
- cleans the mask
- converts it to paths
- falls back to the classical method if the ML result is empty or clearly unsuitable

## 5. Planning Modes

### `nearest_neighbor`

- greedy baseline
- always chooses the next closest remaining path endpoint
- can reverse a path if its end is closer than its start

### `two_opt`

- starts from the nearest-neighbor route
- locally reverses route segments when that reduces pen-up travel
- gives a stronger optimization story for the report and viva

## 6. Submission Runtime Files

These are the files still kept inside `submission_project` because the UI or export flow uses them directly.

### `submission_project/main.py`

Purpose:

- command-line entry point
- runs the full image-to-drawing flow
- writes result files

Functions:

- `resolve_cleanup_settings()`: decides cleanup values and marks them as default or manual.
- `resolve_start_pose()`: computes the initial pen-tip position and heading.
- `parse_args()`: reads CLI arguments.
- `run_pipeline()`: runs perception, scaling, cleanup, ordering, stitching, and command generation.
- `export_outputs()`: writes `edges_preview.png`, `paths_summary.json`, and `plot_commands.txt`.
- `main()`: CLI wrapper that runs the pipeline and prints a short summary.

### `submission_project/src/ui_server.py`

Purpose:

- runs the local browser interface
- accepts uploaded images
- returns previews, summaries, comparison results, and export files

Important functions:

- `parse_args()`: reads UI server options like host and port.
- `json_response()`: sends JSON back to the browser.
- `text_response()`: sends plain-text responses.
- `load_request_json()`: reads JSON request bodies.
- `sanitize_filename()`: cleans uploaded filenames.
- `make_session_id()`: creates unique run folders.
- `relative_to_repo()`: turns absolute paths into repo-relative paths.
- `build_file_url()`: creates browser links for generated files.
- `list_model_checkpoints()`: scans `models/` for available `.pt` files.
- `pick_recommended_checkpoint()`: chooses the preferred ML checkpoint.
- `build_ui_config()`: sends defaults and model information to the browser.
- `resolve_repo_path()`: blocks paths that escape the repo.
- `decode_data_url()`: decodes uploaded image data from the browser.
- `ensure_parent()`: creates parent folders before writing files.
- `render_path_preview()`: draws the planned path preview image shown in the UI.
- `render_detection_preview()`: draws the detection preview image shown in the UI.
- `build_pipeline_args()`: converts browser options into the same structure expected by `main.py`.
- `summarize_results()`: packages metrics, file links, and run info for the UI.
- `process_image_request()`: handles the normal "generate preview" request.
- `compare_experiments_request()`: runs multiple vision/planning combinations for comparison.
- `export_mcode_request()`: turns plot commands into CyberPi Python and `.mcode`.
- `UIServerHandler`: HTTP handler for `GET` and `POST` routes.
- `main()`: starts the local server.

### `submission_project/src/vision_v2.py`

Purpose:

- perception layer
- turns an image into drawable paths using either classical CV or ML

Key functions:

- `suppress_border_components()`: removes large border-touching regions such as page/background noise.
- `filter_outline_components()`: keeps meaningful outline components and drops tiny fragments.
- `filter_blob_components()`: keeps meaningful filled regions or long strokes.
- `preprocess_foreground_mask()`: builds a dark-foreground mask for classical extraction.
- `preprocess_outline_edges()`: builds a cleaner edge map for line-art style inputs.
- `preprocess_color_regions()`: detects filled colored regions that should be outlined.
- `binary_overlap_metrics()`: measures agreement between 2 binary masks.
- `is_color_region_artwork()`: detects images dominated by many filled colored regions.
- `has_large_dense_component()`: detects big dense blobs that should be outlined.
- `has_large_elongated_component()`: detects long thick bars that should become centerlines.
- `has_large_bbox_component()`: detects coarse masks that cover most of the page.
- `should_trace_ml_mask_as_filled_region()`: decides whether an ML mask should be traced as a filled boundary instead of a centerline.
- `_classical_result()`: packages classical result metadata.
- `get_drawing_paths_classical()`: full classical perception path.
- `get_ml_fallback_reason()`: explains why ML should fall back.
- `get_drawing_paths_ml()`: full ML perception path with fallback support.
- `get_drawing_paths()`: backward-compatible alias to the classical mode.

### `submission_project/src/path_tracing.py`

Purpose:

- converts binary masks into drawable geometric paths

Key functions:

- `clean_binary_line_map()`: removes tiny specks and smooths the binary mask a little.
- `skeletonize_line_map()`: thins thick strokes into centerlines.
- `_build_adjacency()`: builds graph neighbors for skeleton pixels.
- `_edge_key()`: normalizes undirected edge storage.
- `_walk_path()`: walks from one graph node to another to trace a stroke.
- `trace_skeleton_paths()`: converts the skeleton graph into raw path segments.
- `simplify_traced_path()`: simplifies traced paths while keeping visible shape.
- `simplify_closed_contour()`: simplifies a closed contour.
- `align_closed_path()`: rotates or reverses a closed path so it lines up with another one.
- `average_closed_paths()`: averages 2 aligned closed paths into one centerline-style path.
- `closed_path_area()`: computes polygon area.
- `closed_path_centroid()`: computes a simple centroid.
- `closed_path_bbox()`: gets bounding-box size.
- `resample_closed_path()`: resamples a closed path to a target number of points.
- `merge_duplicate_closed_paths()`: merges near-duplicate closed loops into one cleaner loop.
- `extract_ring_paths()`: collapses thick outlined loops into a single centerline-style path.
- `extract_elongated_component_paths()`: turns thick bars into single straight centerline paths.
- `_dedupe_open_paths()`: removes duplicated open segments.
- `extract_axis_aligned_stroke_paths()`: recovers clean vertical/horizontal stroke paths.
- `render_trace_map()`: renders paths back into an image for debugging/previews.
- `extract_filled_region_paths()`: traces filled shapes by their boundaries.
- `extract_paths_from_line_map()`: main mask-to-path routine.
- `extract_contours()`: compatibility wrapper around path extraction.
- `extract_edge_contour_paths()`: converts a thin edge map into contour paths.

### `submission_project/src/search.py`

Purpose:

- cleanup and planning layer
- converts raw paths into robot-friendlier ordered paths and text commands

Classes:

- `PlotCommand`: one `PEN_UP`, `PEN_DOWN`, or `MOVE`.
- `PathCleanupStats`: summary of path cleanup.
- `PlanMetrics`: summary of draw distance and pen-up distance.
- `PathStitchStats`: summary of path stitching.

Functions:

- `distance()`: Euclidean distance.
- `path_length()`: length of one path.
- `point_line_distance()`: point-to-line distance used for simplification.
- `collapse_short_segments()`: removes very tiny consecutive segments.
- `simplify_collinear_points()`: removes nearly straight middle points.
- `clean_path()`: runs the cleanup passes on one path.
- `cleanup_paths()`: cleans all paths and drops tiny leftovers.
- `scale_paths_to_mm()`: converts image-space paths into paper millimeters.
- `summarize_ordered_paths()`: computes draw and pen-up travel metrics.
- `stitch_ordered_paths()`: merges touching consecutive paths into longer strokes.
- `order_paths_nearest_neighbor()`: greedy baseline ordering.
- `order_paths_two_opt()`: local-search route improvement.
- `order_paths()`: dispatches between planning methods.
- `build_plot_commands()`: converts ordered paths into `PEN_UP/PEN_DOWN/MOVE`.
- `commands_to_text()`: writes the plain-text command format.

### `submission_project/src/mblock_script_generator.py`

Purpose:

- converts plot commands into primitive robot actions
- renders final CyberPi Python
- optionally packages it as `.mcode`

Core geometry helpers:

- `normalize_angle()`: keeps angles in the `(-180, 180]` range.
- `distance_mm()`: geometric distance.
- `heading_deg_between()`: heading from one point to another.
- `pen_offset_for_heading()`: converts pen offset from robot-local coordinates to world coordinates.
- `center_point_for_tip()`: computes robot center from pen-tip position.
- `append_turn_action()`: appends a turn if it is large enough to matter.
- `advance_point_for_heading()`: moves a point forward along a heading.
- `clamp_distance_mm()`: enforces minimum move sizes.
- `append_move_action()`: emits a turn and straight movement.

Stroke and tuning helpers:

- `extract_strokes()`: reconstructs drawing strokes from plot commands.
- `is_presentation_house_outline()`: recognizes the saved house demo outline.
- `apply_presentation_house_tuning()`: adjusts the house export so the robot closes it better.
- `is_presentation_heart_outline()`: recognizes the saved heart demo outline.
- `apply_presentation_heart_tuning()`: adjusts the final heart stroke.
- `is_presentation_crown_outline()`: recognizes the saved crown demo outline.
- `apply_presentation_crown_tuning()`: adjusts the crown zig-zag spacing.

Main export functions:

- `compile_actions()`: converts plot commands into primitive actions `PU`, `PD`, `TR`, and `ST`.
- `format_action()`: formats one primitive action for the generated Python file.
- `render_script()`: builds the final CyberPi Python script text.
- `write_script_outputs()`: writes `.py` and optionally `.mcode`.

### `submission_project/src/ml_line_model.py`

Purpose:

- optional ML perception model

Classes:

- `ModelConfig`: checkpoint image-size settings.
- `ConvBlock`: small repeated convolution block.
- `TinyUNet`: compact U-Net style segmentation model.

Functions:

- `load_model_from_checkpoint()`: loads weights and stored config.
- `predict_line_mask()`: runs inference and returns a binary line mask.

### `submission_project/src/image_geometry.py`

Purpose:

- shared image loading and resizing

Functions:

- `_normalize_background_color()`: adapts background color format to image channels.
- `composite_alpha_onto_background()`: flattens transparent images onto white.
- `load_image_bgr()`: loads an image as BGR.
- `load_image_rgb()`: loads an image as RGB.
- `load_image_grayscale()`: loads an image as grayscale.
- `resize_with_aspect_pad()`: resizes without stretching and pads to the target canvas.

### `submission_project/src/evaluate_pipeline.py`

Purpose:

- comparison helpers used by the UI

Functions:

- `load_binary_mask()`: loads and binarizes an optional ground-truth mask.
- `compute_mask_metrics()`: computes precision, recall, Dice, and IoU.
- `_average()`: helper average function.
- `_round_or_none()`: helper rounding function.
- `build_summary()`: groups multiple runs into planning and perception summaries.

### `submission_project/src/bridge_protocol.py`

Purpose:

- text command format shared by planning and export

Contents:

- `BridgeCommand`: lightweight command container.
- `parse_plot_command_line()`: parses one text command line.
- `load_plot_commands()`: reads the full text command file.
- `bridge_command_to_line()`: serializes one command.
- `bridge_commands_to_text()`: serializes a list of commands.
- `save_bridge_commands_as_text()`: writes serialized commands to disk.

## 7. Important Output Files

The main runtime files produced by one run are:

- `edges_preview.png`: detection preview.
- `paths_summary.json`: metrics and run metadata.
- `plot_commands.txt`: command sequence shared with export.
- `exports/<name>.py`: generated CyberPi Python file.
- `exports/<name>.mcode`: generated mBlock project archive.

## 8. Important Practical Details

- The recommended live setup is `classical + two_opt`.
- The ML mode is optional and mainly useful for comparison.
- The ML mode can fall back to classical if the predicted mask is empty or too coarse.
- Paths are scaled to A4 dimensions in millimeters before export.
- The robot export uses a pen offset because the pen tip is not located exactly at the robot center.
- The exporter includes small tuning rules for validated demo shapes like the house, heart, and crown.

## 9. Run Commands

From the repo root:

```powershell
cd submission_project
python -m src.ui_server --open-browser --port 8010
```

Optional CLI run:

```powershell
cd submission_project
python main.py --image images/presentation_cases/07_house_badge.png
```

Optional ML CLI run:

```powershell
cd submission_project
python main.py --image images/presentation_cases/07_house_badge.png --vision-mode ml --model-checkpoint models/line_model_presentation_filledmask_tuned.pt
```
