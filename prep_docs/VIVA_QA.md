# Viva Q and A

This file is for preparation only and stays outside `submission_project`.

## 1. What does the project do?

It converts an input image into drawable paths, orders those paths to reduce unnecessary robot travel, and exports robot-ready files for an `mBot2` with a pen attachment.

## 2. What are the 2 AI-related parts of the project?

The 2 AI-related parts are:

1. `Perception`
   Classical computer vision or ML segmentation decides what should be drawn.

2. `Planning`
   The project compares `nearest_neighbor` and `two_opt` to reduce pen-up travel.

## 3. Where exactly is AI used, and where is it not used?

AI is used in:

- the optional ML perception model
- the planning/search comparison

AI is not used in:

- image upload
- path scaling to millimeters
- plot command formatting
- CyberPi / `.mcode` export

Those parts are standard algorithmic or engineering steps.

## 4. Why do you still keep the classical mode if you already have ML?

Because the classical mode is stable, explainable, and works very well for clean high-contrast demo inputs. It is also the safest live-demo path. The ML mode is useful for comparison and for the course AI framing, but the classical mode is still valuable in practice.

## 5. Why does ML have a fallback to the classical method?

Because a robot needs a usable drawing plan, not just any prediction. If the ML mask is empty or clearly too coarse, the system falls back so the run still produces a practical result instead of failing silently.

## 6. Why is this an image-to-drawing pipeline instead of just an image filter?

Because the final goal is not only to transform pixels. The system must:

- decide what is drawable
- convert that structure to paths
- optimize the order of those paths
- export robot actions

So it is a full pipeline from image input to robot execution.

## 7. Why did you choose segmentation instead of classification?

Because the robot does not need one label for the whole image. It needs pixel-level structure showing which parts belong to drawable foreground. That makes segmentation the right ML formulation.

## 8. What model did you use?

A small `TinyUNet`, which is a compact U-Net style convolutional segmentation model.

## 9. Why did you choose a small U-Net style model?

Because it is:

- simple to explain
- lightweight enough to run locally
- suitable for binary mask prediction
- a reasonable balance between ML content and project scope

## 10. What does the classical perception mode do?

It uses image-processing rules such as:

- grayscale preparation
- thresholding
- Canny edge detection
- connected-component filtering
- color-region detection
- contour tracing
- skeleton-based tracing when needed

## 11. What is the difference between `nearest_neighbor` and `two_opt`?

`nearest_neighbor` is a greedy baseline. It always picks the next closest path.

`two_opt` starts from that baseline and then improves the route by reversing segments when that reduces pen-up travel.

## 12. Why is `two_opt` important if some examples give the same result?

Because on simple single-stroke or well-stitched examples there may be nothing left to optimize. The value of `two_opt` becomes clearer when the drawing contains many separate fragments.

## 13. What is pen-up distance?

It is the robot travel distance while the pen is lifted. This is wasted drawing time from the output point of view, so reducing it makes the route more efficient.

## 14. Why do you report both draw distance and pen-up distance?

Because they tell different things:

- `draw distance` tells how much actual drawing path exists
- `pen-up distance` tells how much non-drawing travel the robot makes

## 15. What is path cleanup, and why is it needed?

Path cleanup removes very tiny or nearly redundant geometry so the robot does not waste time on useless micro-movements. It improves physical drawing quality and makes commands more stable.

## 16. What is path stitching?

It merges consecutive paths whose endpoints already touch. This reduces unnecessary `PEN_UP` and `PEN_DOWN` toggles and creates smoother robot strokes.

## 17. Why do you scale paths to millimeters?

Because the robot works on real paper, not pixels. The system must convert image-space coordinates into physical drawing coordinates that match A4 paper and the chosen margins.

## 18. Why is the start pose important?

Because both planning and export need a known initial robot state. The project uses a defined pen-tip start position and heading so route planning and movement generation stay consistent.

## 19. Why do you model the pen offset?

Because the pen tip is not exactly at the robot center. If the export ignores that, turns and line placement drift more noticeably on paper.

## 20. Why do you generate both Python and `.mcode`?

Because the Python file is the actual generated robot program, while the `.mcode` packaging makes it easier to open the result directly in the mBlock workflow.

## 21. What libraries and tools are used?

Main software tools:

- Python
- OpenCV
- NumPy
- PyTorch
- built-in `http.server`
- HTML, CSS, JavaScript
- mBlock / CyberPi / mBot2

## 22. Why did you use the built-in Python server instead of Flask?

Because the UI only needs a local lightweight interface with a few endpoints. `http.server` was enough, simpler, and avoids extra dependency overhead.

## 23. What does the UI allow the user to do?

The UI allows the user to:

- upload an image
- choose perception and planning modes
- generate previews
- compare multiple methods on the same image
- export the final robot files

## 24. What files are generated during a normal UI run?

Main generated files:

- `edges_preview.png`
- `paths_summary.json`
- `plot_commands.txt`
- exported Python file
- exported `.mcode` file

## 25. Why is the recommended live setup `classical + two_opt`?

Because it gives the most reliable live-demo behavior on the current machine and robot setup. It keeps the perception stable while still using the stronger planning method.

## 26. What happens if the ML checkpoint is missing?

The ML mode cannot run. The UI still works, but ML comparison or ML generation requires a valid `.pt` checkpoint in `models/`.

## 27. How does the comparison mode work?

It runs the selected combinations of:

- vision mode
- path-ordering mode

Then it summarizes metrics like:

- commands count
- paths count
- draw distance
- pen-up distance
- optional mask overlap scores if a ground-truth mask is provided

## 28. Why do you have special tuning for shapes like the house, heart, and crown?

Those are small validated corrections for known demo shapes so the physical drawing closes better on the real robot. They are isolated checks and do not change unrelated drawings.

## 29. Is the project only AI, or also robotics and software engineering?

It is a mixed project:

- AI / CV for perception
- search / optimization for planning
- robotics export for execution
- UI and file handling for usability

That combination is part of the strength of the project.

## 30. What are the current limitations?

Main limitations:

- ML quality depends on the available checkpoints
- some simple examples do not strongly separate `nearest_neighbor` from `two_opt`
- final physical quality depends on calibration, pen mount, wheel motion, and surface friction

## 31. What future improvements make sense?

Possible improvements:

- train on a larger and cleaner dataset
- compare stronger segmentation models
- add more multi-fragment examples for planning evaluation
- improve robot calibration
- add better automatic validation for robot-side accuracy

## 32. If the doctor asks "why is this AI and not just image processing?", what should we say?

Short answer:

The project contains both classical image processing and AI-related components. The ML perception mode is a learned segmentation model, and the planning section compares search strategies for route optimization. So the project is not only filtering an image; it includes learned perception and optimization-based decision making inside a full robot workflow.

## 33. If the doctor asks "what is the main engineering value of your system?", what should we say?

Short answer:

The main engineering value is that the system is complete and runnable end to end. It does not stop at detecting shapes. It produces ordered paths, robot commands, and final export files that can be executed on the adapted `mBot2`.

## 34. If the doctor asks "what should I run to try the project?", what should we answer?

From the repo root:

```powershell
cd submission_project
python -m src.ui_server --open-browser --port 8010
```

Then use the browser UI to upload an image, generate a preview, and export the result.
