# SPOTS: Stable Placement of Objects with Reasoning in Semi-Autonomous Teleoperation Systems

## News

We are happy to announce that **SPOTS** has been accepted to ICRA2024! 😆🎉🎉
Codes will be uploaded soon.

[ [Project Page](https://joonhyung-lee.github.io/spots/) | [Paper](https://arxiv.org/abs/2309.13937) | [Video](https://joonhyung-lee.github.io/spots/) ]

Official Implementation of the paper ***SPOTS: Stable Placement of Objects with Reasoning in Semi-Autonomous Teleoperation Systems***

![fig_overview](https://github.com/joonhyung-lee/spots/raw/github-page/assets/images/fig_overview.png)

# Simulation

### Dish Rack Scene

<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
    <div style="flex: 1; text-align: center; margin: 0 10px;">
        <video width="100%" controls>
            <source src="assets/videos/scene-kitchen-white.mp4" type="video/mp4">
        </video>
        <p><strong>Small Gap Dish Rack</strong></p>
    </div>
    <div style="flex: 1; text-align: center; margin: 0 10px;">
        <video width="100%" controls>
            <source src="assets/videos/scene-kitchen-black.mp4" type="video/mp4">
        </video>
        <p><strong>Medium Gap Dish Rack</strong></p>
    </div>
    <div style="flex: 1; text-align: center; margin: 0 10px;">
        <video width="100%" controls>
            <source src="assets/videos/scene-kitchen-small.mp4" type="video/mp4">
        </video>
        <p><strong>Large Gap Dish Rack</strong></p>
    </div>
</div>

### Bookshelf Scene

<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
    <div style="flex: 1; text-align: center; margin: 0 10px;">
        <video width="100%" controls>
            <source src="assets/videos/scene-bookshelf-two-tiered.mp4" type="video/mp4">
        </video>
        <p><strong>Small Gap Dish Rack</strong></p>
    </div>
    <div style="flex: 1; text-align: center; margin: 0 10px;">
        <video width="100%" controls>
            <source src="assets/videos/scene-bookshelf-three-tiered.mp4" type="video/mp4">
        </video>
        <p><strong>Large Gap Dish Rack</strong></p>
    </div>
</div>

### Category Scene

<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
    <div style="flex: 1; text-align: center; margin: 0 10px;">
        <video width="100%" controls>
            <source src="assets/videos/scene-shelf-three-tiered.mp4" type="video/mp4">
        </video>
        <p><strong>Small Gap Dish Rack</strong></p>
    </div>
</div>


# Code Explanation
We have 3 Scenarios in this repository. Each scene consists of MJCF (MuJoCo XML) files for environment configuration and Jupyter notebooks for execution.

### A) Dish Rack Scene
The task is **To place the dish into a dish-rack**. There are three types of dish-racks available.

**Environment Files (MJCF)**:
- [scene_kitchen_dish_rack_black.xml](https://github.com/joonhyung-lee/spots/blob/main/asset/scene_kitchen_dish_rack_black.xml)
- [scene_kitchen_dish_rack_white.xml](https://github.com/joonhyung-lee/spots/blob/main/asset/scene_kitchen_dish_rack_white.xml)
- [scene_kitchen_dish_rack_small.xml](https://github.com/joonhyung-lee/spots/blob/main/asset/scene_kitchen_dish_rack_small.xml)

**Execution Files (Jupyter Notebooks)**:
You can find the execution code [here](https://github.com/joonhyung-lee/spots/tree/main/demo/scene/kitchen_with_dish):
- [Black-dish-rack.ipynb](https://github.com/joonhyung-lee/spots/blob/main/demo/scene/kitchen_with_dish/kitchen_rack_black.ipynb)
- [White-dish-rack.ipynb](https://github.com/joonhyung-lee/spots/blob/main/demo/scene/kitchen_with_dish/kitchen_rack_white.ipynb)
- [Small-dish-rack.ipynb](https://github.com/joonhyung-lee/spots/blob/main/demo/scene/kitchen_with_dish/kitchen_rack_small.ipynb)

### B) Bookshelf Scene
The task is **To place the book into a bookshelf**.

**Environment File (MJCF)**:
- [scene_office_bookshelf.xml](https://github.com/joonhyung-lee/spots/blob/main/asset/scene_office_bookshelf.xml)

**Execution File (Jupyter Notebook)**:
You can find the execution code [here](https://github.com/joonhyung-lee/spots/tree/main/demo/scene/office_booksehlf):
- [office_bookshelf.ipynb](https://github.com/joonhyung-lee/spots/blob/main/demo/scene/office_booksehlf/office_bookshelf.ipynb)

### C) Category Scene
The task is **To place the object into a shelf**.

**Environment File (MJCF)**:
- [scene_office_bookshelf.xml](https://github.com/joonhyung-lee/spots/blob/main/asset/scene_office_bookshelf.xml)

**Execution File (Jupyter Notebook)**:
You can find the execution code [here](https://github.com/joonhyung-lee/spots/tree/main/demo/scene/office_booksehlf):
- [office_bookshelf.ipynb](https://github.com/joonhyung-lee/spots/blob/main/demo/scene/office_booksehlf/office_bookshelf.ipynb)
- 
### Utility Code
- [mujoco_parser.py](https://github.com/joonhyung-lee/spots/blob/main/utils/mujoco_parser.py): Contains all functions related to the MuJoCo engine
- [util.py](https://github.com/joonhyung-lee/spots/blob/main/utils/util.py): Contains helper functions and utilities
