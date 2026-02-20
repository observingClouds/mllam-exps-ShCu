#!/bin/bash
cd src
#uv run python icon_grid_coarsening.py --domain domain03 --num-levels 8 --output-path ../data/multilevel_graph_hierarchy.domain03.pkl
uv run python icon_grid_coarsening.py --domain domain03 --num-levels 8 --fully-connected --output-path ../data/multilevel_graph_hierarchy.domain03.connected.pkl

