import table_path_code as resto
import approach_path_planner as ap
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import cv2


# Set of tests for multi-goal legibility
test_scenarios = []


# Triangle
#length, width
dim = (600, 600)
start = (200, 300)
goals = [(600, 0), (600,600), (0, 300)]

l1 = "triangle"
r1 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

# Two and one
dim = (400, 400)
start = (400, 200)
goals = [(100, 100), (100, 300), (200, 100)]

l2 = 'two_and_one'
r2 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

# Additional
dim = (400, 400)
start = (200, 200)
goals = [(100, 100), (100, 300), (300, 300), (300, 100)]

l3 = 'square'
r3 = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=dim)

test_scenarios = [[l1, r1], [l2, r2], [l3, r3]]


for ri in range(len(test_scenarios)):
	label, scenario = test_scenarios[ri]
	ap.select_paths_and_draw(scenario, label)