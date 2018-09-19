# coding=utf-8
"""
Arousing Events in Roller Coaster
For detection of those events the video "NVR_S06_run_1.mp4" was used.
- create (time-)array with events of each roller coaster (Space, Andes), first row:= event name, 2nd row := event time
- save them externally (*.txt)
"""

import numpy as np
import csv

# Init variables
n_event_space = 4  # including start
n_event_ande = 8

start_space = 49  # start in video (must be substracted later)
start_ande = 4*60 + 53  # 04:53 min:sec

# Space
space_events_names = np.array(["start", "rapid_curve_spiral", "looping", "spiral"])
space_events_times = np.array([start_space, 1*60+36, 1*60+49, 2*60+2]) - start_space
# concatenate/stack: space
space_events = np.row_stack((space_events_names, space_events_times))
# space_events = np.concatenate((space_events_names, space_events_times))
# space_events = space_events.reshape((2, -1))
# print(space_events, "\nshape:", space_events.shape)

# Ande
ande_events_names = np.array(["start", "fire", "steep_fall", "jump_1", "landing_1",
                              "fire_looping", "jump_2", "landing_2"])
ande_events_times = np.array([start_ande, 5*60+13, 5*60+17, 5*60+24, 5*60+26,
                              5*60+44, 6*60+0, 6*60+5]) - start_ande
# concatenate/stack: ande
ande_events = np.row_stack((ande_events_names, ande_events_times))
# ande_events = np.concatenate((ande_events_names, ande_events_times))
# ande_events = ande_events.reshape((2, -1))
# print(ande_events, "\nshape:", ande_events.shape)

# Save in Data-Folder
with open("../../Data/space_events.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(space_events)

with open("../../Data/ande_events.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(ande_events)
