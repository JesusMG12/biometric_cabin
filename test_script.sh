#!/bin/bash

# Start a new gnome-terminal
gnome-terminal \
  --tab --title="ROS Visualization" -- bash -c "roslaunch vehicle_platform visualization.launch; exec bash" \
  --tab --title="Biometric Camera" -- bash -c "/home/dev/biometric_cabin/demo_script.sh; exec bash"

