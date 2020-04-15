#!/usr/bin/env python3
#
# game_keyboard_mapping.py
# Include keyboard mappings for different games, i.e.
# converting bunch of button names into MultiDiscrete action
# lists
#

# Structure:
# game_name: [
#   (possible_button_for_action1, another_possible_button_for_action1),
#   (possible_button_for_action2),
#   ...
# ]
# Action space is then MultiDiscrete(nvec=[len(buttons) for buttons in game_name]).
# This is to turn a list of pressed buttons into a fixed-size multi-discrete action,
# which is when any of the buttons is pressed (multiple buttons, because in some 
# games you can do same action with many buttons, or sometimes Linux/Windows naming
# screws up)

KEY_MAPPING = {
    "Downwell": [
        ("left shift", ),  # Jump/Fire
        ("a", "left", ),  # Move left
        ("d", "right", )  # Move right
    ],
    "CoTN_Bard": [
        ("left", ),  # Move left
        ("right", ),  # Move right
        ("down", ),  # Move down
        ("up", ),  # Move up
    ],
    "SuperHexagon": [
        ("a", "left", ),  # Move left
        ("d", "right", ),  # Move right
    ],
    "BindingOfIsaac": [
        ("left", ),
        ("right", ),
        ("up", ),
        ("down", ),
        ("a", ),  # Fire to different directions
        ("d", ),
        ("s", ),
        ("w", ),
        ("e", ),  # Place bomb
        ("q", ),  # Use item trinket
        ("space", ),  # Use item
    ],
    "BosonX": [
        ("a", "left"),
        ("d", "right"),
        ("w", "up")
    ],
    "BeamNG": [
        ("up", ),  # Accelerate
        ("down", ),  # Brake/reverse
        ("left", ),  # Turn left
        ("right", ),  # Turn right
    ],
}
