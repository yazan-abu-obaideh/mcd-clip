from decode_mcd import ContinuousTarget

BACK_TARGET = ContinuousTarget(label='Back Angle', lower_bound=5, upper_bound=45)
ARMPIT_WRIST_TARGET = ContinuousTarget(label='Armpit Angle', lower_bound=5, upper_bound=90)
KNEE_TARGET = ContinuousTarget(label='Knee Extension', lower_bound=10, upper_bound=37.5)
AERODYNAMIC_DRAG_TARGET = ContinuousTarget(label="Aerodynamic Drag", lower_bound=0, upper_bound=75)
