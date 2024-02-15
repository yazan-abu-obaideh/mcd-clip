from decode_mcd import ContinuousTarget, DesignTargets

BACK_TARGET = ContinuousTarget(label='Back Angle', lower_bound=5, upper_bound=45)

ARMPIT_WRIST_TARGET = ContinuousTarget(label='Armpit Angle', lower_bound=5, upper_bound=90)

KNEE_TARGET = ContinuousTarget(label='Knee Extension', lower_bound=10, upper_bound=37.5)
ERGO_TARGETS = DesignTargets([KNEE_TARGET, BACK_TARGET, ARMPIT_WRIST_TARGET, ])
AERO_TARGETS = DesignTargets(continuous_targets=[ContinuousTarget(label="Aerodynamic Drag",
                                                                  lower_bound=0,
                                                                  upper_bound=75)])
