FRAMED_TO_CLIPS_UNITS = {'SS OD': 'ssd', 'HT Length': 'Head tube length textfield', 'CS OD': 'csd',
                         'ST UX': 'Seat tube extension2', 'HT LX': 'Head tube lower extension2',
                         'HT UX': 'Head tube upper extension2', 'ST Length': 'Seat tube length',
                         'BB Drop': 'BB textfield', 'CSB Offset': 'CHAINSTAYbrdgshift', 'SS E': 'Seat stay junction0',
                         'Dropout Offset': 'Dropout spacing', 'SS Z': 'SSTopZOFFSET', 'Stack': 'Stack',
                         'CS Length': 'CS textfield', 'DT Length': 'DT Length', 'HT OD': 'Head tube diameter',
                         'SSB Offset': 'SEATSTAYbrdgshift', 'BB OD': 'BB diameter', 'DT OD': 'dtd', 'TT OD': 'ttd',
                         'BB Length': 'BB length', 'ST OD': 'Seat tube diameter'}
FRAMED_TO_CLIPS_IDENTICAL = {'HT Angle': 'Head angle', 'SSB_Include': 'SEATSTAYbrdgCheck', 'ST Angle': 'Seat angle',
                             'CSB_Include': 'CHAINSTAYbrdgCheck'}

CLIPS_IGNORED_MATERIAL = ['MATERIAL OHCLASS: BAMBOO', 'MATERIAL OHCLASS: CARBON', 'MATERIAL OHCLASS: OTHER']

BIKE_FIT_COLUMNS = [
    "DT Length", "HT Length", "HT Angle", "HT LX", "Stack",
    "ST Length", "ST Angle",
    "Seatpost LENGTH",
    "Saddle height", "Stem length", "Stem angle", "Headset spacers", "Crank length", "Handlebar style"
]

UNIQUE_BIKE_FIT_COLUMNS = [
    'Stem length',
    'Stem angle',
    'Handlebar style',
    'Crank length',
    'Headset spacers',
]

FRAMED_MATERIAL_COLUMNS = ['Material=Steel', 'Material=Aluminum', 'Material=Titanium']

FRAMED_COLUMNS = (FRAMED_MATERIAL_COLUMNS +
                  ['SSB_Include', 'CSB_Include', 'CS Length', 'BB Drop', 'Stack', 'SS E',
                   'ST Angle', 'BB OD', 'TT OD', 'HT OD', 'DT OD', 'CS OD', 'SS OD',
                   'ST OD', 'CS F', 'HT LX', 'ST UX', 'HT UX', 'HT Angle', 'HT Length',
                   'ST Length', 'BB Length', 'Dropout Offset', 'SSB OD', 'CSB OD',
                   'SSB Offset', 'CSB Offset', 'SS Z', 'SS Thickness', 'CS Thickness',
                   'TT Thickness', 'BB Thickness', 'HT Thickness', 'ST Thickness',
                   'DT Thickness', 'DT Length'])

FRAMED_CLIPS_INTERSECTION_COLUMNS = ['DT Length', 'Stack']

ONE_HOT_ENCODED_CLIPS_COLUMNS = ['MATERIAL', 'Dropout spacing style',
                                 'Head tube type', 'BELTorCHAIN',
                                 'bottle SEATTUBE0 show', 'RIM_STYLE front',
                                 'RIM_STYLE rear', 'Handlebar style',
                                 'bottle DOWNTUBE0 show', 'Stem kind',
                                 'Fork type', 'Top tube type']

CLIPS_COLUMNS = ['CS textfield', 'BB textfield', 'Stack', 'Head angle',
                 'Head tube length textfield', 'Seat stay junction0', 'Seat tube length',
                 'Seat angle', 'DT Length', 'BB diameter', 'ttd', 'dtd', 'csd', 'ssd',
                 'Chain stay position on BB', 'SSTopZOFFSET',
                 'Head tube upper extension2', 'Seat tube extension2',
                 'Head tube lower extension2', 'SEATSTAYbrdgshift', 'CHAINSTAYbrdgshift',
                 'SEATSTAYbrdgdia1', 'CHAINSTAYbrdgdia1', 'SEATSTAYbrdgCheck',
                 'CHAINSTAYbrdgCheck', 'Dropout spacing',
                 'Wall thickness Bottom Bracket', 'Wall thickness Top tube',
                 'Wall thickness Head tube', 'Wall thickness Down tube',
                 'Wall thickness Chain stay', 'Wall thickness Seat stay',
                 'Wall thickness Seat tube', 'ERD rear', 'Wheel width rear', 'BSD front',
                 'Wheel width front', 'ERD front', 'BSD rear', 'Display AEROBARS',
                 'BB length', 'Head tube diameter', 'Wheel cut', 'Seat tube diameter',
                 'Front Fender include', 'Rear Fender include', 'Number of cogs',
                 'Number of chainrings', 'Display RACK', 'FIRST color R_RGB',
                 'FIRST color G_RGB', 'FIRST color B_RGB', 'SPOKES composite front',
                 'SBLADEW front', 'SBLADEW rear', 'Saddle length', 'Saddle height',
                 'Down tube diameter', 'Seatpost LENGTH', 'MATERIAL OHCLASS: ALUMINIUM',
                 'MATERIAL OHCLASS: BAMBOO', 'MATERIAL OHCLASS: CARBON',
                 'MATERIAL OHCLASS: OTHER', 'MATERIAL OHCLASS: STEEL',
                 'MATERIAL OHCLASS: TITANIUM', 'Dropout spacing style OHCLASS: 0',
                 'Dropout spacing style OHCLASS: 1', 'Dropout spacing style OHCLASS: 2',
                 'Dropout spacing style OHCLASS: 3', 'Fork type OHCLASS: 0',
                 'Fork type OHCLASS: 1', 'Fork type OHCLASS: 2', 'Stem kind OHCLASS: 0',
                 'Stem kind OHCLASS: 1', 'Stem kind OHCLASS: 2',
                 'Handlebar style OHCLASS: 0', 'Handlebar style OHCLASS: 1',
                 'Handlebar style OHCLASS: 2', 'Head tube type OHCLASS: 0',
                 'Head tube type OHCLASS: 1', 'Head tube type OHCLASS: 2',
                 'Head tube type OHCLASS: 3', 'Top tube type OHCLASS: 0',
                 'Top tube type OHCLASS: 1', 'bottle SEATTUBE0 show OHCLASS: False',
                 'bottle SEATTUBE0 show OHCLASS: True',
                 'bottle DOWNTUBE0 show OHCLASS: False',
                 'bottle DOWNTUBE0 show OHCLASS: True', 'BELTorCHAIN OHCLASS: 0',
                 'BELTorCHAIN OHCLASS: 1', 'RIM_STYLE front OHCLASS: DISC',
                 'RIM_STYLE front OHCLASS: SPOKED', 'RIM_STYLE front OHCLASS: TRISPOKE',
                 'RIM_STYLE rear OHCLASS: DISC', 'RIM_STYLE rear OHCLASS: SPOKED',
                 'RIM_STYLE rear OHCLASS: TRISPOKE']
CLIPS_ONE_HOT_ENCODING_SEP = ' OHCLASS: '
