import numpy as np
import json

level = 4  # ICON mesh zoom level
domains = {
    "domain01": {
        "cells": [slice(4766237 + (16**level) * 13, 4766237 + (16**level) * 14),
                  slice(4766237 + (16**level) * 14, 4766237 + (16**level) * 15),
                  slice(4766237 + (16**level) * 15, 4766237 + (16**level) * 16),
                  slice(4766237 + (16**level) * 16, 4766237 + (16**level) * 17),
                  slice(6034032 + (16**level) * 13, 6034032 + (16**level) * 14),
                  slice(6034032 + (16**level) * 14, 6034032 + (16**level) * 16),
                  slice(6034032 + (16**level) * 16, 6034032 + (16**level) * 17),
                  ],
    },
    "domain02": {
        "cells": [slice(6034032 + (16**level) * 1, 6034032 + (16**level) * 2),
                  slice(6034032 + (16**level) * 2, 6034032 + (16**level) * 3),
                  slice(6034032 + (16**level) * 3, 6034032 + (16**level) * 4),
                    slice(6034032 + (16**level) * 4, 6034032 + (16**level) * 5),
                    slice(6034032 + (16**level) * 9, 6034032 + (16**level) * 10),
                    slice(6034032 + (16**level) * 10, 6034032 + (16**level) * 11),
                    slice(6034032 + (16**level) * 11, 6034032 + (16**level) * 12),
                    slice(6034032 + (16**level) * 12, 6034032 + (16**level) * 13),
                  ]
    },
    "domain03": {
        "cells": [
            slice(10468729 + (16**level) * 12, 10468729 + (16**level) * 13),
            slice(10468729 + (16**level) * 13, 10468729 + (16**level) * 14),
            slice(10468729 + (16**level) * 14, 10468729 + (16**level) * 15),
            slice(10468729 + (16**level) * 15, 10468729 + (16**level) * 16),
            slice(3344491 + (16**level) * 10, 3344491 + (16**level) * 11),
            slice(3344491 + (16**level) * 11, 3344491 + (16**level) * 12),
            slice(3344491 + (16**level) * 9, 3344491 + (16**level) * 10),
            slice(3344491 + (16**level) * 8, 3344491 + (16**level) * 9),
        ]
    },
    "domain04": {
        "cells": [
            slice(10468729 + (16**level) * 4, 10468729 + (16**level) * 5),
            slice(10468729 + (16**level) * 5, 10468729 + (16**level) * 6),
            slice(10468729 + (16**level) * 6, 10468729 + (16**level) * 7),
            slice(10468729 + (16**level) * 7, 10468729 + (16**level) * 8),
            slice(10468729 + (16**level) * 8, 10468729 + (16**level) * 9),
            slice(10468729 + (16**level) * 9, 10468729 + (16**level) * 10),
            slice(10468729 + (16**level) * 10, 10468729 + (16**level) * 11),
            slice(10468729 + (16**level) * 11, 10468729 + (16**level) * 12),
        ]
    },
    "domain05": {
        "cells": [
            slice(10468729 + (16**level) * 0, 10468729 + (16**level) * 1),
            slice(10468729 + (16**level) * 1, 10468729 + (16**level) * 2),
            slice(10468729 + (16**level) * 2, 10468729 + (16**level) * 3),
            slice(10468729 + (16**level) * 3, 10468729 + (16**level) * 4),
            slice(7869523 + (16**level) *16, 7869523 + (16**level) *17),
            slice(7869523 + (16**level) *17, 7869523 + (16**level) *18),
            slice(7869523 + (16**level) *18, 7869523 + (16**level) *19),
            slice(7869523 + (16**level) *19, 7869523 + (16**level) *20),
        ]
    },
    "domain06": {
        "cells": [
            slice(6034032 + (16**level) * 5, 6034032 + (16**level) * 6),
            slice(6034032 + (16**level) * 6, 6034032 + (16**level) * 7),
            slice(6034032 + (16**level) * 7, 6034032 + (16**level) * 8),
            slice(6034032 + (16**level) * 8, 6034032 + (16**level) * 9),
            slice(4766237 + (16**level) * 9, 4766237 + (16**level) * 10),
            slice(4766237 + (16**level) * 10, 4766237 + (16**level) * 11),
            slice(4766237 + (16**level) * 11, 4766237 + (16**level) * 12),
            slice(4766237 + (16**level) * 12, 4766237 + (16**level) * 13),
        ]
    },
    "domain07": {
        "cells": [
            slice(3344491 + (16**level) * 2, 3344491 + (16**level) * 3),
            slice(3344491 + (16**level) * 3, 3344491 + (16**level) * 4),
            slice(3344491 + (16**level) * 1, 3344491 + (16**level) * 2),
            slice(3344491 + (16**level) * 0, 3344491 + (16**level) * 1),
            slice(3344491 + (16**level) * 12, 3344491 + (16**level) * 13),
            slice(3344491 + (16**level) * 13, 3344491 + (16**level) * 14),
            slice(3344491 + (16**level) * 14, 3344491 + (16**level) * 15),
            slice(3344491 + (16**level) * 15, 3344491 + (16**level) * 16),
        ]
    },
    "domain08": {
        "cells": [
            slice(3344491 + (16**level) * 5, 3344491 + (16**level) * 6),
            slice(3344491 + (16**level) * 4, 3344491 + (16**level) * 5),
            slice(3344491 + (16**level) * 6, 3344491 + (16**level) * 7),
            slice(3344491 + (16**level) * 7, 3344491 + (16**level) * 8),
            slice(10468729 + (16**level) * 16, 10468729 + (16**level) * 17),
            slice(10468729 + (16**level) * 17, 10468729 + (16**level) * 18),
            slice(10468729 + (16**level) * 18, 10468729 + (16**level) * 19),
            slice(10468729 + (16**level) * 19, 10468729 + (16**level) * 20),
        ]
    },
    "domain09": {
        "cells": [
            slice(7869523 + (16**level) * 4, 7869523 + (16**level) * 5),
            slice(7869523 + (16**level) * 5, 7869523 + (16**level) * 6),
            slice(7869523 + (16**level) * 6, 7869523 + (16**level) * 7),
            slice(7869523 + (16**level) * 7, 7869523 + (16**level) * 8),
            slice(7869523 + (16**level) * 8, 7869523 + (16**level) * 9),
            slice(7869523 + (16**level) * 9, 7869523 + (16**level) * 10),
            slice(7869523 + (16**level) * 10, 7869523 + (16**level) * 11),
            slice(7869523 + (16**level) * 11, 7869523 + (16**level) * 12),
        ]
    },
    "domain10": {
        "cells": [
            slice(2484500 + (16**level) * 8, 2484500 + (16**level) * 9),
            slice(2484500 + (16**level) * 9, 2484500 + (16**level) * 10),
            slice(2484500 + (16**level) * 10, 2484500 + (16**level) * 11),
            slice(2484500 + (16**level) * 11, 2484500 + (16**level) * 12),
            slice(4419422 + (16**level) * 0, 4419422 + (16**level) * 1),
            slice(4419422 + (16**level) * 1, 4419422 + (16**level) * 2),
            slice(4419422 + (16**level) * 2, 4419422 + (16**level) * 3),
            slice(4419422 + (16**level) * 3, 4419422 + (16**level) * 4),
        ]
    },
    "domain11": {
        "cells": [
            slice(2484500 + (16**level) * 0, 2484500 + (16**level) * 1),
            slice(2484500 + (16**level) * 1, 2484500 + (16**level) * 2),
            slice(2484500 + (16**level) * 2, 2484500 + (16**level) * 3),
            slice(2484500 + (16**level) * 3, 2484500 + (16**level) * 4),
            slice(2484500 + (16**level) * 4, 2484500 + (16**level) * 5),
            slice(2484500 + (16**level) * 5, 2484500 + (16**level) * 6),
            slice(2484500 + (16**level) * 6, 2484500 + (16**level) * 7),
            slice(2484500 + (16**level) * 7, 2484500 + (16**level) * 8),
        ]
    },
    "domain12": {
        "cells": [
            slice(7869523 + (16**level) *24, 7869523 + (16**level) *25),
            slice(7869523 + (16**level) *25, 7869523 + (16**level) *26),
            slice(7869523 + (16**level) *26, 7869523 + (16**level) *27),
            slice(7869523 + (16**level) *27, 7869523 + (16**level) *28),
            slice(4766237 + (16**level) * 1, 4766237 + (16**level) * 2),
            slice(4766237 + (16**level) * 2, 4766237 + (16**level) * 3),
            slice(4766237 + (16**level) * 3, 4766237 + (16**level) * 4),
            slice(4766237 + (16**level) * 4, 4766237 + (16**level) * 5),
        ]
    },
    "domain13": {
        "cells": [
            slice(7869523 + (16**level) * 20, 7869523 + (16**level) * 21),
            slice(7869523 + (16**level) * 21, 7869523 + (16**level) * 22),
            slice(7869523 + (16**level) * 22, 7869523 + (16**level) * 23),
            slice(7869523 + (16**level) * 23, 7869523 + (16**level) * 24),
            slice(9851502 + (16**level) * 0, 9851502 + (16**level) * 1),
            slice(9851502 + (16**level) * 1, 9851502 + (16**level) * 2),
            slice(9851502 + (16**level) * 2, 9851502 + (16**level) * 3),
            slice(9851502 + (16**level) * 3, 9851502 + (16**level) * 4),
        ]
    },
}

domains_indices = {}

for domain in domains.keys():
    slices = domains[domain]["cells"]
    domains_indices[domain] = np.hstack([np.arange(s.start, s.stop) for s in slices]).tolist()

with open("domains_indices.json", "w") as f:
    json.dump(domains_indices, f)