* Idealised Ice Stream

This is a 20km long, 5km wide ice stream flowing west to east. Bed elevation decreases linearly from -500m to -540m, while ice thickness is a constant 600m.

** Case generation

This case was made using scripts/files in fice\_toolbox/cases/ice\_stream

U_obs is generated via run\_momsolve.py as in ice\_stream.sh

** Testing this case

The Taylor verification of this case proved impossible unless:

 - Glen's n = 2
 - No flotation is permitted

Therefore, a separate set of velocity 'observations' are stored for this verification process 'ice\_stream\_U\_obs\_tv.h5'

