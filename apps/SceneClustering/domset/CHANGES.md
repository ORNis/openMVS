# CHANGES
- Updated NanoFLANN
- Trimmed some files
- Wording: computeViewSimilaity -> computeViewSimilarity
- Added some changes introduced into OpenMVG but
    - kept the nanoflann interface
    (in any case the current FLANN usage in openMVG seems to give wrong results) 
    - kept the DOMSET_USE_OPENMP FLAG 
- Changed CMAKE to play nicely with OpenMVS


TODO: ADD overlap formulation

TODO: Maybe ADD double interface for later...