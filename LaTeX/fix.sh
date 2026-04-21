#!/bin/bash

FILE="Sign Language Detection.tex"

echo "🔧 Fixing YOLO → MediaPipe inconsistencies in $FILE"

# 1. Fix incorrect citation usage (YOLO paper used for MediaPipe)
sed -i 's/MediaPipe \\cite{redmon2016}/MediaPipe/g' "$FILE"

# 2. Replace "object detector" phrasing
sed -i 's/object detector/hand tracking module/g' "$FILE"
sed -i 's/single-stage object detector/real-time hand tracking system/g' "$FILE"

# 3. Remove COCO/person-class nonsense
sed -i '/COCO/d' "$FILE"
sed -i '/person class/d' "$FILE"

# 4. Replace YOLO-style detection wording
sed -i 's/detection-classification pipelines using YOLO variants/landmark-based real-time hand tracking systems/g' "$FILE"

# 5. Replace incorrect Hand Detection subsection entirely
sed -i '/\\subsection{Hand Detection Module}/,/\\subsection{Classification Model (SLD)}/c\
\\subsection{Hand Detection Module}\
\
MediaPipe Hands is used for real-time hand localization and tracking. \
Unlike traditional object detection frameworks, MediaPipe employs a \
two-stage pipeline consisting of a palm detector followed by a hand \
landmark model. The system directly predicts 21 2D keypoints corresponding \
to anatomical hand joints, along with a bounding region derived from these \
landmarks.\
\
Given an input frame, MediaPipe returns:\
\\begin{itemize}\
    \\item 21 hand landmarks (normalized coordinates)\
    \\item Handedness (left/right classification)\
    \\item A region of interest (ROI) derived from landmark extents\
\\end{itemize}\
\
This approach provides fine-grained spatial information about finger \
articulation, enabling robust downstream classification even in cluttered \
backgrounds. The ROI extracted from the landmarks is used as input to the \
classification network.\
' "$FILE"

# 6. Replace YOLO mentions globally (last pass cleanup)
sed -i 's/YOLO[^ ]*/MediaPipe/g' "$FILE"

echo "✅ Done. Review changes with: git diff $FILE"