#!/bin/bash

# Step 1: Extract 5-minute clips from cam6.mp4 and cam1.mp4
ffmpeg -i cam6.mp4 -t 00:05:00 -c copy cam6_5min.mp4
ffmpeg -i cam1.mp4 -t 00:05:00 -c copy cam1_5min.mp4

# Step 2: Standardize all videos to 1080p, 25 FPS, H.264 (for consistency and looping)
# Full videos
ffmpeg -i huge_crowd.avi  -r 25 -c:v libx264 -preset fast huge_crowd_std.mp4
ffmpeg -i normal_crowd.avi  -r 25 -c:v libx264 -preset fast normal_crowd_std.mp4
ffmpeg -i huge_crowd2.mpg  -r 25 -c:v libx264 -preset fast huge_crowd2_std.mp4
ffmpeg -i fight2.mpg  -r 25 -c:v libx264 -preset fast fight2_std.mp4
ffmpeg -i accident.mp4  -r 25 -c:v libx264 -preset fast accident_std.mp4
ffmpeg -i human_anomaly.mp4  -r 25 -c:v libx264 -preset fast human_anomaly_std.mp4
ffmpeg -i huge_crowd_week.mpg  -r 25 -c:v libx264 -preset fast huge_crowd_week_std.mp4

# Extracted clips
ffmpeg -i cam6_5min.mp4  -r 25 -c:v libx264 -preset fast cam6_std.mp4
ffmpeg -i cam1_5min.mp4  -r 25 -c:v libx264 -preset fast cam1_std.mp4

# Step 3: Concatenate into 4 cam angle videos
# Cam Angle 1: cam6_std + huge_crowd_std + normal_crowd_std
echo "file 'cam6_std.mp4'" > list1.txt
echo "file 'huge_crowd_std.mp4'" >> list1.txt
echo "file 'normal_crowd_std.mp4'" >> list1.txt
ffmpeg -f concat -safe 0 -i list1.txt -c copy cam_angle1.mp4

# Cam Angle 2: cam1_std + huge_crowd2_std + human_anomaly_std
echo "file 'cam1_std.mp4'" > list2.txt
echo "file 'huge_crowd2_std.mp4'" >> list2.txt
echo "file 'human_anomaly_std.mp4'" >> list2.txt
ffmpeg -f concat -safe 0 -i list2.txt -c copy cam_angle2.mp4

# Cam Angle 3: fight2_std + accident_std
echo "file 'fight2_std.mp4'" > list3.txt
echo "file 'accident_std.mp4'" >> list3.txt
ffmpeg -f concat -safe 0 -i list3.txt -c copy cam_angle3.mp4

# Cam Angle 4: huge_crowd_week_std (full, no concat needed)
cp huge_crowd_week_std.mp4 cam_angle4.mp4

# Cleanup temporary files
rm *_std.mp4 cam*_5min.mp4 list*.txt

echo "Generated: cam_angle1.mp4, cam_angle2.mp4, cam_angle3.mp4, cam_angle4.mp4"




# Step 2: Standardize all videos to 1080p, 25 FPS, H.264
# Full videos
ffmpeg -i accident_1.mp4  -r 25 -c:v libx264 -preset fast accident_1_std.mp4
ffmpeg -i fight_1.mpeg  -r 25 -c:v libx264 -preset fast fight_1_std.mp4
ffmpeg -i fight_2.mpeg  -r 25 -c:v libx264 -preset fast fight_2_std.mp4
ffmpeg -i huge_crowd2.mp4  -r 25 -c:v libx264 -preset fast huge_crowd2_std.mp4
ffmpeg -i huge_crowd3.avi  -r 25 -c:v libx264 -preset fast huge_crowd3_std.mp4
ffmpeg -i huge_crowd4.avi  -r 25 -c:v libx264 -preset fast huge_crowd4_std.mp4
ffmpeg -i huge_crowd_1.mp4  -r 25 -c:v libx264 -preset fast huge_crowd_1_std.mp4
ffmpeg -i human_anamoly_1.mp4  -r 25 -c:v libx264 -preset fast human_anamoly_1_std.mp4
ffmpeg -i normal_activity_1.mp4  -r 25 -c:v libx264 -preset fast normal_activity_1_std.mp4

# Extracted clips
ffmpeg -i cam6_5min.mp4  -r 25 -c:v libx264 -preset fast cam6_std.mp4
ffmpeg -i cam1_5min.mp4  -r 25 -c:v libx264 -preset fast cam1_std.mp4

# Step 3: Concatenate into 4 cam angle videos
# Cam Angle 1: cam6_std + huge_crowd_1_std + huge_crowd3_std + normal_activity_1_std
echo "file 'cam6_std.mp4'" > list1.txt
echo "file 'huge_crowd_1_std.mp4'" >> list1.txt
echo "file 'huge_crowd3_std.mp4'" >> list1.txt
echo "file 'normal_activity_1_std.mp4'" >> list1.txt
ffmpeg -f concat -safe 0 -i list1.txt -c copy cam_angle1.mp4

# Cam Angle 2: cam1_std + huge_crowd2_std + huge_crowd4_std + human_anamoly_1_std
echo "file 'cam1_std.mp4'" > list2.txt
echo "file 'huge_crowd2_std.mp4'" >> list2.txt
echo "file 'huge_crowd4_std.mp4'" >> list2.txt
echo "file 'human_anamoly_1_std.mp4'" >> list2.txt
ffmpeg -f concat -safe 0 -i list2.txt -c copy cam_angle2.mp4

# Cam Angle 3: fight_1_std + fight_2_std + accident_1_std
echo "file 'fight_1_std.mp4'" > list3.txt
echo "file 'fight_2_std.mp4'" >> list3.txt
echo "file 'accident_1_std.mp4'" >> list3.txt
ffmpeg -f concat -safe 0 -i list3.txt -c copy cam_angle3.mp4

# Cam Angle 4: huge_crowd_1_std (standalone for extended crowd view)
cp huge_crowd_1_std.mp4 cam_angle4.mp4

# Cleanup temporary files
rm *_std.mp4 cam*_5min.mp4 list*.txt

echo "Generated: cam_angle1.mp4, cam_angle2.mp4, cam_angle3.mp4, cam_angle4.mp4"