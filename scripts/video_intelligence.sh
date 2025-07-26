export PROJECT_ID="regal-river-463309-a3"
gcloud config set project $PROJECT_ID

gcloud services enable \
  videointelligence.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  iam.googleapis.com

gsutil mb -l us-central1 gs://video-intelligence

export SA_NAME="crowd-sensor-sa"
gcloud iam service-accounts create $SA_NAME \
  --display-name="Video Intelligence SA"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/videointelligence.admin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

gcloud iam service-accounts keys create ~/video-intel-sa.json \
  --iam-account="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export GOOGLE_APPLICATION_CREDENTIALS=~/video-intel-sa.json


bq mk --dataset ${PROJECT_ID}:video_ai

bq mk \
  --table ${PROJECT_ID}:video_ai.object_annotations \
  input_uri:STRING,segment_start:TIMESTAMP,segment_end:TIMESTAMP,entity:STRING,confidence:FLOAT,track_id:STRING,box_LEFT:FLOAT,box_TOP:FLOAT,box_RIGHT:FLOAT,box_BOTTOM:FLOAT

bq mk \
  --table ${PROJECT_ID}:video_ai.face_annotations \
  input_uri:STRING,segment_start:TIMESTAMP,segment_end:TIMESTAMP,confidence:FLOAT,box_LEFT:FLOAT,box_TOP:FLOAT,box_RIGHT:FLOAT,box_BOTTOM:FLOAT
