export PROJECT_ID="	regal-river-463309-a3"
export LOCATION="us-central1"
export STREAM_ID="crowd-sensor-stream"
export APP_ID="crowd-sensor-app"
export BQ_DATASET="crowd_dataset"
export BQ_TABLE="crowd_events"
export CLUSTER="application-cluster-0"
export SERVICE_ENDPOINT="visionai.googleapis.com"
export VM_NAME="crowd-sensor-vm"
export ZONE="us-central1-a"

export GOOGLE_APPLICATION_CREDENTIALS="vertexvision.json"

# Pick a fresh project ID
export PROJECT_ID="regal-river-463309-a3"
gcloud config set project $PROJECT_ID
#export PROJECT_NUMBER="877807411346"
export SA_NAME="vertex-ai"
#gcloud projects create $PROJECT_ID
#gcloud config set project $PROJECT_ID

# Enable mandatory services
gcloud services enable \
    visionai.googleapis.com \
    bigquery.googleapis.com \
    iam.googleapis.com \
    storage.googleapis.com

gcloud iam service-accounts list --project=$PROJECT_ID

# 2-2 Grant minimal roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/visionai.editor"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"


bq mk --location=US drishti_ds
bq mk \
  --table \
  --description "Crowd events pushed by Vertex AI Vision" \
  drishti_ds.crowd_events \
  ingestion_time:TIMESTAMP,application:STRING,node:STRING,annotation:STRING


--- streams and app and processors
export REGION="us-central1"
export STREAM_ID="crowd-manage"
export APP_ID="crowd-sensor-app"

# 4-1 Create the stream (video ingestion endpoint)
gcloud beta visionai streams create $STREAM_ID \
  --location=$REGION \
  --display-name="Gate 1 Live Camera"

# 4-2 Create the application graph
gcloud beta visionai applications create $APP_ID \
  --location=$REGION \
  --display-name="Crowd Sensor Agent"

# 4-3 Add graph nodes:
#  Ingestion → Occupancy Analytics → BigQuery
gcloud beta visionai applications nodes create ingestion-node \
  --application=$APP_ID --location=$REGION \
  --type="INGEST_STREAM" --stream=$STREAM_ID

gcloud beta visionai applications nodes create occ-node \
  --application=$APP_ID --location=$REGION \
  --type="OCCUPANCY_ANALYTICS" \
  --people-count=true --vehicle-count=false

gcloud beta visionai applications nodes create bq-node \
  --application=$APP_ID --location=$REGION \
  --type="BIGQUERY_SINK" \
  --dataset="drishti_ds" --table="crowd_events"

# 4-4 Wire the edges
gcloud beta visionai applications edges create \
  --application=$APP_ID --location=$REGION \
  --source=ingestion-node --target=occ-node
gcloud beta visionai applications edges create \
  --application=$APP_ID --location=$REGION \
  --source=occ-node --target=bq-node

# 4-5 Deploy (starts cluster)
gcloud beta visionai applications deploy $APP_ID --location=$REGION



export GOOGLE_APPLICATION_CREDENTIALS="vertexvision.json"

# Pick a fresh project ID
export PROJECT_ID="regal-river-463309-a3"
gcloud config set project $PROJECT_ID

export SA_NAME="crowd-sensor-sa"
gcloud iam service-accounts create $SA_NAME \
  --description="Runs Crowd Sensor Agent pipelines" \
  --display-name="Crowd Sensor Agent"


# 2-2 Grant minimal roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/visionai.editor"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

# 2-3 Create a JSON key for local use by vaictl
gcloud iam service-accounts keys create ~/crowd-sensor-sa.json \
  --iam-account="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

export GOOGLE_APPLICATION_CREDENTIALS=~/crowd-sensor-sa.json
vaictl -p $PROJECT_ID -l $REGION list streams

bq mk --location=US drishti_ds
bq mk \
  --table \
  --description "Crowd events pushed by Vertex AI Vision" \
  drishti_ds.crowd_events \
  ingestion_time:TIMESTAMP,application:STRING,node:STRING,annotation:STRING

export REGION="us-central1"
export STREAM_ID="gate1-stream"
export APP_ID="crowd-sensor-app"



# This command streams a video file to a stream. Video is looped into the stream until you stop the command.
vaictl -p regal-river-463309-a3 \
         -l us-central1 \
         -c application-cluster-0 \
         --service-endpoint visionai.googleapis.com \
send video-file to streams gate1-stream --file-path send1.mp4 --loop


# This will print packets from a stream to stdout.
# This will work for *any* stream, independent of the data type.
vaictl -p regal-river-463309-a3 \
         -l us-central1 \
         -c application-cluster-0 \
         --service-endpoint visionai.googleapis.com \
receive streams packets gate1-stream


curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d "" \
     "https://visionai.googleapis.com/v1/projects/$PROJECT_ID/locations/us-central1/applications/$APP_ID:deploy"

export OPERATION_ID="operation-1753435450395-63abd81284c79-975969c7-5f3d9f3a"
curl -X GET \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     "https://visionai.googleapis.com/v1/projects/$PROJECT_ID/locations/us-central1/operations/$OPERATION_ID"



export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT="regal-river-463309-a3"
export GOOGLE_CLOUD_LOCATION="us-central1"
export AGENT_PATH="./dristi_adk_agent" # Assuming capital_agent is in the current directory
# Set a name for your Cloud Run service (optional)
export SERVICE_NAME="dristi-adk-agent-service"

# Set an application name (optional)
export APP_NAME="dristi-adk-agent"

adk deploy cloud_run \
--project=$GOOGLE_CLOUD_PROJECT \
--region=$GOOGLE_CLOUD_LOCATION \
--service_name=$SERVICE_NAME \
--app_name=$APP_NAME \
--with_ui \
$AGENT_PATH



vaictl -p regal-river-463309-a3          -l us-central1          -c application-cluster-0        
  --service-endpoint visionai.googleapis.com send video-file to streams gate1-stream --file-path send1.mp4 --loop