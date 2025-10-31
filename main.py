import json
import base64
import time
import boto3

MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
EMBEDDING_DIMENSION = 3072

# Initialize Amazon Bedrock Runtime client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Extract embedding
response_body = json.loads(response["body"].read())
embedding = response_body["embeddings"][0]["embedding"]

print(f"Generated embedding with {len(embedding)} dimensions")

# Read and encode image
print(f"Generating image embedding with {MODEL_ID} ...")

with open("photo.jpg", "rb") as f:
    image_bytes = base64.b64encode(f.read()).decode("utf-8")

# Create embedding
request_body = {
    "taskType": "SINGLE_EMBEDDING",
    "singleEmbeddingParams": {
        "embeddingPurpose": "GENERIC_INDEX",
        "embeddingDimension": EMBEDDING_DIMENSION,
        "image": {
            "format": "jpeg",
            "source": {"bytes": image_bytes}
        },
    },
}

response = bedrock_runtime.invoke_model(
    body=json.dumps(request_body),
    modelId=MODEL_ID,
    contentType="application/json",
)

# Extract embedding
response_body = json.loads(response["body"].read())
embedding = response_body["embeddings"][0]["embedding"]

print(f"Generated embedding with {len(embedding)} dimensions")

# Initialize Amazon S3 client
s3 = boto3.client("s3", region_name="us-east-1")

print(f"Generating video embedding with {MODEL_ID} ...")

# Amazon S3 URIs
S3_VIDEO_URI = "s3://my-video-bucket/videos/presentation.mp4"
S3_EMBEDDING_DESTINATION_URI = "s3://my-embedding-destination-bucket/embeddings-output/"

# Create async embedding job for video with audio
model_input = {
    "taskType": "SEGMENTED_EMBEDDING",
    "segmentedEmbeddingParams": {
        "embeddingPurpose": "GENERIC_INDEX",
        "embeddingDimension": EMBEDDING_DIMENSION,
        "video": {
            "format": "mp4",
            "embeddingMode": "AUDIO_VIDEO_COMBINED",
            "source": {
                "s3Location": {"uri": S3_VIDEO_URI}
            },
            "segmentationConfig": {
                "durationSeconds": 15  # Segment into 15-second chunks
            },
        },
    },
}

response = bedrock_runtime.start_async_invoke(
    modelId=MODEL_ID,
    modelInput=model_input,
    outputDataConfig={
        "s3OutputDataConfig": {
            "s3Uri": S3_EMBEDDING_DESTINATION_URI
        }
    },
)

invocation_arn = response["invocationArn"]
print(f"Async job started: {invocation_arn}")

# Poll until job completes
print("\nPolling for job completion...")
while True:
    job = bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)
    status = job["status"]
    print(f"Status: {status}")

    if status != "InProgress":
        break
    time.sleep(15)

# Check if job completed successfully
if status == "Completed":
    output_s3_uri = job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    print(f"\nSuccess! Embeddings at: {output_s3_uri}")

    # Parse S3 URI to get bucket and prefix
    s3_uri_parts = output_s3_uri[5:].split("/", 1)  # Remove "s3://" prefix
    bucket = s3_uri_parts[0]
    prefix = s3_uri_parts[1] if len(s3_uri_parts) > 1 else ""

    # AUDIO_VIDEO_COMBINED mode outputs to embedding-audio-video.jsonl
    # The output_s3_uri already includes the job ID, so just append the filename
    embeddings_key = f"{prefix}/embedding-audio-video.jsonl".lstrip("/")

    print(f"Reading embeddings from: s3://{bucket}/{embeddings_key}")

    # Read and parse JSONL file
    response = s3.get_object(Bucket=bucket, Key=embeddings_key)
    content = response['Body'].read().decode('utf-8')

    embeddings = []
    for line in content.strip().split('\n'):
        if line:
            embeddings.append(json.loads(line))

    print(f"\nFound {len(embeddings)} video segments:")
    for i, segment in enumerate(embeddings):
        print(f"  Segment {i}: {segment.get('startTime', 0):.1f}s - {segment.get('endTime', 0):.1f}s")
        print(f"    Embedding dimension: {len(segment.get('embedding', []))}")
else:
    print(f"\nJob failed: {job.get('failureMessage', 'Unknown error')}")

# Initialize Amazon S3 Vectors client
s3vectors = boto3.client("s3vectors", region_name="us-east-1")

# Configuration
VECTOR_BUCKET = "my-vector-store"
INDEX_NAME = "embeddings"

# Create vector bucket and index (if they don't exist)
try:
    s3vectors.get_vector_bucket(vectorBucketName=VECTOR_BUCKET)
    print(f"Vector bucket {VECTOR_BUCKET} already exists")
except s3vectors.exceptions.NotFoundException:
    s3vectors.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
    print(f"Created vector bucket: {VECTOR_BUCKET}")

try:
    s3vectors.get_index(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME)
    print(f"Vector index {INDEX_NAME} already exists")
except s3vectors.exceptions.NotFoundException:
    s3vectors.create_index(
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        dataType="float32",
        distanceMetric="cosine"
    )
    print(f"Created index: {INDEX_NAME}")

texts = [
    "Machine learning on AWS",
    "Amazon Bedrock provides foundation models",
    "S3 Vectors enables semantic search"
]

print(f"\nGenerating embeddings for {len(texts)} texts...")

# Generate embeddings using Amazon Nova for each text
vectors = []
for text in texts:
    response = bedrock_runtime.invoke_model(
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingDimension": EMBEDDING_DIMENSION,
                "text": {"truncationMode": "END", "value": text}
            }
        }),
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response["body"].read())
    embedding = response_body["embeddings"][0]["embedding"]

    vectors.append({
        "key": f"text:{text[:50]}",  # Unique identifier
        "data": {"float32": embedding},
        "metadata": {"type": "text", "content": text}
    })
    print(f"  ✓ Generated embedding for: {text}")

# Add all vectors to store in a single call
s3vectors.put_vectors(
    vectorBucketName=VECTOR_BUCKET,
    indexName=INDEX_NAME,
    vectors=vectors
)

print(f"\nSuccessfully added {len(vectors)} vectors to the store in one put_vectors call!")