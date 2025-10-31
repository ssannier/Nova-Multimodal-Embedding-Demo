import boto3
import json
import os
import base64
import mimetypes

# Environment variables for Lambda
VECTOR_BUCKET = os.environ.get('VECTOR_BUCKET')
S3_REGION = os.environ['S3_REGION']
VECTOR_INDEX = os.environ.get('VECTOR_INDEX')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID')
EMBEDDING_DIMENSION = int(os.environ.get('EMBEDDING_DIMENSION'))

s3_client = boto3.client('s3', region_name=S3_REGION)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=S3_REGION)
s3vectors_client = boto3.client('s3vectors', region_name=S3_REGION)

def lambda_handler(event, context):
    invocation_id = event['invocationId']
    results = []

    for task in event['tasks']:
        task_id = task['taskId']
        s3_uri = task['s3Key']
        bucket_name = task['s3BucketArn'].split(':::')[-1]

        try:
            # Download the file content
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_uri)
            file_content = response['Body'].read()

            # Try to get the ContentType from S3
            s3_content_type = response.get('ContentType') 
            # Fallback: Use mime types if ContentType is not set on the S3 object (defaults to binary)
            if not s3_content_type or s3_content_type == 'binary/octet-stream' or s3_content_type == 'application/octet-stream':
                 content_type, _ = mimetypes.guess_type(s3_uri)
            else:
                 content_type = s3_content_type
            
            if content_type is None:
                raise ValueError(f"Could not determine data type for {s3_uri}")
            
            # Determine Nova MME Payload components
            content_data = None
            bedrock_media_type = None
            content_encoding = None

            # Nova MME requires a unified structure where mediaType is a string literal.
            if content_type.startswith('image/'):
                bedrock_media_type = 'image'
            elif content_type.startswith('audio/'):
                bedrock_media_type = 'audio'
            elif content_type.startswith('video/'):
                bedrock_media_type = 'video'
            elif content_type == 'application/pdf':
                bedrock_media_type = 'document'
            elif content_type.startswith('text/'):
                bedrock_media_type = 'text'
            else:
                raise ValueError(f"Unsupported MIME type: {content_type}")

            
            # Binary files (media and documents) must be Base64 encoded
            if bedrock_media_type in ['image', 'video', 'audio', 'document']:
                content_data = base64.b64encode(file_content).decode('utf-8')
                content_encoding = 'base64'
            # Text files should be sent as raw UTF-8 strings
            elif bedrock_media_type == 'text':
                content_data = file_content.decode('utf-8')
                content_encoding = 'utf8'

            # Construct the correct Bedrock request body for Nova MME
            request_body = {
                "input": {
                    # This must be the string literal: 'image', 'video', 'audio', 'text', or 'document'
                    "mediaType": bedrock_media_type, 
                    "encoding": content_encoding,    
                    # This must be the Base64 string or the raw text string
                    "data": content_data 
                },
                "config": {
                    "embeddingPurpose": "GENERIC_INDEX"
                }
            }

            # Invoke Bedrock Model
            bedrock_response = bedrock_runtime.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Extract embedding
            response_body = json.loads(bedrock_response['body'].read())
            # Note: The Nova MME response format uses 'embeddings' -> [0] -> 'embedding'
            embedding = response_body["embeddings"][0]["embedding"]

            # Store the vector in S3 Vector Bucket
            vector_to_store = {
                "key": s3_uri,
                "data": {"float32": embedding},
                "metadata": {
                    "source_bucket": bucket_name,
                    "mime_type": content_type 
                }
            }
            
            s3vectors_client.put_vectors(
                vectorBucketName=VECTOR_BUCKET,
                indexName=VECTOR_INDEX,
                vectors=[vector_to_store]
            )

            # Task succeeded
            results.append({
                'taskId': task_id,
                'resultCode': 'Succeeded',
                'resultString': f'Successfully embedded {s3_uri} ({content_type}) with {EMBEDDING_DIMENSION} dimensions'
            })

        except Exception as e:
            # Error handling for the S3 Batch Job
            print(f"Error processing {s3_uri}: {e}")
            results.append({
                'taskId': task_id,
                'resultCode': 'PermanentFailure',
                'resultString': str(e)
            })

    return {
        'invocationSchemaVersion': '1.0',
        'treatMissingKeysAs': 'Succeeded',
        'results': results
    }