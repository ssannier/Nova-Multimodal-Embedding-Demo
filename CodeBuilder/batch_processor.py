
import boto3
import os
import uuid

# Environment variables from buildspec.yml
INPUT_BUCKET = os.environ['INPUT_BUCKET']
S3_REGION = os.environ['S3_REGION']
BATCH_ROLE_ARN = os.environ['BATCH_ROLE_ARN']
LAMBDA_ARN = os.environ['LAMBDA_ARN']
# Not pretty, but used to exclude the source files from processing
SOURCE_ZIP_KEY = 'source.zip'
# The VECTOR_BUCKET and VECTOR_INDEX are used inside Lambda fn, but defined here for context

def create_manifest_file(s3_client, bucket_name, manifest_key):
    """Lists objects in the source bucket and creates the CSV manifest file."""
    print(f"Listing objects in s3://{bucket_name}...")
    
    # Use paginator for large numbers of objects
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name)

    # Batch process S3 objects and write to manifest file
    object_count = 0
    with open("manifest.csv", "w") as f:
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    # Skip folders, and the manifest file itself if it's in the same bucket
                    # Also skip any key that ends with the SOURCE_ZIP_KEY (case-insensitive)
                    key = obj["Key"]
                    if (not key.endswith('/') and key != manifest_key and not key.lower().endswith(SOURCE_ZIP_KEY.lower())):
                        # Format is: BucketName,KeyName
                        f.write(f"{bucket_name},{key}\n")
                        object_count += 1
    
    # Upload manifest to S3
    s3_client.upload_file("manifest.csv", bucket_name, manifest_key)

    # Store ETag of newly uploaded manifest file
    head_object_response = s3_client.head_object(Bucket=bucket_name, Key=manifest_key)
    # CreateJob API requires raw hash, so strip quotes.
    etag = head_object_response['ETag'].strip('"')
    
    print(f"Created manifest with {object_count} objects and uploaded to s3://{bucket_name}/{manifest_key}")
    return object_count, etag # Return both count and etag

def create_s3_batch_job(s3control_client, account_id, manifest_key, manifest_etag):
    """Creates the S3 Batch Operations job to process the manifest using Lambda."""
    
    job_id = str(uuid.uuid4())
    print(f"Creating S3 Batch Job with ID: {job_id}")

    # Manifest ETag should be the raw hash (without surrounding quotes). The caller
    # (create_manifest_file) returns the ETag stripped of quotes, so pass it through.
    manifest_etag_for_request = manifest_etag
    job_request = {
        'AccountId': account_id,
        'ConfirmationRequired': False,
        'ClientRequestToken': job_id,
        'Operation': {
            'LambdaInvoke': {
                'FunctionArn': LAMBDA_ARN
            }
        },
        'Report': {
            # The CreateJob API expects the report bucket as an S3 ARN when calling S3Control.
            # Use the bucket ARN in the payload for real AWS calls.
            'Bucket': f'arn:aws:s3:::{INPUT_BUCKET}',
            'Prefix': 'batch-job-reports/',
            'Format': 'Report_CSV_20180820',
            'Enabled': True,
            'ReportScope': 'AllTasks',
            'ExpectedBucketOwner': account_id
        },
        'Manifest': {
            'Spec': {
                'Format': 'S3BatchOperations_CSV_20161005',
                'Fields': ['Bucket', 'Key']
            },
            'Location': {
                'ObjectArn': f'arn:aws:s3:::{INPUT_BUCKET}/{manifest_key}',
                'ETag': manifest_etag_for_request
            }
        },
        'Priority': 10,
        'RoleArn': BATCH_ROLE_ARN,
        'Description': f'Nova-Embeddings-Batch-Job-{job_id[:8]}'
    }

    # Print payload for debugging (do not leak secrets in production logs)
    print("S3 Batch create_job payload:", job_request)

    try:
        response = s3control_client.create_job(**job_request)
    except Exception as e:
        # Surface the exception and return None so callers can inspect logs
        print("Error creating S3 Batch job:", repr(e))
        raise

    print(f"S3 Batch Job created. Job ARN: {response.get('JobArn')}")
    return response.get('JobId')

# --- Main Execution ---
if __name__ == "__main__":
    
    s3_client = boto3.client('s3', region_name=S3_REGION)
    s3control_client = boto3.client('s3control', region_name=S3_REGION)
    sts_client = boto3.client('sts')
    
    # Get the AWS Account ID for the S3 Control service
    account_id = sts_client.get_caller_identity()['Account']
    
    MANIFEST_KEY = 'batch-job-manifests/multimedia-manifest.csv'
    
    # Receives both count and ETag now. woo
    object_count, manifest_etag = create_manifest_file(s3_client, INPUT_BUCKET, MANIFEST_KEY)
    
    if object_count > 0:
        job_id = create_s3_batch_job(s3control_client, account_id, MANIFEST_KEY, manifest_etag)
        print(f"Successfully launched S3 Batch Operations job: {job_id}")
    else:
        print("No files found to process. Skipping S3 Batch Job creation.")