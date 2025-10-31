
import boto3
import os
import uuid

# Environment variables from buildspec.yml
INPUT_BUCKET = os.environ['INPUT_BUCKET']
S3_REGION = os.environ['S3_REGION']
BATCH_ROLE_ARN = os.environ['BATCH_ROLE_ARN']
LAMBDA_ARN = os.environ['LAMBDA_ARN']
# The VECTOR_BUCKET and VECTOR_INDEX are used inside Lambda fn, but defined here for context

def create_manifest_file(s3_client, bucket_name, manifest_key):
    """Lists objects in the source bucket and creates the CSV manifest file."""
    print(f"Listing objects in s3://{bucket_name}...")
    
    # Simple CSV manifest format: Bucket,Key
    manifest_content = f"{bucket_name},{manifest_key}\n" # Include manifest header if needed, but S3 Batch Ops uses bucket,key
    
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
                    if not obj["Key"].endswith('/') and obj["Key"] != manifest_key:
                        # Format is: BucketName,KeyName
                        f.write(f"{bucket_name},{obj['Key']}\n")
                        object_count += 1
    
    # Upload the manifest to S3
    s3_client.upload_file("manifest.csv", bucket_name, manifest_key)
    print(f"Created manifest with {object_count} objects and uploaded to s3://{bucket_name}/{manifest_key}")
    return object_count

def create_s3_batch_job(s3control_client, account_id, manifest_key):
    """Creates the S3 Batch Operations job to process the manifest using Lambda."""
    
    job_id = str(uuid.uuid4())
    print(f"Creating S3 Batch Job with ID: {job_id}")

    response = s3control_client.create_job(
        AccountId=account_id,
        Operation={
            'LambdaInvoke': {
                'FunctionArn': LAMBDA_ARN
            }
        },
        Report={
            'Bucket': f'arn:aws:s3:::{INPUT_BUCKET}',
            'Prefix': 'batch-job-reports',
            'Format': 'Report_CSV_20180820',
            'Enabled': True,
            'Scope': 'All' # Report on all tasks
        },
        Manifest={
            'Spec': {
                'Format': 'S3BatchOperations_CSV_20161005',
                'Fields': ['Bucket', 'Key']
            },
            'Location': {
                'ObjectArn': f'arn:aws:s3:::{INPUT_BUCKET}/{manifest_key}',
                'ETag': 'not-required-for-csv' # ETag is optional for CSV
            }
        },
        Priority=10,
        RoleArn=BATCH_ROLE_ARN,
        # The job will start in a 'Suspended' state; you can resume it later if needed
        ClientRequestToken=job_id,
        Description=f'Nova-Embeddings-Batch-Job-{job_id[:8]}'
    )
    
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
    
    object_count = create_manifest_file(s3_client, INPUT_BUCKET, MANIFEST_KEY)
    
    if object_count > 0:
        job_id = create_s3_batch_job(s3control_client, account_id, MANIFEST_KEY)
        print(f"Successfully launched S3 Batch Operations job: {job_id}")
    else:
        print("No files found to process. Skipping S3 Batch Job creation.")