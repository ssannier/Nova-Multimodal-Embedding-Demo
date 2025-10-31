import unittest
import os
from unittest.mock import patch, MagicMock

# --- 1. Set Environment Variables BEFORE import ---
# Use the values from your buildspec.yml for mock consistency
os.environ['INPUT_BUCKET'] = "cic-multimedia-test"
os.environ['S3_REGION'] = "us-east-1"
os.environ['BATCH_ROLE_ARN'] = "arn:aws:iam::123456789012:role/S3BatchOpsRole"
os.environ['LAMBDA_ARN'] = "arn:aws:lambda:us-east-1:123456789012:function:NovaEmbeddingProcessor"

# Now we can safely import the module
from batch_processor import create_manifest_file, create_s3_batch_job

# Mock constants for test assertions
MOCK_ACCOUNT_ID = '987654321098'
MOCK_JOB_ID = 'test-job-id-1234'
MOCK_ETAG = 'test-etag-1234'
MANIFEST_KEY = 'batch-job-manifests/multimedia-manifest.csv'

class TestBatchProcessor(unittest.TestCase):

    @patch('batch_processor.boto3')
    @patch('batch_processor.uuid')
    @patch('batch_processor.open', new_callable=MagicMock)
    def test_end_to_end_job_creation(self, mock_open, mock_uuid, mock_boto3):
        
        # --- Mock Setup ---
        mock_uuid.uuid4.return_value = MagicMock(hex=MOCK_JOB_ID)
        
        # 1. Mock STS
        mock_sts_client = MagicMock()
        mock_sts_client.get_caller_identity.return_value = {'Account': MOCK_ACCOUNT_ID}

        # 2. Mock S3 Control
        mock_s3control_client = MagicMock()
        mock_s3control_client.create_job.return_value = {'JobId': MOCK_JOB_ID}
        
        # 3. Mock S3 Client: Listing, Uploading, and ETag fetching
        mock_s3_client = MagicMock()
        mock_s3_client.head_object.return_value = {'ETag': f'"{MOCK_ETAG}"'} # ETag is returned with quotes
        
        mock_s3_objects = [{'Key': 'image1.jpg'}, {'Key': 'audio/song.mp3'}]
        mock_paginator = MagicMock()
        # Mock the list_objects_v2 response
        mock_paginator.paginate.return_value = [{'Contents': mock_s3_objects}]
        mock_s3_client.get_paginator.return_value = mock_paginator

        mock_boto3.client.side_effect = lambda service_name, region_name: {
            's3': mock_s3_client,
            's3control': mock_s3control_client,
            'sts': mock_sts_client
        }.get(service_name)
        
        # --- Execution ---
        
        # A. Test manifest creation (Correctly captures count and ETag)
        object_count, manifest_etag = create_manifest_file(mock_s3_client, os.environ['INPUT_BUCKET'], MANIFEST_KEY)
        
        # B. Test job creation (Passes the required ETag)
        job_id = create_s3_batch_job(mock_s3control_client, MOCK_ACCOUNT_ID, MANIFEST_KEY, manifest_etag)
        
        # --- Assertions ---
        
        self.assertEqual(object_count, 2, "Should have counted 2 valid objects.")
        self.assertEqual(manifest_etag, MOCK_ETAG, "Should have returned the raw ETag.")
        self.assertEqual(job_id, MOCK_JOB_ID, "Should have returned the mocked job ID.")

        # 1. Assert S3 Control Job Parameters (Rigorously check the call)
        mock_s3control_client.create_job.assert_called_once()
        job_call_kwargs = mock_s3control_client.create_job.call_args[1]
        
        # Check all previously problematic/required fields
        self.assertFalse(job_call_kwargs['ConfirmationRequired'], "ConfirmationRequired must be False.")
        self.assertEqual(job_call_kwargs['AccountId'], MOCK_ACCOUNT_ID, "AccountId must match.")
        
        # Check Operation Block
        self.assertEqual(job_call_kwargs['Operation']['LambdaInvoke']['FunctionArn'], os.environ['LAMBDA_ARN'], "Lambda ARN must match.")

        # Check Report Block (Ensure correct bucket ARN and prefix slash)
        expected_report_bucket_arn = f"arn:aws:s3:::{os.environ['INPUT_BUCKET']}"
        self.assertEqual(job_call_kwargs['Report']['Bucket'], expected_report_bucket_arn, "Report Bucket must be the bucket ARN.")
        self.assertEqual(job_call_kwargs['Report']['Prefix'], 'batch-job-reports/', "Report Prefix must include trailing slash.")
        
        # Check Manifest Block (Ensure correct ARN format and ETag)
        expected_manifest_arn = f"arn:aws:s3:::{os.environ['INPUT_BUCKET']}/{MANIFEST_KEY}"
        self.assertEqual(job_call_kwargs['Manifest']['Location']['ObjectArn'], expected_manifest_arn, "Manifest ObjectArn must be in ARN format.")
        self.assertEqual(job_call_kwargs['Manifest']['Location']['ETag'], MOCK_ETAG, "Manifest ETag must be the raw hash.")

if __name__ == '__main__':
    unittest.main()