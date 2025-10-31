import unittest
import os
import io
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
MANIFEST_KEY = 'batch-job-manifests/multimedia-manifest.csv'

class TestBatchProcessor(unittest.TestCase):

    # Patch the Boto3 clients the script uses
    @patch('batch_processor.boto3')
    @patch('batch_processor.uuid')
    @patch('batch_processor.open', new_callable=MagicMock) # Mock the file I/O for manifest.csv
    def test_end_to_end_job_creation(self, mock_open, mock_uuid, mock_boto3):
        
        # --- Mock Setup ---
        
        # 1. Mock STS to return the account ID
        mock_sts_client = MagicMock()
        mock_sts_client.get_caller_identity.return_value = {'Account': MOCK_ACCOUNT_ID}

        # 2. Mock S3 Control to return a job ID
        mock_s3control_client = MagicMock()
        mock_s3control_client.create_job.return_value = {'JobId': MOCK_JOB_ID}
        
        # 3. Mock S3 Client to handle object listing and uploading
        mock_s3_client = MagicMock()
        
        # Define objects to be listed by S3.get_paginator
        mock_s3_objects = [
            {'Key': 'image1.jpg'},
            {'Key': 'folder/'}, # Should be skipped
            {'Key': MANIFEST_KEY}, # Should be skipped
            {'Key': 'audio/song.mp3'},
        ]
        
        # Simulate the Paginator (it returns an iterable of pages)
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{'Contents': mock_s3_objects}]
        mock_s3_client.get_paginator.return_value = mock_paginator

        # Map the mocked clients to what boto3.client returns
        mock_boto3.client.side_effect = lambda service_name, region_name: {
            's3': mock_s3_client,
            's3control': mock_s3control_client,
            'sts': mock_sts_client
        }.get(service_name)
        
        # --- Execution ---
        
        # We need to run the main block of the script (which we can't directly call
        # because of the 'if __name__ == "__main__":' guard).
        # We can just call the core functions directly and verify the results.
        
        # A. Test manifest creation
        object_count = create_manifest_file(mock_s3_client, os.environ['INPUT_BUCKET'], MANIFEST_KEY)
        
        # B. Test job creation
        job_id = create_s3_batch_job(mock_s3control_client, MOCK_ACCOUNT_ID, MANIFEST_KEY)
        
        # --- Assertions ---
        
        # 1. Assert Manifest File Contents (check what was written to the mocked 'manifest.csv')
        # Get the mock file handle used by the 'with open' block
        mock_file_handle = mock_open.return_value.__enter__.return_value
        
        # Check that the two valid objects were written
        expected_content = f"{os.environ['INPUT_BUCKET']},image1.jpg\n{os.environ['INPUT_BUCKET']},audio/song.mp3\n"
        
        # Check the writes were correct
        mock_file_handle.write.assert_any_call(f"{os.environ['INPUT_BUCKET']},image1.jpg\n")
        mock_file_handle.write.assert_any_call(f"{os.environ['INPUT_BUCKET']},audio/song.mp3\n")
        
        # Check object count
        self.assertEqual(object_count, 2, "Should have counted 2 valid objects.")
        
        # Check manifest uploaded
        mock_s3_client.upload_file.assert_called_once_with(
            "manifest.csv", os.environ['INPUT_BUCKET'], MANIFEST_KEY
        )

        # 2. Assert S3 Control Job Parameters
        mock_s3control_client.create_job.assert_called_once()
        job_call_kwargs = mock_s3control_client.create_job.call_args[1]
        
        self.assertEqual(job_call_kwargs['AccountId'], MOCK_ACCOUNT_ID)
        self.assertEqual(job_call_kwargs['Operation']['LambdaInvoke']['FunctionArn'], os.environ['LAMBDA_ARN'])
        self.assertEqual(job_call_kwargs['RoleArn'], os.environ['BATCH_ROLE_ARN'])
        
        # Check Manifest Location ARN
        expected_manifest_arn = f"arn:aws:s3:::{os.environ['INPUT_BUCKET']}/{MANIFEST_KEY}"
        self.assertEqual(job_call_kwargs['Manifest']['Location']['ObjectArn'], expected_manifest_arn)
        
        # Check the returned Job ID
        self.assertEqual(job_id, MOCK_JOB_ID)

if __name__ == '__main__':
    unittest.main()