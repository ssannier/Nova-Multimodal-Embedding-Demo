import unittest
import json
import os
from unittest.mock import patch, MagicMock
from io import BytesIO

# Import the handler function and constants from your lambda file
from vector_embed_lambda import lambda_handler 

# Define mock constants to simulate environment variables
MOCK_REGION = 'us-east-1'
MOCK_VECTOR_BUCKET = 'test-vector-bucket'
MOCK_VECTOR_INDEX = 'test-index'
MOCK_BEDROCK_MODEL_ID = 'amazon.nova-multimodal-embeddings-v1:0:8192'
MOCK_EMBEDDING_DIMENSION = 1024

class TestVectorEmbedLambda(unittest.TestCase):

    def setUp(self):
        # Set mock environment variables for the function
        os.environ['S3_REGION'] = MOCK_REGION
        os.environ['VECTOR_BUCKET'] = MOCK_VECTOR_BUCKET
        os.environ['VECTOR_INDEX'] = MOCK_VECTOR_INDEX
        os.environ['BEDROCK_MODEL_ID'] = MOCK_BEDROCK_MODEL_ID
        os.environ['EMBEDDING_DIMENSION'] = str(MOCK_EMBEDDING_DIMENSION)

        # 1. Define a standard S3 Batch Operations event structure
        self.mock_event = {
            'invocationId': 'test-invocation-id',
            'tasks': [
                {
                    'taskId': 'test-task-1',
                    's3Key': 'media/test_image.jpg',
                    's3BucketArn': 'arn:aws:s3:::source-bucket-1',
                }
            ]
        }
        
        # 2. Define mock file content (a small chunk of binary data)
        self.mock_file_content = b'\xff\xd8\xff\xe0' * 100 # Mock binary content for a JPG
        
        # 3. Define the mock Bedrock response
        # Nova MME returns a list of vectors, even for a single input
        self.mock_embedding = [0.1] * MOCK_EMBEDDING_DIMENSION # Mock 1024-dim vector
        self.mock_bedrock_response_body = {
            "embeddings": [{"embedding": self.mock_embedding}]
        }

    # Use the patch decorator to mock all external AWS calls
    @patch('vector_embed_lambda.s3vectors_client')
    @patch('vector_embed_lambda.bedrock_runtime')
    @patch('vector_embed_lambda.s3_client')
    def test_image_file_processing_success(self, mock_s3, mock_bedrock, mock_s3vectors):
        
        # --- Mock S3 Get Object Response ---
        # We need to simulate the s3_client.get_object call returning the file body
        mock_s3_response = {
            'Body': MagicMock(read=lambda: self.mock_file_content),
            'ContentType': 'image/jpeg' # Crucial for testing your S3 ContentType logic
        }
        mock_s3.get_object.return_value = mock_s3_response

        # --- Mock Bedrock Invoke Model Response ---
        # We simulate the bedrock_runtime.invoke_model call returning a streaming body
        mock_stream = MagicMock()
        mock_stream.read.return_value = json.dumps(self.mock_bedrock_response_body).encode('utf-8')
        mock_bedrock.invoke_model.return_value = {'body': mock_stream}

        # --- Execute the Lambda Handler ---
        response = lambda_handler(self.mock_event, None)
        
        # --- Assertions ---
        
        # 1. Check the Lambda S3 Batch response structure
        self.assertEqual(response['results'][0]['resultCode'], 'Succeeded')

        # 2. Check the Bedrock call payload (most important test)
        # Get the arguments passed to bedrock_runtime.invoke_model
        args, kwargs = mock_bedrock.invoke_model.call_args
        request_body_json = kwargs['body']
        request_body = json.loads(request_body_json)

        # Assert correct model configuration for Nova MME
        self.assertEqual(kwargs['modelId'], MOCK_BEDROCK_MODEL_ID)
        self.assertEqual(kwargs['contentType'], 'application/json')
        
        # Assert the content classification and encoding are correct
        self.assertEqual(request_body['input']['mediaType'], 'image')
        self.assertEqual(request_body['input']['encoding'], 'base64')
        
        # 3. Check the S3Vectors call
        mock_s3vectors.put_vectors.assert_called_once()
        s3vectors_call_kwargs = mock_s3vectors.put_vectors.call_args[1]

        self.assertEqual(s3vectors_call_kwargs['vectorBucketName'], MOCK_VECTOR_BUCKET)
        self.assertEqual(s3vectors_call_kwargs['indexName'], MOCK_VECTOR_INDEX)
        
        # Check the stored vector data
        stored_vector = s3vectors_call_kwargs['vectors'][0]
        self.assertEqual(stored_vector['data']['float32'], self.mock_embedding)
        self.assertEqual(stored_vector['metadata']['mime_type'], 'image/jpeg')


if __name__ == '__main__':
    unittest.main()