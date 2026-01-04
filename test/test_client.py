import unittest
from unittest.mock import AsyncMock, patch

from aitoolman.client import LLMLocalClient
from aitoolman.model import LLMRequest, Message


class TestLLMLocalClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            'default': {'timeout': 30, 'max_retries': 3, 'parallel': 2},
            'api': {
                'gpt-3.5-turbo': {
                    'model': 'gpt-3.5-turbo',
                    'url': 'test_url',
                    'api_type': 'openai'
                }
            }
        }

    async def test_local_client_lifecycle(self):
        async with LLMLocalClient(self.config) as client:
            self.assertIsNotNone(client.provider_manager)
            self.assertEqual(client.client_id,
                             client.client_id)  # Should be set

    async def test_make_request(self):
        async with LLMLocalClient(self.config) as client:
            request = client._make_request(
                model_name="gpt-3.5-turbo",
                messages=[Message({"role": "user", "content": "Hello"})],
                stream=False
            )

            self.assertIsInstance(request, LLMRequest)
            self.assertEqual(request.model_name, "gpt-3.5-turbo")

    @patch('aitoolman.client.LLMProviderManager')
    async def test_cancel_request(self, mock_provider_manager_class):
        mock_provider_manager = AsyncMock()
        mock_provider_manager_class.return_value = mock_provider_manager
        mock_provider_manager.cancel_request = AsyncMock()
        mock_provider_manager.initialize = AsyncMock()
        mock_provider_manager.cleanup = AsyncMock()

        async with LLMLocalClient(self.config) as client:
            client.provider_manager = mock_provider_manager

            await client.cancel("test_request_id")
            mock_provider_manager.cancel_request.assert_called_with(
                "test_request_id")

