#!/usr/bin/env python3
"""Network integration tests for LLMProviderManager against mock_openai_server.

Covers G02 design scenarios:
a. batch request normal response
b. stream request normal response
c. batch timeout triggers timeout_batch
d. stream timeout triggers timeout
e. model-level timeout_batch overrides [default]
plus authentication success/failure tests.
"""

import socket
import subprocess
import sys
import time
import unittest
import urllib.request
import uuid
from pathlib import Path

from aitoolman.model import LLMProviderRequest, Message
from aitoolman.provider import LLMProviderManager

MOCK_SERVER_PATH = Path(__file__).parent / "mock_llm_server.py"
CLIENT_TOKEN = "test-secret-token"
MOCK_DELAY_SLOW_BATCH = "2"
MOCK_DELAY_SLOW_STREAM = "2"


def _get_free_port() -> int:
    """Return a free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(url: str, process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Wait until the mock server health endpoint is reachable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"mock server exited early with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(url + "/", timeout=0.5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("mock server did not become ready in time")


class TestNetwork(unittest.IsolatedAsyncioTestCase):
    """Integration tests for LLMProviderManager with the mock OpenAI server."""

    @classmethod
    def setUpClass(cls):
        port = _get_free_port()
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.server = subprocess.Popen(
            [
                sys.executable,
                str(MOCK_SERVER_PATH),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--client-token", CLIENT_TOKEN,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_server(cls.base_url, cls.server)

    @classmethod
    def tearDownClass(cls):
        if cls.server.poll() is None:
            cls.server.terminate()
            try:
                cls.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server.kill()
                cls.server.wait(timeout=5)

    def _make_config(self):
        """Build a temporary provider config for the mock server."""
        return {
            "default": {
                "timeout": 0.5,
                "timeout_batch": 1.0,
                "max_retries": 0,
                "parallel": 2,
            },
            "api": {
                "mock-normal": {
                    "url": self.base_url + "/v1/chat/completions",
                    "model": "mock-normal",
                    "headers": {"Authorization": f"Bearer {CLIENT_TOKEN}"},
                },
                "mock-stream": {
                    "url": self.base_url + "/v1/chat/completions",
                    "model": "mock-stream",
                    "headers": {"Authorization": f"Bearer {CLIENT_TOKEN}"},
                },
                "mock-slow-batch": {
                    "url": self.base_url + "/v1/chat/completions",
                    "model": "mock-slow-batch",
                    "headers": {
                        "Authorization": f"Bearer {CLIENT_TOKEN}",
                        "X-Mock-Response-Delay": MOCK_DELAY_SLOW_BATCH,
                    },
                },
                "mock-slow-stream": {
                    "url": self.base_url + "/v1/chat/completions",
                    "model": "mock-slow-stream",
                    "headers": {
                        "Authorization": f"Bearer {CLIENT_TOKEN}",
                        "X-Mock-Response-Delay": MOCK_DELAY_SLOW_STREAM,
                    },
                },
                "mock-slow-batch-override": {
                    "url": self.base_url + "/v1/chat/completions",
                    "model": "mock-slow-batch",
                    "timeout_batch": 5.0,
                    "headers": {
                        "Authorization": f"Bearer {CLIENT_TOKEN}",
                        "X-Mock-Response-Delay": MOCK_DELAY_SLOW_BATCH,
                    },
                },
                "mock-auth-fail": {
                    "url": self.base_url + "/v1/chat/completions",
                    "model": "mock-normal",
                    "headers": {"Authorization": "Bearer wrong-token"},
                },
            },
        }

    async def asyncSetUp(self):
        self.config = self._make_config()
        self.manager = LLMProviderManager(self.config)
        await self.manager.initialize()

    async def asyncTearDown(self):
        await self.manager.cleanup()

    async def _request(self, model_name: str, stream: bool = False):
        request = LLMProviderRequest(
            client_id="test-client",
            context_id="test-context",
            request_id=f"{model_name}-{uuid.uuid4().hex}",
            model_name=model_name,
            messages=[Message(role="user", content="Hello")],
            stream=stream,
        )
        self.manager.process_request(request)
        return await request.response

    async def test_batch_normal_response(self):
        """Scenario a: batch request returns a normal response."""
        resp = await self._request("mock-normal", stream=False)
        self.assertEqual(resp.finish_reason, "stop")
        self.assertIsNone(resp.error_text)
        self.assertEqual(resp.response_text, "This is a mock response.")
        self.assertEqual(resp.prompt_tokens, 10)
        self.assertEqual(resp.completion_tokens, 20)

    async def test_stream_normal_response(self):
        """Scenario b: stream request accumulates SSE chunks."""
        resp = await self._request("mock-stream", stream=True)
        self.assertEqual(resp.finish_reason, "stop")
        self.assertIsNone(resp.error_text)
        self.assertEqual(resp.response_text, "Hello from mock server!")

    async def test_batch_timeout_triggers_timeout_batch(self):
        """Scenario c: slow batch response exceeds timeout_batch."""
        resp = await self._request("mock-slow-batch", stream=False)
        self.assertEqual(resp.finish_reason, "error: request")
        self.assertIsNotNone(resp.error_text)
        self.assertIn("ReadTimeout", resp.error_text or "")

    async def test_stream_timeout_triggers_timeout(self):
        """Scenario d: slow stream response exceeds timeout."""
        resp = await self._request("mock-slow-stream", stream=True)
        self.assertEqual(resp.finish_reason, "error: request")
        self.assertIsNotNone(resp.error_text)
        self.assertIn("ReadTimeout", resp.error_text or "")

    async def test_model_level_timeout_batch_overrides_default(self):
        """Scenario e: model-level timeout_batch overrides [default]."""
        # The same mock-slow-batch model with a 2s delay succeeds when the
        # model explicitly sets timeout_batch=5.0 (default is 1.0).
        resp = await self._request("mock-slow-batch-override", stream=False)
        self.assertEqual(resp.finish_reason, "stop")
        self.assertIsNone(resp.error_text)
        self.assertEqual(resp.response_text, "This is a slow batch response.")

    async def test_auth_success(self):
        """Auth: correct Authorization header is accepted."""
        resp = await self._request("mock-normal", stream=False)
        self.assertEqual(resp.finish_reason, "stop")
        self.assertIsNone(resp.error_text)

    async def test_auth_failure(self):
        """Auth: wrong Authorization header is rejected with 401."""
        resp = await self._request("mock-auth-fail", stream=False)
        self.assertEqual(resp.finish_reason, "error: request")
        self.assertIsNotNone(resp.error_text)
        self.assertIn("401", resp.error_text or "")
