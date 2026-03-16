"""Shared pytest fixtures for leeg-eval."""

import os

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv

load_dotenv()

LEEG_API_URL = os.getenv("LEEG_API_URL", "http://localhost:8000")
LEEG_API_TOKEN = os.getenv("LEEG_API_TOKEN", "")


@pytest_asyncio.fixture
async def leeg_client():
    """Authenticated async HTTP client pointed at LEEG_API_URL.

    Yields an httpx.AsyncClient with Authorization: Bearer header pre-set.
    Skips the test if LEEG_API_TOKEN is not configured.
    """
    if not LEEG_API_TOKEN:
        pytest.skip("LEEG_API_TOKEN not set — skipping live API test")

    headers = {"Authorization": f"Bearer {LEEG_API_TOKEN}"}
    async with httpx.AsyncClient(base_url=LEEG_API_URL, headers=headers, timeout=30.0) as client:
        yield client


@pytest_asyncio.fixture
async def api_up(leeg_client):
    """Health-check fixture — skips the entire suite if leeg-app is unreachable.

    Tests that depend on a live Leeg instance should declare this fixture.
    """
    try:
        response = await leeg_client.get("/health")
        response.raise_for_status()
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        pytest.skip(f"leeg-app is not reachable at {LEEG_API_URL}: {exc}")
