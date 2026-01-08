"""Tests for the agent server middleware functionality."""

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openhands.agent_server.middleware import ActivityTrackingMiddleware
from openhands.agent_server.server_details_router import (
    _last_event_time,
    update_last_execution_time,
)


def test_activity_tracking_middleware_updates_last_activity_time():
    """Test that ActivityTrackingMiddleware updates last activity time on requests."""
    # Create a simple FastAPI app with the middleware
    app = FastAPI()
    app.add_middleware(ActivityTrackingMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    client = TestClient(app)

    # Record the initial last event time
    initial_time = _last_event_time

    # Wait a small amount to ensure time difference
    time.sleep(0.01)

    # Make a request
    response = client.get("/test")
    assert response.status_code == 200

    # Import the module-level variable again to get the updated value
    from openhands.agent_server import server_details_router

    # The last event time should have been updated
    assert server_details_router._last_event_time > initial_time


def test_activity_tracking_middleware_updates_on_every_request():
    """Test that ActivityTrackingMiddleware updates on every HTTP request."""
    # Create a simple FastAPI app with the middleware
    app = FastAPI()
    app.add_middleware(ActivityTrackingMiddleware)

    @app.get("/test1")
    async def test_endpoint1():
        return {"status": "ok"}

    @app.get("/test2")
    async def test_endpoint2():
        return {"status": "ok"}

    client = TestClient(app)

    # Make first request
    response1 = client.get("/test1")
    assert response1.status_code == 200

    from openhands.agent_server import server_details_router

    time_after_first = server_details_router._last_event_time

    # Wait a small amount
    time.sleep(0.01)

    # Make second request
    response2 = client.get("/test2")
    assert response2.status_code == 200

    # The last event time should have been updated again
    assert server_details_router._last_event_time > time_after_first


def test_activity_tracking_middleware_updates_on_error_responses():
    """Test that ActivityTrackingMiddleware updates even when endpoint returns error."""
    # Create a simple FastAPI app with the middleware
    app = FastAPI()
    app.add_middleware(ActivityTrackingMiddleware)

    @app.get("/error")
    async def error_endpoint():
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Test error")

    client = TestClient(app, raise_server_exceptions=False)

    from openhands.agent_server import server_details_router

    initial_time = server_details_router._last_event_time

    # Wait a small amount
    time.sleep(0.01)

    # Make a request that will return an error
    response = client.get("/error")
    assert response.status_code == 500

    # The last event time should still have been updated
    assert server_details_router._last_event_time > initial_time


def test_update_last_execution_time_function():
    """Test that update_last_execution_time function works correctly."""
    from openhands.agent_server import server_details_router

    initial_time = server_details_router._last_event_time

    # Wait a small amount
    time.sleep(0.01)

    # Call the function
    update_last_execution_time()

    # The time should have been updated
    assert server_details_router._last_event_time > initial_time
