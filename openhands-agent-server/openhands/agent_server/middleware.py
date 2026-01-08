from urllib.parse import urlparse

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from openhands.agent_server.server_details_router import update_last_execution_time


class LocalhostCORSMiddleware(CORSMiddleware):
    """Custom CORS middleware that allows any request from localhost/127.0.0.1 domains,
    while using standard CORS rules for other origins.
    """

    def __init__(self, app: ASGIApp, allow_origins: list[str]) -> None:
        super().__init__(
            app,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def is_allowed_origin(self, origin: str) -> bool:
        if origin and not self.allow_origins and not self.allow_origin_regex:
            parsed = urlparse(origin)
            hostname = parsed.hostname or ""

            # Allow any localhost/127.0.0.1 origin regardless of port
            if hostname in ["localhost", "127.0.0.1"]:
                return True

        # For missing origin or other origins, use the parent class's logic
        result: bool = super().is_allowed_origin(origin)
        return result


class ActivityTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks HTTP request activity for idle detection.

    This middleware updates the last activity timestamp on every HTTP request,
    ensuring that the runtime-api can accurately detect when the server is
    actually idle vs actively serving requests.

    This fixes the issue where runtime-api would kill active runtimes because
    it only tracked its own API calls, not the actual HTTP traffic to the pods.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[type-arg]
        # Update activity timestamp before processing the request
        update_last_execution_time()
        response = await call_next(request)
        return response
