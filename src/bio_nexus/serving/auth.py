"""
Authentication, RBAC (viewer, operator, admin), and Rate Limiting.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from bio_nexus.serving.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory sliding window rate limiter: api_key -> deque of timestamps
_RATE_LIMITS: dict[str, deque[float]] = defaultdict(deque)


@dataclass
class AuthenticatedUser:
    api_key: str
    role: str  # viewer, operator, admin
    is_dev: bool = False


def get_current_user(
    raw_key: str | None = Security(api_key_header),
    auth_header: str | None = Header(None, alias="Authorization"),
) -> AuthenticatedUser:
    key = raw_key
    if not key and auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            key = parts[1]

    # Dev mode bypass if no key is provided
    if not key:
        if settings.dev_mode:
            return AuthenticatedUser(api_key="dev-default", role="admin", is_dev=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "missing_api_key", "message": "API key required in X-API-Key header"},
        )

    # Check key in configured dictionary
    role = settings.api_keys.get(key)
    if not role:
        if settings.dev_mode:
            # In dev mode, allow any key starting with dev- as operator
            if key.startswith("dev-"):
                return AuthenticatedUser(api_key=key, role="operator", is_dev=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_api_key", "message": "Supplied API key is invalid"},
        )

    # Rate limiting
    now = time.time()
    q = _RATE_LIMITS[key]
    while q and q[0] < now - 60.0:
        q.popleft()
    if len(q) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "rate_limit_exceeded", "message": "Too many requests per minute"},
        )
    q.append(now)

    return AuthenticatedUser(api_key=key, role=role, is_dev=False)


def require_role(min_role: str) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    role_hierarchy = {"viewer": 1, "operator": 2, "admin": 3}

    def role_checker(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(min_role, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "insufficient_role",
                    "message": f"Operation requires '{min_role}' role (user has '{user.role}')",
                },
            )
        return user

    return role_checker
