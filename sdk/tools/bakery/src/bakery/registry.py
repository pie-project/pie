"""Registry client for the Pie Inferlet Registry.

This module provides HTTP client functionality for interacting with
the Pie Registry API at registry.pie-project.org.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import httpx

REGISTRY_URL = "https://registry.pie-project.org/api/v1"


@dataclass
class VersionInfo:
    """Information about a specific inferlet version."""

    num: str
    checksum: str
    size_bytes: int
    description: Optional[str]
    runtime: Optional[dict[str, str]]
    parameters: Optional[dict[str, Any]]
    dependencies: Optional[dict[str, str]]
    wit: Optional[str]
    yanked: bool
    created_at: datetime
    authors: Optional[list[str]] = None
    keywords: Optional[list[str]] = None
    repository: Optional[str] = None
    readme: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "VersionInfo":
        """Create a VersionInfo from API response dict."""
        return cls(
            num=data["num"],
            checksum=data["checksum"],
            size_bytes=data["size_bytes"],
            description=data.get("description"),
            runtime=data.get("runtime"),
            parameters=data.get("parameters"),
            dependencies=data.get("dependencies"),
            wit=data.get("wit"),
            yanked=data.get("yanked", False),
            created_at=datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            ),
            authors=data.get("authors"),
            keywords=data.get("keywords"),
            repository=data.get("repository"),
            readme=data.get("readme"),
        )


@dataclass
class InferletListItem:
    """Summary info for an inferlet in search results."""

    id: str
    name: str
    description: Optional[str]
    downloads: int
    latest_version: Optional[str]
    created_at: datetime

    @classmethod
    def from_dict(cls, data: dict) -> "InferletListItem":
        """Create an InferletListItem from API response dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            downloads=data.get("downloads", 0),
            latest_version=data.get("latest_version"),
            created_at=datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            ),
        )


@dataclass
class InferletDetail:
    """Detailed information about an inferlet including all versions."""

    id: str
    name: str
    downloads: int
    created_at: datetime
    versions: list[VersionInfo]

    @classmethod
    def from_dict(cls, data: dict) -> "InferletDetail":
        """Create an InferletDetail from API response dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            downloads=data.get("downloads", 0),
            created_at=datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            ),
            versions=[VersionInfo.from_dict(v) for v in data.get("versions", [])],
        )


@dataclass
class SearchResponse:
    """Response from the search API."""

    total: int
    page: int
    per_page: int
    pages: int
    items: list[InferletListItem]

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResponse":
        """Create a SearchResponse from API response dict."""
        return cls(
            total=data["total"],
            page=data["page"],
            per_page=data["per_page"],
            pages=data["pages"],
            items=[InferletListItem.from_dict(item) for item in data.get("items", [])],
        )


@dataclass
class PublishStartRequest:
    """Request to start publishing an inferlet."""

    name: str
    version: str
    checksum: str
    size_bytes: int
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to API request dict."""
        result = {
            "name": self.name,
            "version": self.version,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class PublishStartResponse:
    """Response from starting a publish operation."""

    upload_url: str
    storage_path: str
    expires_in_seconds: int

    @classmethod
    def from_dict(cls, data: dict) -> "PublishStartResponse":
        """Create a PublishStartResponse from API response dict."""
        return cls(
            upload_url=data["upload_url"],
            storage_path=data["storage_path"],
            expires_in_seconds=data["expires_in_seconds"],
        )


@dataclass
class PublishCommitRequest:
    """Request to finalize a publish operation."""

    name: str
    version: str
    storage_path: str
    checksum: str
    size_bytes: int
    description: Optional[str] = None
    runtime: Optional[dict[str, str]] = None
    parameters: Optional[dict[str, Any]] = None
    dependencies: Optional[dict[str, str]] = None
    authors: Optional[list[str]] = None
    keywords: Optional[list[str]] = None
    repository: Optional[str] = None
    readme: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to API request dict."""
        result = {
            "name": self.name,
            "version": self.version,
            "storage_path": self.storage_path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }
        if self.description:
            result["description"] = self.description
        if self.runtime:
            result["runtime"] = self.runtime
        if self.parameters:
            result["parameters"] = self.parameters
        if self.dependencies:
            result["dependencies"] = self.dependencies
        if self.authors:
            result["authors"] = self.authors
        if self.keywords:
            result["keywords"] = self.keywords
        if self.repository:
            result["repository"] = self.repository
        if self.readme:
            result["readme"] = self.readme
        return result


@dataclass
class PublishCommitResponse:
    """Response from finalizing a publish operation."""

    id: str
    name: str
    version: str
    storage_path: str

    @classmethod
    def from_dict(cls, data: dict) -> "PublishCommitResponse":
        """Create a PublishCommitResponse from API response dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            storage_path=data["storage_path"],
        )


@dataclass
class UserInfo:
    """Information about the authenticated user."""

    id: str
    login: str
    name: Optional[str]
    avatar_url: Optional[str]
    is_superuser: bool

    @classmethod
    def from_dict(cls, data: dict) -> "UserInfo":
        """Create a UserInfo from API response dict."""
        return cls(
            id=data["id"],
            login=data["login"],
            name=data.get("name"),
            avatar_url=data.get("avatar_url"),
            is_superuser=data.get("is_superuser", False),
        )



class RegistryError(Exception):
    """Error from the registry API."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Registry error ({status_code}): {detail}")


class RegistryClient:
    """HTTP client for the Pie Registry API."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = REGISTRY_URL,
        timeout: float = 30.0,
    ):
        """Initialize the registry client.

        Args:
            token: Optional JWT token for authenticated requests.
            base_url: Base URL for the registry API.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "RegistryClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle API response, raising on errors."""
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
                if isinstance(detail, list):
                    # Validation error format
                    detail = "; ".join(
                        f"{e.get('loc', [])}: {e.get('msg', '')}" for e in detail
                    )
            except Exception:
                detail = response.text
            raise RegistryError(response.status_code, detail)
        return response.json()

    def search(
        self,
        query: str = "",
        page: int = 1,
        per_page: int = 20,
    ) -> SearchResponse:
        """Search for inferlets.

        Args:
            query: Search query string.
            page: Page number (1-indexed).
            per_page: Items per page (max 100).

        Returns:
            SearchResponse with matching inferlets.
        """
        params = {"q": query, "page": page, "per_page": per_page}

        response = self._get_client().get("/inferlets", params=params)
        return SearchResponse.from_dict(self._handle_response(response))

    def info(self, name: str) -> InferletDetail:
        """Get detailed information about an inferlet.

        Args:
            name: Inferlet name (e.g., "text-completion").

        Returns:
            InferletDetail with full metadata and versions.
        """
        response = self._get_client().get(f"/inferlets/{name}")
        return InferletDetail.from_dict(self._handle_response(response))

    def get_me(self) -> UserInfo:
        """Get information about the authenticated user.

        Returns:
            UserInfo for the current user.

        Raises:
            RegistryError: If not authenticated or token is invalid.
        """
        response = self._get_client().get("/auth/me")
        return UserInfo.from_dict(self._handle_response(response))

    def start_publish(self, request: PublishStartRequest) -> PublishStartResponse:
        """Start a publish operation.

        Validates permissions and returns a presigned URL for uploading
        the artifact directly to storage.

        Args:
            request: Publish request with package metadata.

        Returns:
            PublishStartResponse with upload URL.

        Raises:
            RegistryError: If authentication fails or request is invalid.
        """
        response = self._get_client().post("/inferlets/new", json=request.to_dict())
        return PublishStartResponse.from_dict(self._handle_response(response))

    def commit_publish(self, request: PublishCommitRequest) -> PublishCommitResponse:
        """Finalize a publish operation.

        Called after the artifact has been uploaded to the presigned URL.

        Args:
            request: Commit request with storage path.

        Returns:
            PublishCommitResponse with finalized metadata.
        """
        response = self._get_client().post(
            "/inferlets/new/commit", json=request.to_dict()
        )
        return PublishCommitResponse.from_dict(self._handle_response(response))

    def upload_artifact(self, upload_url: str, artifact_bytes: bytes) -> None:
        """Upload an artifact to the presigned URL.

        Args:
            upload_url: Presigned URL from start_publish.
            artifact_bytes: The artifact binary data.

        Raises:
            RegistryError: If upload fails.
        """
        # Use a fresh client without base_url for the presigned URL
        response = httpx.put(
            upload_url,
            content=artifact_bytes,
            headers={"Content-Type": "application/wasm"},
            timeout=300.0,  # Longer timeout for uploads
        )
        if response.status_code >= 400:
            raise RegistryError(response.status_code, f"Upload failed: {response.text}")
