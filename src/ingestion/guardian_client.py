"""
Client for interacting with the Open Guardian API. This module defines the `GuardianClient` class,
which is responsible for making HTTP requests to the Open Guardian API and processing the
responses.
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote

import requests

from src.config.settings import Settings


@dataclass(frozen=True)
class GuardianSearchRequest:
    """Resolved query settings for a named Guardian search profile."""

    topic: str
    run_date: Optional[Any]
    page_size: int
    query: Optional[str] = None
    extra_filters: Dict[str, Any] = field(default_factory=dict)
    order_by: str = "newest"
    use_next_fallback: bool = True


@dataclass
class IterationState:
    """Mutable iteration state for enforcing item limits and continuation ids."""

    remaining: Optional[int]
    last_id: Optional[str] = None
    exhausted: bool = False


class GuardianClient:
    """
    Class representing the client for interacting with the Open Guardian API. This class is
    responsible for making HTTP requests to the API, handling authentication, and processing the
    responses. It provides methods for fetching articles based on configured query profiles.
    """

    def __init__(self, requests_per_second: float = 1.0):
        """Initialize client transport settings and configured query profiles."""
        resolved_settings = Settings.load_settings()
        self._settings = resolved_settings

        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be greater than 0")
        self.requests_per_second = requests_per_second
        self._last_request_time = 0.0

        config_values = Settings.load_ingestion_config()
        self._profiles = self._load_profiles(config_values.get("profiles"))

    @property
    def api_key(self) -> str:
        """Return configured Guardian API key."""
        return self._settings.api_key

    @property
    def base_url(self) -> str:
        """Return normalized Guardian API base URL."""
        return self._settings.base_url.rstrip("/")

    @property
    def default_page_size(self) -> int:
        """Return default page size from settings."""
        return self._settings.default_page_size

    @property
    def max_page_size(self) -> int:
        """Return max page size from settings."""
        return self._settings.max_page_size

    @property
    def timeout_seconds(self) -> int:
        """Return request timeout from settings."""
        return self._settings.timeout_seconds

    def _load_profiles(
        self,
        profile_values: Optional[Dict[str, Any]],
    ) -> Dict[str, GuardianSearchRequest]:
        """Validate and load configured query profiles from YAML."""
        if not isinstance(profile_values, dict) or not profile_values:
            raise ValueError("Ingestion config must define a non-empty 'profiles' mapping")

        resolved_profiles: Dict[str, GuardianSearchRequest] = {}
        for profile_name, raw_profile in profile_values.items():
            resolved_profiles[profile_name] = self._build_profile_request(
                profile_name=profile_name,
                raw_profile=raw_profile,
            )
        return resolved_profiles

    def _build_profile_request(
        self,
        profile_name: str,
        raw_profile: Any,
    ) -> GuardianSearchRequest:
        """Create a validated request object from one raw profile mapping."""
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Profile '{profile_name}' must be a mapping")

        topic = raw_profile.get("topic")
        if not topic:
            raise ValueError(f"Profile '{profile_name}' is missing required 'topic'")

        raw_page_size = raw_profile.get("page_size", self.default_page_size)
        page_size = self._validate_page_size(int(raw_page_size))

        extra_filters = raw_profile.get("extra_filters", {})
        if extra_filters is None:
            extra_filters = {}
        if not isinstance(extra_filters, dict):
            raise ValueError(f"Profile '{profile_name}' field 'extra_filters' must be a mapping")

        use_next_fallback = raw_profile.get("use_next_fallback", True)
        if not isinstance(use_next_fallback, bool):
            raise ValueError(
                f"Profile '{profile_name}' field 'use_next_fallback' must be a boolean"
            )

        return GuardianSearchRequest(
            topic=str(topic),
            run_date=raw_profile.get("run_date"),
            page_size=page_size,
            query=raw_profile.get("query"),
            extra_filters=dict(extra_filters),
            order_by=str(raw_profile.get("order_by", "newest")),
            use_next_fallback=use_next_fallback,
        )

    def _get_profile_request(self, profile: str) -> GuardianSearchRequest:
        """Return a configured request profile or raise a helpful error."""
        if not profile:
            raise ValueError("profile must not be empty")

        request = self._profiles.get(profile)
        if request is None:
            raise ValueError(f"Profile '{profile}' is not configured")
        return request

    def _default_date(self) -> str:
        """Return today's date in UTC, formatted as YYYY-MM-DD."""
        return datetime.now(timezone.utc).date().isoformat()

    def _normalize_date(self, run_date: Optional[Any]) -> str:
        """Normalize supported date inputs to YYYY-MM-DD."""
        if run_date is None:
            return self._default_date()
        if isinstance(run_date, datetime):
            return run_date.astimezone(timezone.utc).date().isoformat()
        if isinstance(run_date, date):
            return run_date.isoformat()
        if isinstance(run_date, str):
            try:
                return date.fromisoformat(run_date).isoformat()
            except ValueError as exc:
                raise ValueError("run_date must be in YYYY-MM-DD format") from exc
        raise ValueError("run_date must be a date, datetime, string, or None")

    def _validate_page_size(self, page_size: int) -> int:
        """Validate and return a supported Guardian API page size."""
        if page_size < 1 or page_size > self.max_page_size:
            raise ValueError(f"page_size must be between 1 and {self.max_page_size}")
        return page_size

    def _throttle(self) -> None:
        """Enforce a minimum delay between requests to avoid rate limits."""
        min_interval = 1.0 / self.requests_per_second
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _request_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request and parse the response body as JSON."""
        self._throttle()
        url = f"{self.base_url}{path}"
        try:
            response = requests.get(
                url=url,
                params=params or {},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text if exc.response is not None else ""
            raise RuntimeError(f"Guardian API HTTP error {status_code}: {body}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Guardian API connection error: {exc}") from exc

    def _extract_response_or_raise(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate the top-level response object."""
        if not isinstance(payload, dict) or "response" not in payload:
            raise ValueError("Invalid Guardian API payload: missing response")
        response = payload["response"]
        if response.get("status") != "ok":
            raise RuntimeError(f"Guardian API returned status={response.get('status')}")
        return response

    def _build_search_params(
        self,
        request: GuardianSearchRequest,
        page: int,
    ) -> Dict[str, Any]:
        """Build a validated parameter set for the /search endpoint."""
        if not request.topic:
            raise ValueError("topic must not be empty")

        params: Dict[str, Any] = {
            "api-key": self.api_key,
            "format": "json",
            "q": request.topic,
            "from-date": self._normalize_date(request.run_date),
            "to-date": self._normalize_date(request.run_date),
            "use-date": "published",
            "page": page,
            "page-size": self._validate_page_size(request.page_size),
            "order-by": request.order_by,
        }
        if request.query:
            params["q"] = request.query
        if request.extra_filters:
            params.update(request.extra_filters)
        return params

    def _search_page(self, request: GuardianSearchRequest, page: int) -> Dict[str, Any]:
        """Fetch one /search page for a configured request profile."""
        payload = self._request_json("/search", self._build_search_params(request, page))
        return self._extract_response_or_raise(payload)

    def _search_next_page(
        self,
        last_content_id: str,
        base_query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch deep-pagination continuation using /content/{id}/next."""
        if not last_content_id:
            raise ValueError("last_content_id must not be empty")
        encoded_id = quote(last_content_id, safe="/")
        path = f"/content/{encoded_id}/next"
        params = dict(base_query_params)
        params.setdefault("api-key", self.api_key)
        params.setdefault("format", "json")
        payload = self._request_json(path, params)
        return self._extract_response_or_raise(payload)

    def _iter_search_responses(self, request: GuardianSearchRequest) -> Iterator[Dict[str, Any]]:
        """Yield sequential /search responses until pagination is exhausted."""
        page = 1
        while True:
            response = self._search_page(request, page)
            results = response.get("results", [])
            if not results:
                return
            yield response

            current_page = int(response.get("currentPage", page))
            total_pages = int(response.get("pages", current_page))
            if current_page >= total_pages:
                return
            page += 1

    def _build_next_query_params(self, request: GuardianSearchRequest) -> Dict[str, Any]:
        """Build /next continuation params from the base search request."""
        params = self._build_search_params(request, page=1)
        params.pop("page", None)
        return params

    def _iter_next_responses(
        self,
        start_id: str,
        base_query_params: Dict[str, Any],
        page_size: int,
    ) -> Iterator[Dict[str, Any]]:
        """Yield /next responses until the continuation stream is exhausted."""
        current_id: Optional[str] = start_id
        while current_id is not None:
            response = self._search_next_page(current_id, base_query_params)
            results = response.get("results", [])
            if not results:
                return
            yield response

            current_id = results[-1].get("id", current_id)
            if len(results) < page_size:
                return

    def _yield_items(
        self,
        results: List[Dict[str, Any]],
        state: IterationState,
    ) -> Iterator[Dict[str, Any]]:
        """Yield items while maintaining iteration state and optional limit."""
        for item in results:
            state.last_id = item.get("id", state.last_id)
            yield item
            if state.remaining is not None:
                state.remaining -= 1
                if state.remaining == 0:
                    state.exhausted = True
                    return

    def get_articles_list_by_topic(
        self,
        topic: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch one topic list response using ad-hoc params for compatibility."""
        payload = params.copy() if params else {}
        page = int(payload.pop("page", 1))
        order_by = payload.pop("order-by", "newest")
        request = GuardianSearchRequest(
            topic=topic,
            run_date=payload.pop("run_date", payload.pop("date", None)),
            page_size=self._validate_page_size(
                int(payload.pop("page-size", payload.pop("page_size", self.default_page_size)))
            ),
            query=payload.pop("q", None),
            extra_filters=payload,
            order_by=order_by,
            use_next_fallback=True,
        )
        response = self._search_page(request, page=page)
        return {
            "topic": topic,
            "date": self._normalize_date(request.run_date),
            "total_available": response.get("total", 0),
            "page": response.get("currentPage", page),
            "pages": response.get("pages", 0),
            "items": response.get("results", []),
        }

    def iter_topic_articles(
        self,
        profile: str,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield article list items using a named YAML profile."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than 0")

        request = self._get_profile_request(profile)
        state = IterationState(remaining=limit)
        last_batch_size = 0

        for response in self._iter_search_responses(request):
            results = response.get("results", [])
            last_batch_size = len(results)
            yield from self._yield_items(results, state)
            if state.exhausted:
                return

        if not request.use_next_fallback or not state.last_id:
            return
        if last_batch_size < request.page_size:
            return

        base_query_params = self._build_next_query_params(request)
        for response in self._iter_next_responses(
            start_id=state.last_id,
            base_query_params=base_query_params,
            page_size=request.page_size,
        ):
            yield from self._yield_items(response.get("results", []), state)
            if state.exhausted:
                return

    def get_articles_for_topic_day(
        self,
        profile: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a bounded list of articles for one configured profile/day."""
        request = self._get_profile_request(profile)
        items = list(self.iter_topic_articles(profile=profile, limit=limit))
        return {
            "topic": request.topic,
            "profile": profile,
            "date": self._normalize_date(request.run_date),
            "total_available": None,
            "fetched_count": len(items),
            "items": items,
            "pagination_summary": {
                "page_size": request.page_size,
                "used_next_fallback": request.use_next_fallback,
            },
        }

    def get_article_by_id(
        self,
        content_id: str,
        show_fields: str = "all",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch a full single content item by its Guardian content id."""
        if not content_id:
            raise ValueError("content_id must not be empty")
        encoded_id = quote(content_id, safe="/")
        params: Dict[str, Any] = {
            "api-key": self.api_key,
            "format": "json",
            "show-fields": show_fields,
        }
        if extra_params:
            params.update(extra_params)
        payload = self._request_json(f"/{encoded_id}", params)
        response = self._extract_response_or_raise(payload)
        content = response.get("content")
        if not content:
            raise ValueError("Guardian API single-item response is missing content")
        return content

    def get_articles_by_ids(
        self,
        content_ids: List[str],
        show_fields: str = "all",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch many articles by id and collect successes and failures."""
        items: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for content_id in content_ids:
            try:
                article = self.get_article_by_id(
                    content_id=content_id,
                    show_fields=show_fields,
                    extra_params=extra_params,
                )
                items.append(article)
            except Exception as exc:  # pylint: disable=broad-except
                failures.append({"id": content_id, "error": str(exc)})
        return {
            "fetched_count": len(items),
            "failed_count": len(failures),
            "items": items,
            "failures": failures,
        }
