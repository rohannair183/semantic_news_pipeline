"""
Client for interacting with the Open Guardian API. This module defines the `GuardianClient` class,
which is responsible for making HTTP requests to the Open Guardian API and processing the
responses.
"""

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterator, List, Optional, Union
from urllib.parse import quote

import requests

from src.config.settings import GuardianProfileConfig, Settings
from src.enums.guardian_use_date import GuardianUseDate
from src.utils.dates import coerce_day, format_day_iso, utc_today_date


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
        """Initialize client transport settings and configured query profiles.

        This initializes internal settings from configuration and prepares named
        query profiles used by the client. 

        Parameters:
            requests_per_second: Maximum number of requests to make per second.

        Returns:
            None
        """
        resolved_settings = Settings.load_settings()
        self._settings = resolved_settings

        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be greater than 0")
        self.requests_per_second = requests_per_second
        self._last_request_time = 0.0

        self._profiles = Settings.load_guardian_profile_configs()

    @property
    def api_key(self) -> str:
        """Return configured Guardian API key.

        Returns:
            str: Configured Guardian API key.
        """
        return self._settings.api_key

    @property
    def base_url(self) -> str:
        """Return normalized Guardian API base URL.

        Returns the base URL from settings with any trailing slash removed.

        Returns:
            str: Normalized Guardian API base URL.
        """
        return self._settings.base_url.rstrip("/")

    @property
    def default_page_size(self) -> int:
        """Return default page size from settings.

        Returns:
            int: Default page size to request from the API.
        """
        return self._settings.default_page_size

    @property
    def max_page_size(self) -> int:
        """Return max page size from settings.

        Returns:
            int: Maximum allowed page size for API requests.
        """
        return self._settings.max_page_size

    @property
    def timeout_seconds(self) -> int:
        """Return request timeout from settings.

        Returns:
            int: Timeout in seconds for HTTP requests.
        """
        return self._settings.timeout_seconds

    def _get_profile_request(self, profile: str) -> GuardianProfileConfig:
        """Return a configured request profile or raise a helpful error.

        Parameters:
            profile: The configured profile name to retrieve.

        Returns:
            GuardianProfileConfig: The request profile associated with `profile`.
        """
        if not profile:
            raise ValueError("profile must not be empty")

        request = self._profiles.get(profile)
        if request is None:
            raise ValueError(f"Profile '{profile}' is not configured")
        return request

    def _default_date(self) -> str:
        """Return today's date in UTC, formatted as YYYY-MM-DD.

        Returns:
            str: ISO formatted date string (YYYY-MM-DD) in UTC.
        """
        return format_day_iso(utc_today_date())

    def _normalize_date(
        self,
        run_date: Optional[Union[date, datetime, str]],
    ) -> str:
        """Normalize supported date inputs to YYYY-MM-DD.

        Parameters:
            run_date: A `datetime`, `date`, ISO date `str`, or `None`.

        Returns:
            str: ISO formatted date string (YYYY-MM-DD).
        """
        if run_date is None:
            return self._default_date()
        return format_day_iso(coerce_day(run_date))

    def _validate_page_size(self, page_size: int) -> int:
        """Validate and return a supported Guardian API page size.

        Parameters:
            page_size: Requested page size to validate.

        Returns:
            int: The validated page size.
        """
        if page_size < 1 or page_size > self.max_page_size:
            raise ValueError(f"page_size must be between 1 and {self.max_page_size}")
        return page_size

    def _throttle(self) -> None:
        """Enforce a minimum delay between requests to avoid rate limits.

        Returns:
            None
        """
        min_interval = 1.0 / self.requests_per_second
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _request_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request and parse the response body as JSON.

        Parameters:
            path: API path to append to the base URL (e.g. '/search').
            params: Query parameters to include with the request.

        Returns:
            Dict[str, Any]: Parsed JSON payload returned by the API.
        """
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
        """Extract and validate the top-level response object.

        Parameters:
            payload: Raw JSON payload returned from the Guardian API.

        Returns:
            Dict[str, Any]: The validated `response` object from the payload.
        """
        if not isinstance(payload, dict) or "response" not in payload:
            raise ValueError("Invalid Guardian API payload: missing response")
        response = payload["response"]
        if response.get("status") != "ok":
            raise RuntimeError(f"Guardian API returned status={response.get('status')}")
        return response

    def _build_search_params(
        self,
        request: GuardianProfileConfig,
        page: int,
    ) -> Dict[str, Any]:
        """Build a validated parameter set for the /search endpoint.

        Parameters:
            request: The resolved `GuardianProfileConfig` for which to build params.
            page: Page number to request.

        Returns:
            Dict[str, Any]: Query parameters suitable for the `/search` endpoint.
        """
        if not request.topic:
            raise ValueError("topic must not be empty")

        normalized_date = self._normalize_date(request.run_date)
        query = request.query or request.topic
        params: Dict[str, Any] = {
            "api-key": self.api_key,
            "format": "json",
            "q": query,
            "from-date": normalized_date,
            "to-date": normalized_date,
            "use-date": GuardianUseDate.PUBLISHED.value,
            "page": page,
            "page-size": self._validate_page_size(request.page_size),
            "order-by": request.order_by.value,
        }
        if request.section:
            params["section"] = request.section
        return params

    def _search_page(self, request: GuardianProfileConfig, page: int) -> Dict[str, Any]:
        """Fetch one /search page for a configured request profile.

        Parameters:
            request: The `GuardianProfileConfig` to execute.
            page: Page number to fetch.

        Returns:
            Dict[str, Any]: The top-level `response` object from the API.
        """
        payload = self._request_json("/search", self._build_search_params(request, page))
        return self._extract_response_or_raise(payload)

    def _search_next_page(
        self,
        last_content_id: str,
        base_query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch deep-pagination continuation using /content/{id}/next.

        Parameters:
            last_content_id: The last-seen content id used to continue pagination.
            base_query_params: Base query parameters to include with the /next call.

        Returns:
            Dict[str, Any]: The top-level `response` object from the API.
        """
        if not last_content_id:
            raise ValueError("last_content_id must not be empty")
        encoded_id = quote(last_content_id, safe="/")
        path = f"/content/{encoded_id}/next"
        params = dict(base_query_params)
        params.setdefault("api-key", self.api_key)
        params.setdefault("format", "json")
        payload = self._request_json(path, params)
        return self._extract_response_or_raise(payload)

    def _iter_search_responses(self, request: GuardianProfileConfig) -> Iterator[Dict[str, Any]]:
        """Yield sequential /search responses until pagination is exhausted.

        Parameters:
            request: The resolved `GuardianProfileConfig` to iterate.

        Returns:
            Iterator[Dict[str, Any]]: Iterator yielding top-level response dicts.
        """
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

    def _build_next_query_params(self, request: GuardianProfileConfig) -> Dict[str, Any]:
        """Build /next continuation params from the base search request.

        Parameters:
            request: The resolved `GuardianProfileConfig` to base params on.

        Returns:
            Dict[str, Any]: Query parameters for /content/{id}/next calls.
        """
        params = self._build_search_params(request, page=1)
        params.pop("page", None)
        return params

    def _iter_next_responses(
        self,
        start_id: str,
        base_query_params: Dict[str, Any],
        page_size: int,
    ) -> Iterator[Dict[str, Any]]:
        """Yield /next responses until the continuation stream is exhausted.

        Parameters:
            start_id: The content id to start continuation from.
            base_query_params: Base query parameters for the continuation calls.
            page_size: Page size expected for continuation results.

        Returns:
            Iterator[Dict[str, Any]]: Iterator yielding top-level response dicts.
        """
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
        """Yield items while maintaining iteration state and optional limit.

        Parameters:
            results: List of result objects to yield from.
            state: Mutable `IterationState` tracking remaining items and last id.

        Returns:
            Iterator[Dict[str, Any]]: Iterator yielding individual result items.
        """
        for item in results:
            state.last_id = item.get("id", state.last_id)
            yield item
            if state.remaining is not None:
                state.remaining -= 1
                if state.remaining == 0:
                    state.exhausted = True
                    return

    def iter_topic_articles(
        self,
        profile: str,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield article list items using a named YAML profile.

        Use this generator to stream articles defined by a named profile from
        configuration. Prefer this when processing many results or when you need
        to enforce a `limit` while iterating.

        Parameters:
            profile: The configured profile name to use.
            limit: Optional maximum number of items to yield.

        Returns:
            Iterator[Dict[str, Any]]: Iterator yielding article result objects.
        """
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
        """Return a bounded list of articles for one configured profile/day.

        Use this convenience method when you want a single collected result set
        (not a streaming generator) for a configured profile and date. It wraps
        `iter_topic_articles` and returns a dictionary with metadata and items.

        Parameters:
            profile: The configured profile name to use.
            limit: Optional maximum number of items to fetch.

        Returns:
            Dict[str, Any]: Structured result with topic, profile, date and items.
        """
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
        profile: str,
        content_id: str,
    ) -> Dict[str, Any]:
        """Fetch a full single content item using a configured profile.

        Use this to retrieve the content object for a single Guardian
        content id when you need full article fields driven by YAML settings.

        Parameters:
            profile: Configured profile name that controls detail fetch settings.
            content_id: Guardian content id string to fetch (e.g. 'world/2020/...').

        Returns:
            Dict[str, Any]: The `content` object returned by the Guardian API.
        """
        request = self._get_profile_request(profile)
        return self._get_article_by_id_for_request(request=request, content_id=content_id)

    def get_articles_by_ids(
        self,
        profile: str,
        content_ids: List[str],
    ) -> Dict[str, Any]:
        """Fetch many articles by id and collect successes and failures.

        Use this helper to bulk-fetch a set of content ids and receive a
        summarized result listing successful items and any failures. This is
        useful for batching requests where individual failures should not abort
        the whole operation.

        Parameters:
            profile: Configured profile name that controls detail fetch settings.
            content_ids: List of Guardian content id strings to fetch.

        Returns:
            Dict[str, Any]: A summary containing `fetched_count`, `failed_count`,
                `items` and `failures`.
        """
        request = self._get_profile_request(profile)
        items: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for content_id in content_ids:
            try:
                article = self._get_article_by_id_for_request(
                    request=request,
                    content_id=content_id,
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

    def _get_article_by_id_for_request(
        self,
        request: GuardianProfileConfig,
        content_id: str,
    ) -> Dict[str, Any]:
        """Fetch one content item using a resolved profile configuration."""
        if not content_id:
            raise ValueError("content_id must not be empty")

        encoded_id = quote(content_id, safe="/")
        params: Dict[str, Any] = {
            "api-key": self.api_key,
            "format": "json",
            "show-fields": request.content_show_fields,
        }
        payload = self._request_json(f"/{encoded_id}", params)
        response = self._extract_response_or_raise(payload)
        content = response.get("content")
        if not content:
            raise ValueError("Guardian API single-item response is missing content")
        return content
