"""
Client for interacting with the Open Guardian API. This module defines the `GuardianClient` class,
which is responsible for making HTTP requests to the Open Guardian API and processing the
responses.
"""

import json
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from src.application.settings import Settings

class GuardianClient:
    """
    Class representing the client for interacting with the Open Guardian API. This class is
    responsible for making HTTP requests to the API, handling authentication, and processing the
    responses. It provides methods for fetching articles based on specified topics and parameters.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        settings: Optional[Settings] = None,
        base_url: Optional[str] = None,
        requests_per_second: float = 1.0,
        timeout_seconds: Optional[int] = None,
    ):
        """
        Initializes the GuardianClient instance. This method can be used to set up any necessary
        configurations, such as API keys or base URLs for the Open Guardian API.
        """
        resolved_settings = settings or Settings.load_settings(api_key=api_key)
        self.api_key = resolved_settings.api_key
        self.base_url = (base_url or resolved_settings.base_url).rstrip("/")
        self.default_page_size = resolved_settings.default_page_size
        self.max_page_size = resolved_settings.max_page_size
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be greater than 0")
        self.requests_per_second = requests_per_second
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else resolved_settings.timeout_seconds
        )
        self._last_request_time = 0.0

    def _default_date(self) -> str:
        """Return today's date in UTC, formatted as YYYY-MM-DD.
        Returns:
            str: The current date in UTC, formatted as YYYY-MM-DD.
        """
        return datetime.now(timezone.utc).date().isoformat()

    def _normalize_date(self, run_date: Optional[Any]) -> str:
        """Normalize supported date inputs to YYYY-MM-DD.
        Args:
            run_date (Optional[Any]): The date to normalize.
        Returns:
            str: The normalized date in YYYY-MM-DD format.
        """
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
                raise ValueError(
                    "run_date must be in YYYY-MM-DD format") from exc
        raise ValueError("run_date must be a date, datetime, string, or None")

    def _validate_page_size(self, page_size: int) -> int:
        """Validate and return a supported Guardian API page size.
        Args:
            page_size (int): The page size to validate.
        Returns:
            int: The validated page size."""
        if page_size < 1 or page_size > self.max_page_size:
            raise ValueError(
                f"page_size must be between 1 and {self.max_page_size}")
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
        query = urlencode(params or {}, doseq=True)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = Request(url=url, method="GET")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Guardian API HTTP error {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Guardian API connection error: {exc.reason}") from exc

    def _extract_response_or_raise(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate the top-level response object."""
        if not isinstance(payload, dict) or "response" not in payload:
            raise ValueError("Invalid Guardian API payload: missing response")
        response = payload["response"]
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Guardian API returned status={response.get('status')}")
        return response

    def _build_search_params(
        self,
        topic: str,
        run_date: Optional[Any],
        page: int,
        page_size: int,
        query: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
        order_by: str = "newest",
    ) -> Dict[str, Any]:
        """Build a validated parameter set for the /search endpoint."""
        if not topic:
            raise ValueError("topic must not be empty")
        normalized_date = self._normalize_date(run_date)
        validated_page_size = self._validate_page_size(page_size)
        params: Dict[str, Any] = {
            "api-key": self.api_key,
            "format": "json",
            "q": topic,
            "from-date": normalized_date,
            "to-date": normalized_date,
            "use-date": "published",
            "page": page,
            "page-size": validated_page_size,
            "order-by": order_by,
        }
        if query:
            params["q"] = query
        if extra_filters:
            params.update(extra_filters)
        return params

    def _search_page(
        self,
        topic: str,
        run_date: Optional[Any],
        page: int,
        page_size: int,
        query: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
        order_by: str = "newest",
    ) -> Dict[str, Any]:
        """Fetch one /search page for a topic on a given day."""
        params = self._build_search_params(
            topic=topic,
            run_date=run_date,
            page=page,
            page_size=page_size,
            query=query,
            extra_filters=extra_filters,
            order_by=order_by,
        )
        payload = self._request_json("/search", params)
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

    def get_articles_list_by_topic(self, topic: str, params: Optional[Dict[str, Any]] = None) -> dict:
        """
        Fetches articles from the Open Guardian API based on the specified topic and parameters.

        Args:
            topic (str): The topic to search for in the Open Guardian API.
            params (dict, optional): Additional parameters to include in the API request.
        Returns:
            dict: A dictionary containing the API response with the fetched articles.
        """
        params = params or {}
        page = int(params.pop("page", 1))
        page_size = int(params.pop(
            "page-size", params.pop("page_size", self.default_page_size)))
        run_date = params.pop("run_date", params.pop("date", None))
        query = params.pop("q", None)
        order_by = params.pop("order-by", "newest")
        response = self._search_page(
            topic=topic,
            run_date=run_date,
            page=page,
            page_size=page_size,
            query=query,
            extra_filters=params,
            order_by=order_by,
        )
        return {
            "topic": topic,
            "date": self._normalize_date(run_date),
            "total_available": response.get("total", 0),
            "page": response.get("currentPage", page),
            "pages": response.get("pages", 0),
            "items": response.get("results", []),
        }

    def iter_topic_articles(
        self,
        topic: str,
        run_date: Optional[Any] = None,
        limit: Optional[int] = None,
        page_size: Optional[int] = None,
        query: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
        order_by: str = "newest",
        use_next_fallback: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Yield article list items for a topic/day with pagination and optional /next fallback."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than 0")

        normalized_date = self._normalize_date(run_date)
        effective_page_size = page_size if page_size is not None else self.default_page_size
        validated_page_size = self._validate_page_size(effective_page_size)
        remaining = limit
        page = 1
        last_id: Optional[str] = None
        last_batch_size = 0

        while True:
            response = self._search_page(
                topic=topic,
                run_date=normalized_date,
                page=page,
                page_size=validated_page_size,
                query=query,
                extra_filters=extra_filters,
                order_by=order_by,
            )
            results = response.get("results", [])

            if not results:
                break

            last_batch_size = len(results)
            for item in results:
                last_id = item.get("id", last_id)
                yield item
                if remaining is not None:
                    remaining -= 1
                    if remaining == 0:
                        return

            current_page = int(response.get("currentPage", page))
            total_pages = int(response.get("pages", current_page))
            if current_page >= total_pages:
                break
            page += 1

        if not use_next_fallback or not last_id:
            return
        if last_batch_size < validated_page_size:
            return

        base_query_params = self._build_search_params(
            topic=topic,
            run_date=normalized_date,
            page=1,
            page_size=validated_page_size,
            query=query,
            extra_filters=extra_filters,
            order_by=order_by,
        )
        base_query_params.pop("page", None)

        while last_id is not None:
            response = self._search_next_page(last_id, base_query_params)
            results = response.get("results", [])
            if not results:
                break

            for item in results:
                last_id = item.get("id", last_id)
                yield item
                if remaining is not None:
                    remaining -= 1
                    if remaining == 0:
                        return

            if len(results) < validated_page_size:
                break

    def get_articles_for_topic_day(
        self,
        topic: str,
        run_date: Optional[Any] = None,
        limit: Optional[int] = None,
        page_size: Optional[int] = None,
        query: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
        order_by: str = "newest",
        use_next_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Return a bounded list of articles for one topic/day."""
        effective_page_size = page_size if page_size is not None else self.default_page_size
        items = list(
            self.iter_topic_articles(
                topic=topic,
                run_date=run_date,
                limit=limit,
                page_size=effective_page_size,
                query=topic,
                extra_filters=extra_filters,
                order_by=order_by,
                use_next_fallback=use_next_fallback,
            )
        )
        return {
            "topic": topic,
            "date": self._normalize_date(run_date),
            "total_available": None,
            "fetched_count": len(items),
            "items": items,
            "pagination_summary": {
                "page_size": effective_page_size,
                "used_next_fallback": use_next_fallback,
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
            raise ValueError(
                "Guardian API single-item response is missing content")
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
