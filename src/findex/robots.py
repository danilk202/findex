import asyncio
import logging
from dataclasses import dataclass
from urllib.robotparser import RobotFileParser

import httpx

from findex.urls import origin

log = logging.getLogger("findex.crawl.urls")  # рішення щодо robots.txt — у crawl.log


@dataclass(frozen=True, slots=True)
class Robots:
    """Правила одного хоста. parser=None означає «дозволено все» або «заборонено все»."""

    parser: RobotFileParser | None
    allow_all: bool
    status: int | None

    def can_fetch(self, agent: str, url: str) -> bool:
        if self.parser is None:
            return self.allow_all
        return self.parser.can_fetch(agent, url)

    def crawl_delay(self, agent: str) -> float | None:
        if self.parser is None:
            return None
        delay = self.parser.crawl_delay(agent)
        return float(delay) if delay is not None else None


class RobotsCache:
    """robots.txt окремо для кожного хоста (scheme://host:port), завантажується один раз.

    У кеші лежить не результат, а Task: якщо 20 воркерів одночасно прийшли на новий
    хост, robots.txt завантажиться один раз, а решта почекає на ту саму задачу.

    Що робити, якщо файлу немає (RFC 9309): 4xx -> дозволено все; 5xx або сервер
    недоступний -> заборонено все (сайт, можливо, лежить — не добиваємо його).
    """

    def __init__(self, client: httpx.AsyncClient, agent: str, timeout: float = 10.0) -> None:
        self._client = client
        self._agent = agent
        self._timeout = timeout
        self._tasks: dict[str, asyncio.Task[Robots]] = {}

    async def get(self, url: str) -> Robots:
        key = origin(url)
        task = self._tasks.get(key)
        if task is None:
            task = asyncio.create_task(self._load(key), name=f"robots {key}")
            self._tasks[key] = task
        return await asyncio.shield(task)  # скасування одного воркера не вбиває спільне

    async def allowed(self, url: str) -> bool:
        return (await self.get(url)).can_fetch(self._agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        return (await self.get(url)).crawl_delay(self._agent)

    def close(self) -> None:
        for task in self._tasks.values():
            task.cancel()

    async def _load(self, key: str) -> Robots:
        robots_url = f"{key}/robots.txt"
        try:
            async with asyncio.timeout(self._timeout):
                resp = await self._client.get(robots_url, follow_redirects=True)
        except (TimeoutError, httpx.HTTPError) as e:
            log.info("ROBOTS   %s unavailable (%s): disallow all", robots_url, type(e).__name__)
            return Robots(None, allow_all=False, status=None)
        if resp.status_code >= 500:
            log.info("ROBOTS   %s -> %d: disallow all", robots_url, resp.status_code)
            return Robots(None, allow_all=False, status=resp.status_code)
        if resp.status_code >= 400:
            log.info("ROBOTS   %s -> %d: allow all", robots_url, resp.status_code)
            return Robots(None, allow_all=True, status=resp.status_code)
        parser = RobotFileParser(robots_url)
        parser.parse(resp.text.splitlines())
        log.info("ROBOTS   %s -> %d: rules loaded", robots_url, resp.status_code)
        return Robots(parser, allow_all=True, status=resp.status_code)
