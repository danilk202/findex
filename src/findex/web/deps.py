import asyncio
import logging
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from findex.index import Index
from findex.web.service import SearchService
from findex.web.settings import Settings

log = logging.getLogger("findex.web")


class IndexHolder:
    # Стан застосунку: індекс з'являється тут, коли lifespan його дочитає.
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.service: SearchService | None = None
        self.error: str | None = None

    async def load(self) -> None:
        path = self.settings.index_path
        try:
            # читання pickle — блокуюча робота, тож в окремому потоці: поки індекс
            # вантажиться, цикл подій живий і /health уже відповідає 503
            index = await asyncio.to_thread(Index.load, path)
        except Exception as e:
            self.error = f"cannot load index {path}: {e}"
            log.exception("cannot load index %s", path)
            return
        self.service = SearchService(
            index,
            snippet_width=self.settings.snippet_width,
            text_cache=self.settings.text_cache,
        )
        log.info("index ready: %s, %d documents", path, index.num_docs)

    def close(self) -> None:
        if self.service is not None:
            self.service.index.close()
            self.service = None


# Залежності — async def: вони нічого не блокують, а звичайний def FastAPI
# запускав би в пулі потоків (зайвий перехід на кожен запит).


async def get_holder(request: Request) -> IndexHolder:
    return cast(IndexHolder, request.app.state.holder)


async def current_service(
    holder: Annotated[IndexHolder, Depends(get_holder)],
) -> SearchService | None:
    return holder.service


async def require_service(
    service: Annotated[SearchService | None, Depends(current_service)],
) -> SearchService:
    if service is None:
        raise HTTPException(503, "index is not loaded yet", headers={"Retry-After": "5"})
    return service


Service = Annotated[SearchService, Depends(require_service)]
MaybeService = Annotated[SearchService | None, Depends(current_service)]
Holder = Annotated[IndexHolder, Depends(get_holder)]
