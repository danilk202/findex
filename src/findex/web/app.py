import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from findex import __version__
from findex.web import api, pages
from findex.web.deps import IndexHolder
from findex.web.middleware import RequestContextMiddleware
from findex.web.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    # Фабрика: uvicorn викликає її в кожному воркер-процесі (findex serve --workers N),
    # тести — зі своїми Settings. Глобального app на рівні модуля немає.
    settings = settings or get_settings()
    holder = IndexHolder(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Один раз на процес: індекс вантажиться у фоні, сервер одразу приймає запити.
        loading = asyncio.create_task(holder.load())
        try:
            yield
        finally:
            loading.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loading
            holder.close()

    app = FastAPI(
        title="findex",
        version=__version__,
        summary="Навчальний пошуковик: інвертований індекс, BM25, мова запитів",
        lifespan=lifespan,
        docs_url="/api/docs",  # /docs/{doc_id} — наш маршрут, Swagger UI переїжджає
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.holder = holder
    app.add_middleware(RequestContextMiddleware)
    app.include_router(api.router)
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app
