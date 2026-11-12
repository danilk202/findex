from typing import Annotated, NoReturn

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from findex.query import NeedsPositionsError, QuerySyntaxError
from findex.semantic import SemanticUnavailableError
from findex.web.deps import Holder, MaybeService, Service
from findex.web.schemas import (
    DocOut,
    DocParams,
    ErrorOut,
    HealthOut,
    SearchOut,
    SearchParams,
    StatsOut,
)

router = APIRouter(tags=["api"])


def bad_query(e: ValueError) -> NoReturn:
    # той самий формат, що й у 422 від Pydantic, щоб клієнтові був один розбір помилок
    raise HTTPException(
        422, detail=[{"loc": ["query", "q"], "msg": str(e), "type": "query_syntax"}]
    )


# def, а не async def: пошук і сніпети — чистий CPU на десятки мілісекунд.
# FastAPI виконує def-маршрут у пулі потоків, і цикл подій тим часом приймає інші запити.
# async def з тим самим тілом зупиняв би цикл на весь час пошуку (заміри — у README).
@router.get(
    "/search",
    response_model=SearchOut,
    responses={503: {"model": ErrorOut}},
    summary="Ранжований пошук, посторінково",
)
def search(params: Annotated[SearchParams, Query()], service: Service) -> SearchOut:
    try:
        data = service.search(params.q, params.scorer, params.k, params.page, params.mode)
    except (QuerySyntaxError, NeedsPositionsError) as e:
        bad_query(e)
    except SemanticUnavailableError as e:
        raise HTTPException(503, str(e)) from e
    return SearchOut.from_page(data)


@router.get("/docs/{doc_id}", response_model=DocOut, summary="Текст документа")
def document(
    doc_id: Annotated[int, Path(ge=0)],
    params: Annotated[DocParams, Query()],
    service: Service,
) -> DocOut:
    found = service.raw_text(doc_id, params.offset, params.limit)
    if found is None:
        raise HTTPException(404, f"document {doc_id} not found")
    meta, text, chars = found
    return DocOut(
        doc_id=meta.doc_id,
        title=service.heading(doc_id)[0],
        path=meta.path,
        tokens=meta.length,
        chars=chars,
        offset=params.offset,
        text=text,
    )


@router.get("/stats", response_model=StatsOut, summary="Статистика індексу")
def stats(service: Service) -> StatsOut:
    return StatsOut.from_stats(service.stats)


@router.get(
    "/health",
    response_model=HealthOut,
    response_model_exclude_none=True,
    responses={503: {"model": HealthOut}},
    summary="Перевірка для платформи: 200, коли індекс завантажено",
)
async def health(service: MaybeService, holder: Holder) -> HealthOut | JSONResponse:
    if service is not None:
        return HealthOut(status="ok", documents=service.num_docs)
    body = (
        HealthOut(status="error", detail=holder.error)
        if holder.error
        else HealthOut(status="loading")
    )
    return JSONResponse(body.model_dump(exclude_none=True), status_code=503)
