from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Request
from fastapi import Path as PathParam
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from findex import __version__
from findex.query import NeedsPositionsError, QuerySyntaxError
from findex.scoring import SCORER_NAMES
from findex.semantic import SemanticUnavailableError
from findex.web.deps import MaybeService
from findex.web.schemas import SearchParams
from findex.web.service import SearchPage


def _common(request: Request) -> dict[str, Any]:
    # змінні, потрібні кожному шаблону (футер, вибір скорера)
    return {"version": __version__, "scorers": SCORER_NAMES, "modes": MODE_LABELS}


router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(
    directory=Path(__file__).parent / "templates", context_processors=[_common]
)

EXAMPLES = ('"white whale"', "monster OR creature", "treasure pirate", "love NOT marriage")
# приклади для семантичного режиму: жодного «ключового» слова з потрібної книжки
SEMANTIC_EXAMPLES = (
    "a sailor's obsessive hunt for revenge",
    "girl falls into a nonsense dream world",
)
MODE_LABELS = {"lexical": "За словами", "semantic": "За змістом", "hybrid": "Гібрид"}


def _is_partial(request: Request) -> bool:
    # htmx просить лише блок результатів; при «Назад» у браузері — усю сторінку
    h = request.headers
    return h.get("hx-request") == "true" and h.get("hx-history-restore-request") != "true"


def _validation_message(e: ValidationError) -> str:
    err = e.errors()[0]
    return f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"


# HEAD теж: Render і монітори аптайму перевіряють головну саме ним
@router.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def home(request: Request, service: MaybeService) -> HTMLResponse:
    raw = {k: v for k, v in request.query_params.items() if k in SearchParams.model_fields}
    ctx: dict[str, Any] = {
        "q": raw.get("q", "").strip(),
        "scorer": raw.get("scorer", "bm25"),
        "mode": raw.get("mode", "lexical"),
        "semantic": bool(service and service.semantic_available),
        "semantic_examples": SEMANTIC_EXAMPLES,
        "data": None,
        "error": None,
        "examples": EXAMPLES,
        "docs": service.num_docs if service else None,
    }
    status = 200
    if service is None:
        ctx["error"] = "Індекс ще завантажується — оновіть сторінку за кілька секунд."
        status = 503
    elif ctx["q"]:
        try:
            params = SearchParams.model_validate(raw)
            data: SearchPage = service.search(
                params.q, params.scorer, params.k, params.page, params.mode
            )
            ctx["data"] = data
        except ValidationError as e:
            ctx["error"] = f"Некоректні параметри — {_validation_message(e)}"
            status = 422
        except QuerySyntaxError as e:
            ctx["error"] = f"Не вдалося розібрати запит: {e}"
            status = 422
        except NeedsPositionsError:
            ctx["error"] = "Пошук фраз у лапках потребує індексу з позиціями."
            status = 422
        except SemanticUnavailableError:
            ctx["error"] = "Семантичний пошук на цьому сервері вимкнено (немає ембедингів)."
            status = 503
    ctx["partial"] = _is_partial(request)
    name = "_results.html" if ctx["partial"] else "index.html"
    return templates.TemplateResponse(
        request, name, ctx, status_code=status, headers={"Vary": "HX-Request"}
    )


@router.get("/view/{doc_id}", response_class=HTMLResponse)
def view(
    request: Request,
    doc_id: Annotated[int, PathParam()],
    service: MaybeService,
    q: str = "",
    part: int | None = None,
) -> HTMLResponse:
    if service is None:
        return templates.TemplateResponse(
            request, "error.html", {"code": 503, "message": "Індекс ще завантажується."}, 503
        )
    try:
        chunk = service.document(doc_id, q.strip() or None, part)
    except (QuerySyntaxError, NeedsPositionsError):
        chunk = service.document(doc_id, None, part)
    if chunk is None:
        msg = f"Документа №{doc_id} немає в індексі."
        return templates.TemplateResponse(request, "error.html", {"code": 404, "message": msg}, 404)
    return templates.TemplateResponse(request, "doc.html", {"doc": chunk, "q": q.strip()})
