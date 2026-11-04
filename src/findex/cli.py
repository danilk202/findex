"""Командний рядок findex: index / search / stats.

Правила виводу:
- stdout — лише результат (таблиця або --json), щоб його можна було перенаправити;
- stderr — діагностика: логи (-v / -vv), смуга прогресу, повідомлення про помилки;
- очікувана помилка — один рядок у stderr і ненульовий код виходу, без стека викликів.
"""

import json
import logging
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Final, Literal, NoReturn, TypedDict

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from findex import __version__
from findex.corpus import corpus_files
from findex.engine import Searcher
from findex.index import Index, open_index
from findex.parallel import WorkerError, build_parallel, cpu_count
from findex.query import NeedsPositionsError, QuerySyntaxError, parse
from findex.scoring import make_scorer
from findex.snippets import doc_snippet
from findex.storage import IndexFormatError

log = logging.getLogger("findex.cli")

app = typer.Typer(
    name="findex",
    help="findex — навчальний пошуковик по текстовому корпусу.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
out = Console()  # stdout: результати
err = Console(stderr=True)  # stderr: логи, прогрес, помилки

EXIT_ERROR: Final = 1  # файл не знайдено, не індекс тощо
EXIT_BAD_QUERY: Final = 2  # запит не розібрався (як і помилки використання в typer)

# Маркери підсвітки в сніпеті: символи, яких не буває в книжковому тексті.
_HL: Final = ("\x02", "\x03")


class CliError(Exception):
    """Очікувана помилка, яку треба показати одним рядком."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


def fail(message: str, code: int = EXIT_ERROR) -> NoReturn:
    err.print(f"[bold red]error:[/] {escape(message)}", highlight=False, soft_wrap=True)
    raise typer.Exit(code)


@contextmanager
def expected_errors() -> Generator[None]:
    """Перетворює очікувані винятки на один рядок у stderr і код виходу."""
    try:
        yield
    except CliError as e:
        fail(str(e), e.code)
    except FileNotFoundError as e:
        fail(f"file not found: {e.filename}")
    except OSError as e:
        fail(f"cannot access {e.filename}: {e.strerror}")
    except IndexFormatError as e:
        fail(str(e))
    except QuerySyntaxError as e:
        fail(f"bad query: {e}", EXIT_BAD_QUERY)
    except NeedsPositionsError as e:
        fail(f"{e} (rebuild: findex index CORPUS --positions)", EXIT_BAD_QUERY)
    except WorkerError as e:
        # Не очікувана помилка, але й не баг CLI: показуємо стек саме воркера (__cause__).
        # Для процесу в ньому є і текст стека з дочірнього процесу (_RemoteTraceback).
        if e.__cause__ is not None:
            err.print("".join(traceback.format_exception(e.__cause__)), highlight=False, end="")
        fail(str(e))


def setup_logging(verbosity: int) -> None:
    """Єдине місце, де налаштовується logging: 0 — WARNING, -v — INFO, -vv — DEBUG."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        handlers=[RichHandler(console=err, show_time=False, show_path=False, markup=False)],
        force=True,  # перевизначити попередню конфігурацію (важливо для тестів CliRunner)
    )


def _version(value: bool) -> None:
    if value:
        out.print(f"findex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v: INFO, -vv: DEBUG (у stderr).")
    ] = 0,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Показати версію."),
    ] = False,
) -> None:
    """findex — навчальний пошуковик по текстовому корпусу."""
    setup_logging(verbose)


# ------------------------------------------------------------------ index


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Каталог з .txt-файлами.")],
    output: Annotated[Path, typer.Option("--out", "-o", help="Куди зберегти індекс.")] = Path(
        "index.bin"
    ),
    positions: Annotated[
        bool, typer.Option("--positions/--no-positions", help="Позиції слів (для фраз).")
    ] = True,
    fmt: Annotated[
        Literal["pickle", "json"], typer.Option("--format", help="Формат файлу.")
    ] = "pickle",
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-w", min=1, help="Скільки воркерів (типово — усі ядра)."),
    ] = None,
    executor: Annotated[
        # Literal тут, а не аліас ExecutorKind: typer не розгортає PEP 695 `type X = ...`
        Literal["serial", "threads", "processes"],
        typer.Option("--executor", "-x", help="Де виконувати build_partial."),
    ] = "processes",
) -> None:
    """Зібрати індекс з корпусу (зі смугою прогресу): build_partial у воркерах, потім merge."""
    with expected_errors():
        if not corpus.is_dir():
            raise CliError(f"corpus directory not found: {corpus}")
        # перевіряємо, куди писати, ДО збірки — а не після хвилини роботи
        if output.is_dir():
            raise CliError(f"--out points to a directory: {output}")
        if not output.parent.is_dir():
            raise CliError(f"output directory not found: {output.parent}")
        paths = corpus_files(corpus)
        if not paths:
            raise CliError(f"no .txt files in {corpus}")
        n_workers = workers if workers is not None else cpu_count()

        progress = Progress(
            TextColumn("[bold]indexing"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err,
        )
        with progress:
            task = progress.add_task("indexing", total=len(paths))
            # прогрес — по шматках: воркер-процес не може смикати смугу в батьківському
            postings, metas, report = build_parallel(
                paths,
                workers=n_workers,
                executor=executor,
                with_positions=positions,
                on_done=lambda chunk: progress.advance(task, len(chunk.paths)),
            )
        index = Index(postings, metas)

        index.save(output, fmt)
        size = output.stat().st_size / 2**20
        out.print(
            f"indexed {index.num_docs} documents, {len(index)} terms "
            f"-> {output} ({size:.1f} MiB, {fmt}, positions={positions}, "
            f"{report.executor} x{report.workers}, merge {report.merge_s:.2f} s)",
            highlight=False,
            soft_wrap=True,
        )


# ------------------------------------------------------------------ search


class ResultJSON(TypedDict):
    rank: int
    score: float
    doc_id: int
    title: str
    path: str
    tokens: int
    snippet: str | None


class SearchJSON(TypedDict):
    query: str
    tree: str
    scorer: str
    results: list[ResultJSON]


def _highlighted(snippet: str) -> Text:
    """Сніпет з маркерами _HL -> rich Text із підсвіченими термами."""
    text = Text()
    rest = snippet
    while (start := rest.find(_HL[0])) != -1:
        end = rest.find(_HL[1], start)
        text.append(rest[:start])
        text.append(rest[start + 1 : end], style="bold yellow")
        rest = rest[end + 1 :]
    text.append(rest)
    return text


def run_search(
    index: Index,
    query: str,
    scorer: Literal["bm25", "tfidf"],
    limit: int,
    snippets: bool,
    width: int,
) -> SearchJSON:
    """Уся логіка пошуку без виводу — її зручно тестувати окремо від CLI."""
    searcher = Searcher(index, make_scorer(scorer))
    results = searcher.search(query, k=limit)
    tree = parse(query)
    terms = list(tree.terms())
    rows: list[ResultJSON] = []
    for rank, r in enumerate(results, 1):
        meta = index.metas[r.doc_id]
        snippet = doc_snippet(meta.path, terms, width=width, mark=_HL) if snippets else None
        rows.append(
            ResultJSON(
                rank=rank,
                score=round(r.score, 4),
                doc_id=r.doc_id,
                title=meta.title,
                path=meta.path,
                tokens=meta.length,
                snippet=snippet,
            )
        )
    return SearchJSON(query=query, tree=str(tree), scorer=scorer, results=rows)


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Файл індексу.")],
    query: Annotated[
        str, typer.Argument(help="Напр.: 'python AND (async OR await) \"event loop\"'")
    ],
    scorer: Annotated[
        Literal["bm25", "tfidf"], typer.Option("--scorer", "-s", help="Формула рейтингу.")
    ] = "bm25",
    limit: Annotated[int, typer.Option("--limit", "-k", min=1, help="Скільки результатів.")] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="JSON у stdout замість таблиці.")] = False,
    snippets: Annotated[bool, typer.Option("--snippets/--no-snippets")] = True,
    width: Annotated[int, typer.Option(min=10, help="Півширина сніпета, символів.")] = 80,
) -> None:
    """Ранжований пошук: таблиця rich або --json."""
    with expected_errors():
        if not index_path.is_file():
            raise CliError(f"index file not found: {index_path} (create it: findex index CORPUS)")
        with open_index(index_path) as index:
            data = run_search(index, query, scorer, limit, snippets, width)

    if as_json:
        for row in data["results"]:
            if row["snippet"] is not None:
                row["snippet"] = row["snippet"].replace(_HL[0], "").replace(_HL[1], "")
        # без rich: жодних переносів і кольорів, лише дані для перенаправлення в файл/jq;
        # ensure_ascii (типово) — валідний JSON у будь-якому кодуванні консолі (Windows теж)
        typer.echo(json.dumps(data, indent=2))
        return

    if not data["results"]:
        out.print(f"nothing found for {data['tree']}", highlight=False)
        return
    table = Table(
        title=f"{data['tree']}",
        caption=f"{len(data['results'])} results · {scorer}",
        show_lines=snippets,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("score", justify="right")
    table.add_column("document", style="bold cyan")
    table.add_column("tokens", justify="right")
    if snippets:
        table.add_column("snippet", ratio=1)
    for row in data["results"]:
        cells: list[str | Text] = [
            str(row["rank"]),
            f"{row['score']:.3f}",
            row["title"],
            f"{row['tokens']:,}",
        ]
        if row["snippet"] is not None:
            cells.append(_highlighted(row["snippet"]))
        table.add_row(*cells)
    out.print(table)


# ------------------------------------------------------------------ stats


class StatsJSON(TypedDict):
    documents: int
    terms: int
    postings: int
    tokens: int
    avg_doc_length: float
    positions: bool
    file_mib: float
    top_terms: list[tuple[str, int, int]]


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="Файл індексу.")],
    top: Annotated[int, typer.Option("--top", "-n", min=0, help="Скільки частих термів.")] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="JSON у stdout.")] = False,
) -> None:
    """Статистика індексу: розміри, середня довжина, найчастіші терми."""
    with expected_errors():
        if not index_path.is_file():
            raise CliError(f"index file not found: {index_path}")
        with open_index(index_path) as index:
            data = StatsJSON(
                documents=index.num_docs,
                terms=len(index),
                postings=index.num_postings,
                tokens=index.num_tokens,
                avg_doc_length=round(index.avg_doc_length, 1),
                positions=index.has_positions,
                file_mib=round(index_path.stat().st_size / 2**20, 2),
                top_terms=index.top_terms(top),
            )

    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return
    summary = Table(title=f"findex stats: {index_path.name}", show_header=False)
    summary.add_column(style="bold")
    summary.add_column(justify="right")
    summary.add_row("documents", f"{data['documents']:,}")
    summary.add_row("terms", f"{data['terms']:,}")
    summary.add_row("postings", f"{data['postings']:,}")
    summary.add_row("tokens", f"{data['tokens']:,}")
    summary.add_row("avg doc length", f"{data['avg_doc_length']:,.1f}")
    summary.add_row("positions", "yes" if data["positions"] else "no")
    summary.add_row("file size", f"{data['file_mib']} MiB")
    out.print(summary)
    if data["top_terms"]:
        terms = Table(title=f"top {top} terms")
        terms.add_column("term", style="cyan")
        terms.add_column("df", justify="right")
        terms.add_column("total tf", justify="right")
        for term, df, cf in data["top_terms"]:
            terms.add_row(term, str(df), f"{cf:,}")
        out.print(terms)
