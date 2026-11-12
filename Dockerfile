# syntax=docker/dockerfile:1
# Два етапи: у builder — uv, залежності строго з uv.lock, корпус, index.bin і модель
# all-MiniLM-L6-v2; у фінальному образі — лише готовий .venv, тексти, індекс, модель
# і матриця ембедингів з репозиторію (emb/), процес не від root.

# ---------- 1. збірка ----------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# спершу лише залежності: цей шар перезбирається, тільки коли змінився uv.lock
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# тепер сам пакет; --no-editable: у .venv потрапляє копія, а не посилання на src/
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# корпус і індекс збираються тут же: у git їх немає (data/ у .gitignore)
ARG GUTENBERG_MIRROR=https://www.gutenberg.org
RUN /app/.venv/bin/python -m findex.gutenberg data --mirror "$GUTENBERG_MIRROR" \
 && /app/.venv/bin/findex index data -o index.bin

# лаба 8: модель (ревізія і sha256 зафіксовані в findex.semantic) і матриця ембедингів.
# Матриця вже лежить у emb/ — рахувати її тут довго. Але якщо Gutenberg віддав інший текст
# книжки, ніж той, з якого вона збиралась (відбиток у emb/meta.json), — перерахувати.
ENV FINDEX_MODEL_DIR=/app/model
RUN /app/.venv/bin/python -c "from findex.semantic import ensure_model; ensure_model()"
COPY emb ./emb
RUN /app/.venv/bin/findex embed index.bin -o emb --skip-if-fresh

# ---------- 2. запуск ----------
FROM python:3.13-slim

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin findex

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
# шляхи в індексі відносні (data/moby.txt) — тому той самий WORKDIR /app
COPY --from=builder /app/data /app/data
COPY --from=builder /app/index.bin /app/index.bin
COPY --from=builder /app/model /app/model
COPY --from=builder /app/emb /app/emb

# Конфіг — лише змінні оточення; платформа може перевизначити будь-яку (Render ставить PORT)
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    WEB_CONCURRENCY=1 \
    INDEX_PATH=/app/index.bin \
    EMB_PATH=/app/emb \
    FINDEX_MODEL_DIR=/app/model \
    SEMANTIC_THREADS=1 \
    LOG_LEVEL=INFO

USER findex
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"]

CMD ["findex", "serve"]
