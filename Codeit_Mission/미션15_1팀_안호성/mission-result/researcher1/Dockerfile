FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /workspace

ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY train.py .
COPY data ./data

ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["python", "train.py"]
