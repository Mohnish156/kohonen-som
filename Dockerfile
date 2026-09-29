# One image, two entry points:
#   serve (default):  docker run -p 8000:8000 -v ./artifacts:/artifacts som
#   train:            docker run -v ./artifacts:/artifacts som som-train --width 10 --height 10 --out /artifacts
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SOM_ARTIFACT_DIR=/artifacts
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

RUN useradd --create-home app && mkdir /artifacts && chown -R app /app /artifacts
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["som-serve", "--artifacts", "/artifacts", "--host", "0.0.0.0", "--port", "8000"]
