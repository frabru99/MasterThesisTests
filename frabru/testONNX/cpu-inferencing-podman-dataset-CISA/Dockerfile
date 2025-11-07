FROM python:3.10.19-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1
WORKDIR /test
COPY requirements.txt .
COPY ./inferencingTest.py .
COPY ./exportModel.py .
RUN pip install --no-cache-dir --upgrade pip wheel && \ 
    pip install --no-cache-dir --retries 1 -r requirements.txt && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/* && \
    rm requirements.txt
CMD ["/bin/baah"]
