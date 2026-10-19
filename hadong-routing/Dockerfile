# Ảnh chạy được pipeline + giao diện Streamlit, chỉ CPU.
#   docker build -t hadong-routing .
#   docker run --rm -p 8501:8501 hadong-routing                       # giao diện (dùng dữ liệu/kết quả đã có trong ảnh)
#   docker run --rm -v %cd%/results:/app/results hadong-routing python run_all.py --mode smoke
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# torch bản CPU (nhỏ hơn nhiều so với bản CUDA mặc định trên PyPI cho Linux)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu \
 && grep -v '^torch==' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
