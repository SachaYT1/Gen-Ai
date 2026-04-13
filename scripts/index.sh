#!/usr/bin/env bash
set -e

echo "=============================="
echo "📦 Setting up BM25 index (~8.5GB)"
echo "=============================="

INDEX_URL="https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz"
ARCHIVE_NAME="index.tar.gz"
TARGET_DIR="data/pyserini/indexes"

# 1. Проверка curl
if ! command -v curl &> /dev/null
then
    echo "❌ curl is not installed. Please install curl."
    exit 1
fi

# 2. Создаём папки
echo "📁 Creating directories..."
mkdir -p $TARGET_DIR

# 3. Скачивание (с докачкой!)
echo "⬇️ Downloading index (this may take a while)..."
curl -L -C - -o $ARCHIVE_NAME $INDEX_URL

# 4. Распаковка
echo "📦 Extracting..."
tar -xvf $ARCHIVE_NAME

# 5. Перемещение
echo "📂 Moving index to $TARGET_DIR..."
mv lucene-index.wikipedia-dpr-100w.20210120.d1b9e6 $TARGET_DIR/

# 6. Очистка
echo "🧹 Cleaning up..."
rm $ARCHIVE_NAME

echo "=============================="
echo "✅ BM25 index is ready!"
echo "=============================="
echo ""
echo "Теперь можно запускать:"
echo "docker compose run --rm nq"