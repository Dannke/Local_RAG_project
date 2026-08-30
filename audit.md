# Аудит проекта Local RAG Project

## Обзор проекта

**Local RAG Project** — это локальная система Retrieval-Augmented Generation (RAG) для работы с документами форматов PDF, DOCX, TXT и Markdown. Проект написан на Python и предоставляет как веб-интерфейс (Streamlit), так и командную строку (CLI) для индексации документов и задавания вопросов в естественном языке.

### Ключевые характеристики

| Параметр | Значение |
|----------|----------|
| **Язык** | Python ≥ 3.11 |
| **Архитектура** | Модульная, с разделением на пакеты: `ingestion`, `embeddings`, `vectorstores`, `retrieval`, `generation`, `llm`, `pipelines`, `chat_store` |
| **Векторная БД** | FAISS (IndexFlatIP — косинусное сходство) |
| **Эмбеддинги** | sentence-transformers (по умолчанию `paraphrase-multilingual-MiniLM-L12-v2`) |
| **LLM провайдер** | OpenRouter (совместим с OpenAI API) |
| **Рерайкинг** | Cross-Encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) |
| **UI** | Streamlit (многостраничный: Chat / Documents / Settings) |
| **Чат-сессии** | Файловое хранение (JSON) с поддержкой множественных чатов |

---

## Архитектура и компоненты

### 1. Конфигурация (`src/rag_project/config.py`)

Класс `Settings` (dataclass, frozen) хранит все настройки:

- **Пути**: `project_root`, `raw_data_dir`, `processed_data_dir`, `vector_store_dir`
- **Эмбеддинги**: `embedding_model`
- **LLM**: `openrouter_model`, `openrouter_base_url`, `openrouter_timeout_seconds`, `openrouter_api_key`
- **Retrieval**: `top_k` (по умолчанию 5), `max_context_chars` (12000), `temperature` (0.2)
- **Рерайкинг**: `use_reranker` (True), `reranker_model`, `rerank_candidates` (20)
- **Чанкинг**: `chunk_size` (800), `chunk_overlap` (120)

Настройки загружаются из переменных окружения (`.env` файл) с дефолтами.

### 2. Модели данных (`src/rag_project/models.py`)

- **Document** — единица хранения: `id`, `text`, `metadata` (dict)
- **SearchResult** — результат поиска: `document` + `score` (float)

### 3. Загрузка документов (`src/rag_project/ingestion/loaders.py`)

Поддерживаемые форматы:
- **PDF** — построчно через `pypdf.PdfReader`, каждая страница = отдельный Document с метаданными `page`, `page_label`
- **DOCX** — через `python-docx`, параграфы объединяются в один Document (`page=1`)
- **TXT / MD** — прямое чтение текста

Функция `load_documents(input_dir, patterns)` рекурсивно сканирует директорию и возвращает список `Document`.

### 4. Чанкинг (`src/rag_project/ingestion/chunking.py`)

Умный чанкинг с учётом структуры текста:

1. **Разбиение на блоки** (`split_text_blocks`):
   - Параграфы (пустые строки-разделители)
   - Заголовки Markdown (`^#{1,6}\s+\S+`)
   - Строки, заканчивающиеся на `:` (длина ≤ 120)

2. **Сборка чанков** (`chunk_text`):
   - Блоки накапливаются до `chunk_size`
   - При переполнении — сохраняется предыдущий чанк, следующий начинается с перекрытием (`chunk_overlap`)
   - Очень длинные блоки режутся скользящим окном (`_window_chunks`)

3. **`chunk_documents`** — применяет к списку Document, сохраняет метаданные + `parent_id`, `chunk_index`.

### 5. Эмбеддинги (`src/rag_project/embeddings/`)

**Интерфейс** `EmbeddingModel` (Protocol) — метод `embed_texts(texts) -> list[list[float]]`.

Реализации:
- **`SentenceTransformerEmbeddingModel`** — обёртка над `sentence_transformers.SentenceTransformer`, нормализует векторы (L2), возвращает `float32`.
- **`HashingEmbeddingModel`** — детерминированный fallback для тестов (без ML-зависимостей).

### 6. Векторное хранилище (`src/rag_project/vectorstores/`)

**Интерфейс** `VectorStore` (Protocol): `add`, `search`, `count`.

Реализации:
- **`FaissVectorStore`** — основная продакшн-реализация:
  - `IndexFlatIP` (внутренний скалярный продукт = косинусное сходство для нормированных векторов)
  - Метаданные документов в `documents.json` (JSON-сериализация dataclass)
  - Сохранение/загрузка с диска (`save_to_disk`, `load_from_disk_into_self`)
  - Проверка консистентности: `index.ntotal == len(documents)`
- **`InMemoryVectorStore`** — для тестов и прототипов.

### 7. Ретривер (`src/rag_project/retrieval/retriever.py`)

Класс `Retriever(embedding_model, vector_store)`:
- `index(documents)` — эмбеддит тексты и кладёт в vector store
- `search(query, top_k)` — эмбеддит запрос, ищет в vector store, возвращает `SearchResult[]`

### 8. Рерайкинг (`src/rag_project/retrieval/reranker.py`)

Интерфейс `Reranker` + две реализации:
- **`NoOpReranker`** — просто обрезает до `top_k`
- **`CrossEncoderReranker`** — использует `sentence_transformers.CrossEncoder`:
  - Формирует пары `(query, chunk_text)`
  - Предсказывает скоры, сортирует по убыванию, возвращает `top_k`

Включается через настройку `use_reranker`. Количество кандидатов — `rerank_candidates` (дефолт 20).

### 9. Цитирование (`src/rag_project/retrieval/citations.py`)

- **`Citation`** (dataclass): `label` (S1, S2...), `source`, `page`, `chunk_index`, `score`, `text`
- **`build_citations(results)`** — строит список из `SearchResult`
- **`format_citation_context(citations)`** — форматирует для промпта LLM: `[S1] source=..., page=..., chunk=...\n{text}`

### 10. LLM клиент (`src/rag_project/llm/llm_client.py`)

**`OpenRouterClient`** — обёртка над `openai.OpenAI` (совместимый API):
- `generate_answer(question, context_chunks)` — блокирующий вызов
- `stream_generate_answer(...)` — стриминг (yield токены)
- Системный промпт на русском: «Отвечай ТОЛЬКО на основе предоставленного контекста...»
- Обработка ошибок: `MissingAPIKeyError`, `LLMTimeoutError`, `InvalidModelError`, `EmptyLLMResponseError`, `LLMProviderError`
- Лимит контекста: `limit_context_chunks` обрезает до `max_context_chars`

**`OpenRouterGenerator`** — адаптер для пайплайнов (используется в `ChatSession`).

### 11. Генерация (`src/rag_project/generation/llm.py`)

- **`LLM`** (Protocol) — интерфейс `generate`, `stream`
- **`ContextOnlyGenerator`** — заглушка без LLM (возвращает контекст как есть), для тестов/демо.

### 12. Пайплайны (`src/rag_project/pipelines/`)

#### `ingest_pipeline.py`
- **`build_retriever(...)`** — полный цикл: загрузка → чанкинг → эмбеддинг → индексация
- **`ingest_to_faiss(...)`** — главная функция:
  - `incremental=True` (по умолчанию): использует **манифест** для дедупликации
  - `incremental=False`: полная перестройка
  - Прогресс-колбэк для UI

#### `index_manifest.py` — инкрементальная индексация
- **`SourceFileRecord`**: `path`, `sha256`, `size`, `mtime_ns`, `chunk_ids[]`
- **`IndexManifest`**: словарь `files[path] -> SourceFileRecord` + версия
- `scan_source_files` — сканирует `data/raw`, считает SHA256
- Логика:
  - Нет индекса/манифеста → полная индексация
  - Удалены/изменены файлы → полная перестройка
  - Только новые файлы → загрузка, чанкинг, добавление в существующий FAISS, обновление манифеста

#### `search_pipeline.py`
- **`search_index(question, index_dir, top_k)`** — просто поиск без LLM.

#### `chat_pipeline.py` — основной RAG-пайплайн
- **`ChatResponse`**: `answer`, `contexts`, `results`, `citations`
- **`ChatStreamResponse`**: `chunks` (Iterator[str]), `contexts`, `results`, `citations`
- **`ChatSession`**:
  - `from_faiss_index(index_dir, settings)` — фабрика, создаёт retriever + reranker + generator
  - `ask(question, top_k)` — нестриминговый ответ
  - `stream(question, top_k)` — стриминговый ответ
  - `_search`: получает `rerank_candidates` кандидатов, реранжирует до `top_k`

### 13. Хранилище чатов (`src/rag_project/chat_store.py`)

Файловая структура для каждого чата (`data/chats/chat_XXX/`):
```
chat_XXX/
├── meta.json        # id, title, created_at, updated_at
├── messages.json    # [{"role": "user|assistant", "content": "..."}]
├── sources.json     # сохранённые источники последнего ответа
├── documents/       # загруженные файлы (копии из data/raw)
└── index/           # FAISS индекс чата (index.faiss, documents.json, manifest.json)
```

Функции:
- `initialize_chats`, `list_chats`, `create_chat`, `delete_chat`, `rename_chat`
- `load_messages`/`save_messages`, `load_sources`/`save_sources`
- `get_chat_documents_dir`, `get_chat_index_dir`
- `ensure_chat_layout` — создаёт структуру при первом обращении

### 14. CLI (`src/rag_project/cli.py`)

Три команды:
- `rag-project ingest --data data/raw --index data/vector_store`
- `rag-project search "вопрос" --index data/vector_store --top-k 5`
- `rag-project chat --index data/vector_store --top-k 5` (интерактивный цикл)

### 15. Вспомогательные скрипты (`scripts/`)
- `ingest.py` — обёртка над `ingest_to_faiss`
- `query.py "вопрос"` — поиск + вывод
- `chat.py` — интерактивный чат

### 16. Веб-UI (`app.py`)

Streamlit-приложение с навигацией (st.navigation):
- **Chat** — три панели: слева список чатов, центр — диалог, справа — источницы (citations)
- **Documents** — дашборд KB (метрики: доки, чанки, статус индекса, размер), загрузка файлов, карточки документов (удалить/реиндекс), управление индексом (Index / Clear / Rebuild / Clear all)
- **Settings** — параметры ответа (top_k, max_context_chars, temperature), статус reranker, сброс UI

Особенности UI:
- Стриминг ответов («печатает...» → прогрессивный рендер)
- Сохранение сообщений и источников в чат (переживают рефреш)
- Прогресс-бар при индексации с этапами
- Инкрементальная индексация для новых файлов
- Удаление одного документа сбрасывает индекс (FAISS не поддерживает удаление векторов)

---

## Поток данных (Data Flow)

### Индексация (Ingest)
```
data/raw/ (PDF, DOCX, TXT, MD)
    ↓ load_documents()
List[Document] (по страницам для PDF, целиком для остальных)
    ↓ chunk_documents(chunk_size=800, overlap=120)
List[Document] (чанки с parent_id, chunk_index)
    ↓ SentenceTransformerEmbeddingModel.embed_texts()
List[float32 vectors]
    ↓ FaissVectorStore.add() → IndexFlatIP
    ↓ save_to_disk()
data/vector_store/
├── index.faiss
├── documents.json
└── manifest.json (для инкрементального обновления)
```

### Чат / Вопрос-ответ
```
User Question
    ↓ Retriever.search() → top_k * rerank_candidates (дефолт 5*20=100)
List[SearchResult] (document + score)
    ↓ CrossEncoderReranker.rerank() → top_k (5)
List[SearchResult] (переранжированные)
    ↓ build_citations() → List[Citation]
    ↓ format_citation_context() → List[str] (контекст для LLM)
    ↓ OpenRouterClient.stream_generate_answer(question, context_chunks)
Stream токенов → UI
    ↓ Сохранение в чат
messages.json + sources.json
```

---

## Возможности проекта

### Что умеет

1. **Загрузка и индексация документов**
   - PDF (постранично с метаданными страницы)
   - DOCX (параграфы, страница = 1)
   - TXT / Markdown
   - Инкрементальная индексация (только новые/изменённые файлы)
   - Полная перестройка индекса
   - Прогресс-бар с этапами

2. **Поиск и ретривал**
   - Семантический поиск через FAISS (косинусное сходство)
   - Cross-Encoder реранжирование (опционально, включено по умолчанию)
   - Настройка `top_k` и `rerank_candidates`

3. **Генерация ответов (RAG)**
   - Стриминг ответов через OpenRouter
   - Системный промпт на русском с запретом галлюцинаций
   - Цитирование источников `[S1]`, `[S2]` в ответе
   - Лимит контекста `MAX_CONTEXT_CHARS`

4. **Управление чатами**
   - Множественные чаты (создание, переименование, удаление)
   - История сообщений (JSON)
   - Сохранение источников последнего ответа
   - Изолированные индексы и документы для каждого чата

5. **Веб-интерфейс (Streamlit)**
   - 3 страницы: Chat / Documents / Settings
   - Дашборд Knowledge Base (метрики)
   - Drag-and-drop загрузка файлов
   - Карточки документов с действиями
   - Управление индексами (Index / Clear / Rebuild / Clear all)
   - Настройка параметров в UI

6. **CLI**
   - Индексация, поиск, интерактивный чат

7. **Тестируемость**
   - Юнит-тесты для всех основных модулей (`tests/`)
   - Моки для LLM и эмбеддингов
   - InMemoryVectorStore для быстрых тестов

### Ограничения и известные нюансы

| Ограничение | Описание |
|-------------|----------|
| **FAISS не поддерживает удаление векторов** | Удаление документа → полный сброс индекса чата |
| **DOCX без номеров страниц** | Все чанки DOCX имеют `page=1` |
| **Требует OpenRouter API Key** | Без ключа LLM не работает (ошибка в UI) |
| **Только CPU FAISS** | `faiss-cpu` в зависимостях, GPU не используется |
| **Нет авторизации** | Однопользовательский, локальный доступ |
| **Нет оценки качества (eval)** | Нет встроенных метрик RAGAS и т.п. |

---

## Структура директорий

```
Local_RAG_project/
├── app.py                    # Streamlit UI entry point
├── pyproject.toml            # Package config, deps, scripts
├── requirements.txt          # Зависимости (pip)
├── .env.example              # Шаблон переменных окружения
├── .env                      # Локальные настройки (не в git)
├── README.md                 # Документация пользователя
├── audit.md                  # Этот файл
├── assets/
│   └── styles.css            # Кастомные стили Streamlit
├── data/
│   ├── raw/                  # Загруженные исходные документы (global)
│   ├── processed/            # (пусто, зарезервировано)
│   ├── vector_store/         # Глобальный FAISS индекс (legacy)
│   └── chats/                # Чат-сессии (новое)
│       └── chat_XXX/
│           ├── meta.json
│           ├── messages.json
│           ├── sources.json
│           ├── documents/    # Копии документов чата
│           └── index/        # FAISS индекс чата
├── scripts/
│   ├── ingest.py
│   ├── query.py
│   └── chat.py
├── src/
│   └── rag_project/
│       ├── __init__.py
│       ├── config.py
│       ├── models.py
│       ├── cli.py
│       ├── chat_store.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   └── chunking.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── sentence_transformers.py
│       ├── vectorstores/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── faiss_store.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── retriever.py
│       │   ├── reranker.py
│       │   └── citations.py
│       ├── llm/
│       │   ├── __init__.py
│       │   └── llm_client.py
│       ├── generation/
│       │   ├── __init__.py
│       │   └── llm.py
│       └── pipelines/
│           ├── __init__.py
│           ├── ingest_pipeline.py
│           ├── search_pipeline.py
│           ├── chat_pipeline.py
│           └── index_manifest.py
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_chunking.py
│   ├── test_docx_loader.py
│   ├── test_faiss_store.py
│   ├── test_index_manifest.py
│   ├── test_llm_client.py
│   ├── test_reranker_and_citations.py
│   ├── test_chat_store.py
│   └── test_chat_pipeline.py
└── .streamlit/
    └── config.toml
```

---

## Запуск и настройка

### Установка
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Конфигурация (`.env`)
```env
OPENROUTER_API_KEY=sk-or-v1-ваш-ключ
OPENROUTER_MODEL=openrouter/auto
TOP_K=5
MAX_CONTEXT_CHARS=12000
TEMPERATURE=0.2
USE_RERANKER=True
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANK_CANDIDATES=20
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120
```

### Запуск UI
```powershell
streamlit run app.py
```

### CLI команды
```powershell
# Индексация
.\.venv\Scripts\python.exe -m rag_project.cli ingest --data data/raw --index data/vector_store

# Поиск
.\.venv\Scripts\python.exe -m rag_project.cli search "О чем документы?" --index data/vector_store

# Чат
.\.venv\Scripts\python.exe -m rag_project.cli chat --index data/vector_store
```

---

## Тестирование

```powershell
# Все тесты
.\.venv\Scripts\python.exe -m pytest tests/ -v

# С покрытием
.\.venv\Scripts\python.exe -m pytest tests/ --cov=src/rag_project

# Линтинг
.\.venv\Scripts\python.exe -m ruff check src/ tests/
```

---

## Зависимости (requirements.txt / pyproject.toml)

**Основные:**
- `faiss-cpu>=1.8` — векторный поиск
- `sentence-transformers>=3.0` — эмбеддинги + CrossEncoder
- `openai>=1.0` — OpenRouter клиент (OpenAI-совместимый)
- `httpx>=0.28` — HTTP транспорт
- `numpy>=1.26` — массивы
- `pypdf>=4.0` — PDF парсинг
- `python-docx>=1.1` — DOCX парсинг
- `python-dotenv>=1.0` — загрузка .env
- `streamlit>=1.37` — веб UI

**Dev:**
- `pytest>=8.0`
- `ruff>=0.5`

---

## Резюме

**Local RAG Project** — это готовая к использованию локальная RAG-система с:
- ✅ Полным циклом: загрузка → чанкинг → эмбеддинги → FAISS → реранжирование → LLM
- ✅ Инкрементальной индексацией через манифесты (SHA256)
- ✅ Многопользовательскими чатами (изолированные индексы)
- ✅ Современным Streamlit UI (стриминг, прогресс-бары, дашборд)
- ✅ CLI для автоматизации
- ✅ Чёткой модульной архитектурой (Protocols, DI)
- ✅ Тестами и линтером

Проект подходит для локального анализа документов, поиска знаний и QA над приватными данными без утечки во внешние облака (кроме вызова LLM через OpenRouter).