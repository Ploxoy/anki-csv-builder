# Vision — Doedutch / Anki CSV Builder (KISS)

*(technical working doc — last updated: 2025-12-21)*

> Product vision and the migration plan to the next version live in `notes/vision_v2.md`.

---

## 1) Technologies

**Goal:** a working prototype with minimal stack and easy maintenance.

**Core stack**

* **Language:** Python 3.11+
* **UI (MVP):** Streamlit (fast to build, simple sessions, file upload)
* **UI (vNext):** TBD (move away from Streamlit; backend-first)
* **LLM SDK:** OpenAI Python SDK (Responses API)
* **Data/Export:** CSV (pipe‑delimited) and Anki `.apkg` via `genanki`
* **Utilities:** `pandas` (tables), `requests` (optional URL import)

**Intentionally NOT using (for MVP)**

* No database (no Postgres/SQLite). Artifacts are files. *(Note: a tiny SQLite cache keyed by `lemma|def|L1|CEFR|profile|prompt_version` can be added later if cost/speed require.)*
* No separate backend (no FastAPI/Flask). All logic runs inside Streamlit.
* No queues/workers. Concurrency will be minimal and optional.

**vNext (productization)**

* **Backend:** FastAPI on Railway (single service) with a small DB.
* **Auth:** Google OAuth via Supabase Auth.
* **Storage:** settings + usage metrics only (no job artifacts).
* **Billing:** prepaid balance + per-request charging (Stripe top-ups + webhooks).

**Deployment (MVP)**

* Streamlit Community Cloud first.
* Domain: start with default `*.streamlit.app`. Custom domain later (CNAME `www.doedutch.nl` → `share.streamlit.io`; root redirects to `www`). If DNS/host friction arises, keep the Streamlit subdomain for MVP.

**Models**

* Allowed prefixes: `gpt-5`, `gpt-4.1`, `gpt-4o`, `o3`.
* Filter out non‑text models containing: `audio`, `realtime`, `embed/embedding`, `whisper`, `moderation`, `search`, `vision`, `distill`, `batch`, `preview`.
* `temperature`: not supported on some 5th‑gen models (`gpt-5*`, `o3*`) — UI disables automatically.
* `max_output_tokens`: off by default; sentence length guided by CEFR word counts in prompt.

---

## 2) Development Principles

* **KISS** — prefer the simplest working solution.
* **MVP‑first** — ship a usable prototype for real users.
* **Iterative delivery** — small, frequent improvements.
* **No overengineering** — no microservices, no heavy abstractions.

**Git workflow**

* `main` — always working.
* `feature/*` — experiments → PR into `main`.
* Private notes/ideas live outside the repo or are ignored via `.gitignore` (`idea.md`, `notes/`).

**Code style**

* Python 3.11+, PEP8, short English comments.
* Minimal modules:

  * `app/app.py` — Streamlit UI only.
  * `core/*` — pure logic (parsing, prompts, LLM, generation, sanitize/validate, export).
  * `config/*` — settings, templates, i18n, signal word lists.

**Testing & QA**

* Manual tests on 5–10 items per run.
* Optional basic tests later (input parsing, JSON sanity).
* Success criterion: valid CSV/.apkg and readable cards in Anki.

**Release cycle**

* Every iteration produces a working export (CSV and/or .apkg).
* Docs (`README.md`, `vision.md`) evolve with the code.

---

## 3) Project Structure

```
anki-csv-builder/
├─ app/
│  └─ app.py                  # Streamlit UI only (sidebar, upload, preview, progress, export)
│
├─ core/
│  ├─ parsing.py              # input parsing (.txt/.md/markdown/tsv → normalized rows)
│  ├─ prompts.py              # CEFR/L1/profile instructions + strict JSON schema
│  ├─ llm_clients.py          # OpenAI Responses API calls; output_parsed/text extractors
│  ├─ generation.py           # 1 row → card orchestration; (later) micro-batches/parallel
│  ├─ sanitize_validate.py    # cloze normalization, forbidden chars, mandatory fields, separable verbs
│  ├─ export_csv.py           # CSV assembly (Anki fields / localized header)
│  ├─ export_anki.py          # .apkg assembly (genanki, templates, GUID policy)
│  └─ signalwords.py          # groups + balanced selection per CEFR
│
├─ config/
│  ├─ settings.py             # models, filters, defaults, CEFR rules, preview size
│  ├─ signalword_groups.py    # SIGNALWORD_GROUPS dictionary
│  ├─ templates_anki.py       # Front/Back HTML + CSS
│  └─ i18n.py                 # CSV header localization, L1 language labels
│
├─ README.md
├─ README.en.md               # optional English doc
├─ vision.md                  # this document
├─ requirements.txt
└─ .gitignore
```

**Principles**

* UI is separated from logic.
* Few small modules; each does one job.
* One‑way imports: `app → core → config`; `core` is Streamlit‑agnostic; `config` is logic‑free.

---

## 4) Architecture (MVP)

**Components**

* **UI (`app/app.py`)** — sidebar settings (model/CEFR/L1/profile/temperature/token limit), file upload or demo, parsed preview, progress, preview of first N cards, export buttons (CSV/APKG).
* **Core** — parsing, prompt building, signal word selection (balanced), LLM calls (Responses API; structured outputs via `text.format` with `json_schema`), generation (local cloze fixes, validate, single repair‑pass), export.
* **Config** — models/filters, CEFR rules, Anki templates, i18n, signal word groups.
* **External** — OpenAI. No DB, no queues.
* **FS** — generate CSV/APKG in memory and serve via `download_button`.

**Data flow**

1. UI collects settings & input.
2. `parsing.py` → normalized rows `{woord, def_nl, translation?}`.
3. For each row → `generation.py`:

   * build payload + small balanced set of signal words;
   * `llm_clients.py` call (Responses API + `response_format=json_schema`);
   * get `parsed` → `sanitize_validate.py` (fix cloze, validate fields);
   * if issues → **one** repair‑pass (cheap model) → local fixes;
   * append card to results.
4. UI shows progress & preview.
5. Export via `export_csv.py` and/or `export_anki.py`.

**State** (via `st.session_state`)

* `settings_snapshot`, `parsed_input`, `results`, `sig_usage`, `sig_last`, `no_temp_models`.
* `current_index`, `run_active`, `auto_advance`, `run_stats` for batch runs; recommended `batch_size`/`max_workers` applied safely (pending state + rerun) to avoid widget conflicts.
* **Clear** button resets results + signal word counters.

**Concurrency & batching**

* **Now:** micro‑batches with auto‑advance; parallelism inside a batch via `ThreadPoolExecutor(max_workers)`, results merged in input order; per‑item repair when needed.
* **Auto‑tuning:** recommended defaults computed from dataset size (≈20 items per batch, ≤10 workers), with adaptive reduction of workers on transient errors (429/timeout/5xx) for the next batch.
* **UI:** per‑batch progress (“Done/Active/Queued • time • rate”) and run summary (“batches, processed, elapsed, rate, errors”).

**Errors & resilience**

* Per‑card `try/except`, do not stop the run.
* One repair‑pass (always a cheap model, e.g., `gpt-4o-mini`), then local fixes; if still broken → skip with a clear message.
* Optional small delay between calls; exponential backoff for 429/5xx; adaptive Max workers across batches.

---

## 14) Text‑to‑Speech (TTS)

**GUID policy**

* `stable` = hash(L2\_word|L2\_definition|L1|CEFR|profile) — reimport updates notes.
* `unique` = `uuid4()` per export — always new notes.

**Security**

* API key from `st.secrets` or password field; never log the key.
* No external calls besides OpenAI.
* Do not persist user files on disk.

---

## 5) Data Model

### ParsedInputRow

Represents a row parsed from input file:

* `woord: str` — target Dutch word/lemma (required).
* `def_nl: str | ""` — dictionary definition (e.g., Van Dale) **or** a Dutch sentence where the word was found.
* `translation: str | ""` — translation into L1, if known. Optional.

### SettingsSnapshot

Snapshot of user settings at generation time:

* `model: str`
* `cefr: Literal[A1..C2]`
* `profile: str`
* `l1: Literal[RU,EN,ES,DE]`
* `temperature: float | None`
* `limit_tokens: bool`
* `max_output_tokens: int | None`
* `prompt_version: str`

### Card

Final generated card:

* `L2_word: str`
* `L2_cloze: str`
* `L1_sentence: str`
* `L2_collocations: str` — exactly 3, joined by `; `
* `L2_definition: str`
* `L1_gloss: str`
* `L1_hint: str` — currently empty, reserved
* `guid: str` — per selected policy (stable/unique)
* `tags: list[str]` — e.g. `["CEFR:B1","Profile:strict","L1:RU","Model:gpt-5"]`
* `error: str | ""` — if generation failed
* `meta: dict` — misc metadata (e.g., which signal word was injected, sentence length)

### RunState (in `st.session_state`)

* `parsed_input: list[ParsedInputRow]`
* `results: list[Card]`
* `sig_usage: dict[str,int]`
* `sig_last: str | None`
* `no_temp_models: set[str]`

---

## 6) Working with LLM (Responses API)

**Goal:** deterministic, schema‑valid JSON per card with minimal cost and latency.

### 6.1 Prompt construction (KISS)

* Build instructions in **English**; L2 is always Dutch (NL); L1 is user‑selected.
* Include CEFR constraints as **word‑count** guidance (e.g., A1: 6–9 words; B2: 12–16), grammar allowances, and style profile.
* Hard constraints (must not break):

  * Strict JSON object with required keys: `L2_word, L2_cloze, L1_sentence, L2_collocations, L2_definition, L1_gloss`.
  * No `|` in any field; 3 collocations joined by `; `.
  * Cloze: **exactly** `{{c1::…}}`; add `{{c2::…}}` only for separable verbs (particle at the particle position).
  * Prefer present tense; avoid names/digits/quotes; modern Dutch only.
  * Signal words: from B1, \~50% of sentences may include exactly one **mid‑clause** signal word; do not start the sentence with it.
* Payload (per row) includes:

  * `L2_word`, optional `given_L2_definition` (or example sentence), optional `given_L1_gloss` (= `translation`), `L1_code`, `CEFR`, `profile`.
  * `ALLOWED_SIGNALWORDS` — small balanced set (2–3) computed by `signalwords.py` for this row/level.

### 6.2 Structured outputs

* Use **Responses API** with `response_format={"type":"json_schema", "json_schema": ...}`.
* JSON Schema: `type: object`, `additionalProperties: false`, `minLength: 1` for all fields, `required: [...]`.
* Prefer reading `resp.output_parsed`; fallback to `output_text` JSON extraction only if needed.
* Результаты проб `response_format` кешируются в `cache/response_format.json`, чтобы не бомбить API при каждом запуске; UI предоставляет принудительное включение (force schema) при необходимости.

### 6.3 Temperature & tokens

* Do **not** send `temperature` to models that do not support it (e.g., `gpt-5*`, `o3*`).
* Token limit: **off by default**. Rely on CEFR word‑count in prompt; enable `max_output_tokens` only if the user opts in.

### 6.4 Single‑pass + repair‑pass

* **Primary pass:** selected model (can be `gpt-5` for best quality).
* Validate the parsed JSON locally:

  * All keys present & non‑empty; `L2_collocations` has exactly 3 items (split by `; `);
  * No `|`; cloze braces are double and balanced; separable verbs handled properly.
* **Repair‑pass (once):** if validation fails, send a short “fix‑only” instruction with the previous JSON to a **cheap model** (e.g., `gpt-4o-mini`).
* If still invalid → apply local sanitization (minimal edits) or mark the card as failed with an error message.

### 6.5 Signal words balancing

* For each row, compute a small candidate list via `signalwords.py` according to CEFR level (B1 minimal list; B2+ balanced across groups).
* Track usage in `st.session_state.sig_usage` and avoid immediate repetition (`sig_last`).
* After generation, detect which candidate (if any) appeared in `L2_cloze` and increment its counter.

### 6.6 Cloze normalization (post‑processing)

* Replace single/triple braces with double `{{…}}` where safe.
* Ensure only separable verbs get `{{c2::…}}` and place the particle correctly; do **not** add `c2` for non‑separable items.
* Strip/escape forbidden character `|` → replace with `∣`.

### 6.7 Errors, retries, backoff

* Per‑card try/except. Do not halt the whole run.
* On transient API errors (429/5xx): limited retries (e.g., 2) with exponential backoff.
* Respect user‑configured small inter‑request delay if set.

### 6.8 (Optional later) Caching & batching

* Caching (future): key = `hash(woord|def|L1|CEFR|profile|prompt_version)`.
* Micro‑batches (future): 5 items per request, parallel 3–4 workers; on batch failure, repair only failed items.
* Режим «большой список»: сканер на 100–500 слов, очереди заданий, отдельные лимиты для тяжёлых моделей (gpt‑5) и дешёвых repair-моделей, прогресс по блокам.

#### 6.8.1 Example Bank (beyond MVP)

* Долгоживущий «банк примеров» поверх кеша: вместо анонимного `hash(...)` хранить осмысленные записи «lemma + NL-слой + метаданные» с привязкой к версии промта/модели.
* На одно NL-определение/пример допустимо несколько L1: при повторном запросе с тем же `L2_word` и NL-описанием, но другим L1, переиспользуем NL-часть карточки (`L2_cloze`, `L2_collocations`, `L2_definition`) и генерируем только L1-поля (`L1_sentence`, `L1_gloss`) при их отсутствии.
* MVP остаётся без UI для управления этим банком; на следующих этапах возможны: просмотр/поиск по леммам, ручной отбор «любимых» примеров и экспорт curated-наборов.

### 6.9 Word validation (Dutch heuristic)

* Local, cheap heuristics before generation (no LLM calls):

  * Single token (no spaces), length 2–40, no digits or forbidden punctuation.
  * Latin letters with typical NL diacritics allowed; hyphen/apostrophe allowed inside.
  * Reject all‑caps acronyms (≥3 chars) as suspicious.
* Optional frequency check (if available): `wordfreq.zipf_frequency(woord, 'nl')` — accept if ≥2.0; 1.0–2.0 as rare; otherwise flag.
* UI: show ⚠︎ next to flagged `woord` with tooltip reason; add sidebar checkbox **“Force generate for flagged entries”** (off by default).
* Metrics: count flagged items in Run report.

---

## 7) LLM Monitoring (KISS)

**Goal:** minimal but useful visibility into generation quality, cost, and performance.

### 7.1 Metrics per run (session)

* Card counters: `total`, `ok`, `repaired`, `failed`.
* Failure reasons: `schema_fail`, `empty_field`, `cloze_error`, `collocations_count≠3`, `api_error(4xx/5xx)`, `timeout`.
* Signal words: frequency and distribution across groups.
* Sentence length: actual word count vs CEFR target range.
* Time: avg per card, per repair, per run.

### 7.2 Tokens & cost

* If `resp.usage` is available → sum `input_tokens`, `output_tokens`, `total`.
* If not available → show “n/a”.
* Cost estimation: small dict in `config.settings` (`PRICE_PER_1K_TOKENS`) with per‑model rates; compute `tokens × $/1K`.

### 7.3 UI display

* Collapsible **Run report** at bottom: metrics + **Download JSON** of aggregates.
* **Debug mode** (sidebar checkbox): keeps first N raw responses (truncated), only in memory, never saved.

### 7.4 Storage

* All metrics kept in `st.session_state` only. Clear button resets metrics.

### 7.5 Mini‑alerts in UI

* If repaired > 30% → suggest changing model/level/temp.
* If >20% sentences outside CEFR length → suggest stricter prompt or token limit.
* If one signal word dominates >40% → suggest diversifying pool.

### 7.6 Token accounting strategy

* Debug mode: only in memory (no disk).
* Export: JSON with aggregates only (no personal data, no prompts, no keys).
* Prices: manual dict in config; show estimate if usage present.
* If usage not present → do not add tokenizers (KISS).

---

## 7.1bis) Validation of `woord`

* Apply **cheap local heuristics** to ensure the target is a plausible Dutch word.
* Checks:

  * single token (no spaces), 2–40 characters.
  * Latin letters + Dutch diacritics allowed; digits and forbidden punctuation rejected.
  * All‑caps tokens of length ≥3 flagged as acronyms.
  * Regex `_TOKEN_RE` enforces allowed characters (`a–z`, umlauts, accents, hyphen, apostrophe).
* Optional: use `wordfreq` (`zipf_frequency`) for Dutch; accept if ≥1.0.
* If flagged, preview shows ⚠ with reason; user can override with **Force generate for flagged entries** (sidebar checkbox).
* Run report includes number/ratio of flagged words.
* Implemented in `core/sanitize_validate.py` (`is_probably_dutch_word`).

---

## 8) Usage Scenarios (KISS)

### 8.1 Quick start (Try demo)

1. Click **Try demo** → table shows 5–6 preset `{woord, def_nl}` rows.
2. Pick **Model / CEFR / Profile / L1**.
3. Click **Generate** → progress + preview of first N cards.
4. Download **CSV** and/or **.apkg**.
5. **Clear** to reset results and signal word counters.

### 8.2 Import from file

1. Upload `.txt/.md`/`tsv` (or paste text).
2. Preview normalized rows `{woord, def_nl, translation?}`.
3. Pick **Model / CEFR / Profile / L1** (+ optional temp/token limit).
4. **Generate** → progress, preview, **Run report**.
5. Download **CSV** / **.apkg**; **Clear** for a new run.

### 8.3 Manual input / quick edit

1. Switch to **Manual** tab → paste or type multiple rows into a text area.
2. Supported formats (same parser as file upload):
   * `woord`
   * `woord — definitie`
   * `woord — definitie — vertaling`
   * `woord ;; definitie ;; vertaling`
   * TSV (real tabs or literal `\t`)
   * Markdown table row `| woord | definitie | vertaling |`
3. Click **Parse & load text** (optionally append; clear button available), then generate as usual.

### 8.4 Errors & recovery

1. On invalid card → **one** repair‑pass (cheap model).
2. If still invalid → minimal local fixes or mark as *failed* with reason.
3. Preview shows failed items; export includes only valid cards.
4. Run report shows repaired/failed shares and reasons.

### 8.5 (Optional) Import by URL

1. Field **Import by URL** (server‑side fetch).
2. Size limit (≤2–5 MB), `Content-Type: text/*`.
3. Then same flow as 8.2.

---

## 9) Deployment (KISS)

**MVP target:** Streamlit Community Cloud.

### 9.1 Streamlit Cloud

* Connect GitHub repo → select app path `app/app.py` (or `anki_csv_builder.py` until refactor is complete).
* Python version: 3.11.
* `requirements.txt` pinned to compatible OpenAI SDK version (≥1.99.0 recommended).
* Secrets via **Streamlit Secrets** (cloud UI): `OPENAI_API_KEY`.
* Health policy: keep default timeouts; no background tasks.

### 9.2 Custom domain (later)

* Preferred: `www.doedutch.nl` → **CNAME** to Streamlit (`share.streamlit.io`).
* Root `doedutch.nl` → redirect (handled at registrar/Strato).
* SSL handled automatically by Streamlit (Let’s Encrypt) once DNS propagates.

### 9.3 Alternative hosting (later, if needed)

* If custom domain friction persists or you need more control:

  * Deploy as a container on a simple VM (e.g., Fly.io / Railway / Render).
  * Use reverse proxy (Caddy/NGINX) for SSL and domain.
* Still keep KISS: single container, no DB, stateless.

### 9.5 vNext: Railway (accounts + billing)

* Deploy as a single container on Railway (FastAPI service).
* Persistent Postgres (prefer Supabase Postgres if using Supabase Auth).
* Stripe webhooks handled by the same backend (public endpoint).
* Streamlit is a temporary UI only; long-term plan is to replace it.

### 9.4 Backups / artifacts

* No persistent server data. CSV/APKG are generated on demand and downloaded by the user.
* GitHub is the single source of truth for code and configuration.

---

## 10) Configuration Approach

### 10.1 Sources of config

* **Code config:** `config/` modules (`settings.py`, `templates_anki.py`, `signalword_groups.py`, `i18n.py`). Versioned in Git.
* **Secrets:** `.streamlit/secrets.toml` locally; Streamlit Cloud secrets in project settings.
* **Environment variables (optional):** allow `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ELEVENLABS_API_KEY` for overrides.

### 10.2 UI‑level toggles

* Model, CEFR, Profile, L1, temperature, token limit, CSV headers, GUID policy, Debug mode.
* Stored in `st.session_state` during the session; not persisted between sessions.

### 10.3 Defaults & safety

* Sensible defaults in `config/settings.py`.
* Validate user inputs (ranges, allowed values). Fallback to defaults on invalid.
* No runtime writes to files other than generated downloads.

### 10.4 Versioning

* `prompt_version` string increases when prompt/JSON schema changes.
* Useful for future cache keys and reproducibility.

---

## 11) Logging Approach (KISS)

### 11.1 Goals

* Visibility for debugging and user feedback.
* No PII/API keys in logs. No disk writes in MVP.

### 11.2 What we log

* In‑memory event list in `st.session_state` (visible in Debug mode UI):

  * start/end of generation, per‑card status (`ok/repaired/failed`), reason codes.
  * model used, CEFR/profile/L1 snapshot (not API keys).
  * API errors (status code, short message), retry counts.
* Console logging: minimal (`print`) in dev; off in cloud unless Debug is enabled.

### 11.3 Storage & retention

* Memory‑only (session lifetime). **Clear** wipes logs.
* No remote log sinks in MVP.

### 11.4 User‑visible diagnostics

* **Run report** (collapsible): метрики, токены по моделям, длины completion (raw vs финальные поля), доля repair/fallback, оценка стоимости (используются прайсы из `config/pricing.py`), alerts.
* Коммуникация по моделям: 5‑я серия (например, `gpt-5-mini`) даёт те же JSON‑длины, что и `gpt-4.1-mini`, но оплачивает скрытые reasoning‑токены. Документация и пресеты должны подчёркивать: `gpt-4.1-mini` — экономичный дефолт, `gpt-5-mini` — премиум по качеству и цене.
* Optional **Download JSON** with aggregated metrics (no prompts or secrets).

---

## 12) Risks & Assumptions

### 12.1 Risks

* **LLM variability:** even with strict JSON schema, outputs may deviate; handled by repair-pass + local fixes.
* **Cost & speed:** gpt-5 models are slower and more expensive; mitigated by allowing cheaper models and optional token limits.
* **ElevenLabs rate limits:** платформа жёстко ограничивает параллелизм и может выдавать 429; mitigated снижением воркеров до 2, экспоненциальным бэк-оффом и уважением HTTP Retry-After.
* **Signal words bias:** models may overuse certain connectors; balanced candidate sets reduce this, but not eliminate fully.
* **Domain/DNS friction:** custom domain integration with Streamlit Cloud may fail; fallback is to keep default subdomain.
* **Scalability:** sequential generation is slow for large lists; batching/parallelism planned but not MVP.
* **User data:** if very large files are uploaded, Streamlit session may crash; size limits mitigate.

### 12.2 Assumptions

* Users are language learners (NL L2) with basic tech skills.
* MVP is used with small/medium word lists (10–100 words).
* CSV and Anki `.apkg` are sufficient outputs for first phase.
* Only 4 L1 languages (RU, EN, ES, DE) supported initially.
* Internet connection and OpenAI API access are stable.

---

## 13) Roadmap MVP → v1.0

### MVP (now)

* Streamlit app on cloud, file upload + Try demo.
* CEFR, Profile, L1, Model selection.
* Generation via OpenAI Responses API → CSV/APKG export.
* Repair‑pass, cloze normalization, signal word injection.
* Monitoring in Run report; Debug mode in memory.

### Next steps (short term)

* Доделать потоковый прогресс и асинхронность в панели озвучки (OpenAI + ElevenLabs) с live-обновлением статуса.
* Рефакторинг Streamlit UI: вынести `generation_page.py` на отдельные модули (sidebar/run/audio/debug) и сократить количество побочных эффектов в UI-коде.
* Manual input editor (мини-таблица в UI) и импорт по URL/Google Drive.
* ✅ Улучшен Run report: токены/стоимость по моделям, доля fallbacks/repair, экспорт JSON с агрегатами.
* Optional SQLite cache с ключами `prompt_version` — оставить за флагом.
* Исследовать локализацию UI (EN/NL переключатель) на следующем этапе.

### Next steps (productization, vNext)

* User accounts: Google login (Supabase Auth), persist settings per user.
* Budget & billing: prepaid balance in EUR + per-request charging (tokens/audio) with a configurable markup strategy; Stripe top-ups.
* Deploy on Railway and start migrating away from Streamlit UI.


### Towards v1.0

* Thematic/idiom decks, not only single words.
* Audio roadmap: расширение TTS (больше голосов, пакетная генерация, предпрослушка).
* Images (optional, contextual).
* Прямой импорт в локальный Anki через AnkiConnect (HTTP API) без промежуточного сохранения файлов.
* User accounts (if needed) for deck sharing.
* More L1 languages.
* Business model experiment: free core + premium extras.

---

## 14) Text‑to‑Speech (TTS)

### 14.1 Goal

Разрешить постгенерационную озвучку: после того, как карточки готовы, пользователь нажимает отдельную кнопку, и для каждой создаются два MP3 (`AudioWord`, `AudioSentence`) с поддержкой ▶️ в Anki. Архитектура должна позволять смену провайдера.

### 14.2 Integration with Anki

* In card model, add two fields: `AudioWord`, `AudioSentence`.
* In templates: insert these fields; Anki interprets `[sound:filename.mp3]` and attaches media.
* In `.apkg`, files stored in `media_files` with mapping via `media.json` (handled automatically by genanki).

### 14.3 Abstraction layer

* `core/audio.py` централизует работу провайдеров через `ensure_audio_for_cards(...)`, принимающий список карточек, выбранный голос и параметры (word/sentence, инструкции, прогресс‑колбэк).
* `sentence_for_tts(cloze_text)` по‑прежнему убирает `{{c1::...}}/{{c2::...}}`, чтобы в озвучку попадала чистая NL-фраза.
* Имена файлов детерминированы: `word_<slug>__<voice>__<sha1_8>.mp3` и `sentence_<slug>__<voice>__<sha1_8>.mp3` (совпадает с текущей реализацией).

---

## 15) Accounts, Access & Billing (vNext)

**Goal:** move from “public app” to separated user access with saved settings and prepaid billing (no free tier).

### 15.1 Auth (Google)

* Google OAuth via **Supabase Auth**.
* App stores `user_id` (Supabase UUID) and uses JWT for backend requests.
* One account = one budget ledger and one settings namespace.

### 15.2 What we store (and what we do NOT)

* Store:
  * user profile (id/email/name)
  * per-user settings (model, L1, CEFR, profile, TTS prefs, etc.)
  * usage metrics (tokens/audio + estimated cost)
  * billing ledger (top-ups and charges)
* Do NOT store:
  * prompts, generated card JSON, exported files, audio files

### 15.3 Budget model (EUR)

* Prepaid balance in **EUR**.
* Each request produces a **usage event** that is priced and charged:
  * inputs: `model`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, audio usage (where available)
  * outputs: `raw_cost_eur` (provider cost estimate) and `charged_cost_eur` (user price)
* Charging is done after the API response (based on actual `usage`), with safety limits to avoid runaway costs.

### 15.4 Pricing strategy (markup)

* Store provider prices in code (`config/pricing.py`) and compute raw cost from usage.
* Apply a **markup multiplier** that is inverse to prepaid amount (bigger top-up → smaller multiplier).
* Implementation detail: keep this as a configurable tier table, e.g.:
  * top-up ≥ €X → multiplier M
  * store the applied tier/multiplier in each charge event for auditability.

### 15.5 Limits & anti-abuse

* Primary hard limit: **max words per run** (configurable).
* Dynamic controls (server-side): per-user concurrency, rate limits, and max tokens per request based on system load.
* Requests can require a minimum balance before starting a batch.

### 15.6 Admin mode

* Admin dashboard to:
  * search users, view balance and usage
  * manually adjust balance (credit/debit)
  * block/unblock accounts
  * inspect aggregated usage by model/date
* Сессионный кеш (`state.audio_cache`) и кеш результатов TTS (`summary.cache_hits`) исключают повторный вызов на одинаковый текст/голос в рамках одной сессии.
* Результат `ensure_audio_for_cards` возвращает `media_map` для экспорта и агрегат `AudioSynthesisSummary` с провайдером, голосом, статистикой успехов/ошибок.

### 14.4 Provider: OpenAI TTS (baseline)

* Базовая модель `gpt-4o-mini-tts-2025-12-15` (для совместимости также допускается `gpt-4o-mini-tts`); fallback `gpt-4o-tts` подключается автоматически, если основной ID недоступен.
* Используем SDK `client.audio.speech` (streaming → temp file → bytes). Все настройки (инструкции по стилю) передаются как простая строка.
* Разделяем инструкции по слову/предложению через пресеты в `config.settings.AUDIO_TTS_INSTRUCTIONS`.
* Использует тот же API-ключ, что и основная генерация, поэтому отдельного секрета не требуется.
* Качество речи стабильно, но голоса звучат «плоско»: мультиязычная модель иногда читает отдельные слова с англоязычным акцентом, поэтому воспринимается как минимум приемлемый, но не вдохновляющий baseline.

### 14.5 Provider: ElevenLabs (premium)

* В `config.settings.AUDIO_TTS_PROVIDERS` включён тип `elevenlabs`: модель `eleven_multilingual_v2`, язык `nl`, динамическая загрузка голосов.
* `fetch_elevenlabs_voices(api_key, language_codes=['nl'])` вытягивает каталог, фильтруя по NL; UI берёт динамический список и fallback-статический набор на случай пустого ответа.
* Вызовы идут на `https://api.elevenlabs.io/v1/text-to-speech/<voice>` с `spoken_language='nl'`, пользовательские voice settings подтягиваются из пресетов.
* Реализован экспоненциальный бэк-офф на HTTP 429/5xx + уважение `Retry-After`, параллелизм зажат до 2 воркеров.
* Голоса звучат чуть естественнее OpenAI, но возникают эксплуатационные расходы: токены тарифицируются по месяцам и сбрасываются по периодам, поэтому нужно следить за лимитом и показывать пользователю счётчики. Проблема англоязычного ударения на отдельных словах сохраняется.

### 14.6 UI настройки (актуальные)

* Аудио‑панель открывается после генерации: выбор провайдера (OpenAI/ElevenLabs), списка голосов, чекбоксы «слово/предложение», стили.
* Для ElevenLabs ключ берётся из `st.secrets`/env и автозаполняется; пользователю не нужно вводить его каждую сессию.
* Для каждого провайдера хранится последний выбранный голос (`state.audio_voice_map`), так что переключение туды‑сюды не сбрасывает выбор.
* Панель показывает оценку числа запросов, прогресс-бар batch генерации, сводку (успешно/пропущено, cache hits, fallbacks).

### 14.7 Производительность и стоимость (текущее состояние)

* Лимит воркеров для ElevenLabs ≤2 (чтобы не ловить 429), для OpenAI остаётся пользовательский слайдер 1–6.
* Повторное использование кеша (как внутри запуска, так и между кнопками Generate) существенно снижает стоимость на больших списках.
* В сводке отображаются пропуски/ошибки — пользователь понимает, что стоит перезапустить или скорректировать текст.
* Сравнительно с генерацией карточек TTS остаётся быстрым, поэтому переход на realtime TTS откладывается до появления чёткого выигрыша (например, для live-предпрослушки).
* Оценка стоимости TTS ведётся по длине текста (символы) и прайс-листу `config/pricing.py::AUDIO_MODEL_PRICING_USD_PER_1M_CHAR` (цены как на OpenAI — за 1M единиц).

### 14.8 Следующие шаги по TTS

* Сформировать кураторский список NL-голосов и стилистик (OpenAI + ElevenLabs) с пометкой «рекомендуемый»; подсветить в UI.
* Добавить лёгкую предпрослушку/оценку качества: кнопка «Preview» и чекбокс «отклонить» для дорожек с явным англоязычным акцентом.
* Реализовать стриминговые обновления прогресса (callback → перерисовка progress bar в режиме реального времени).
* Поддержать пост‑обработку (нормализация громкости, опция сохранить только уникальные файлы).

---

## 16) UI Refactor (app/app.py)

Goal: split current monolithic `app/app.py` (~1400 LOC) into small, testable modules:

- `app/ui_controls.py` — sidebar widgets, input tabs, recommendations, secrets.
- `app/run_controller.py` — batch orchestration (start/next/stop/auto), progress/run summary, error re‑run.
- `app/batch.py` — batch execution helpers (thread pool, merging, rate/adaptation policy).
- `app/tts_panel.py` — audio panel UI and synthesis orchestration.
- `app/debug_panel.py` — last request and SDK info.

Principles:
- View vs controller separation; no Streamlit state inside worker threads.
- Core logic remains in `core/*`; UI modules are thin.
- Keep public helpers small and cover with focused unit tests.

### 16.1 Usability & onboarding refresh

- **Preset wizard (🎛️ Preset selector)** — добавить вверху сайдбара блок с типовыми наборами настроек (Starter = B1 + RU + `gpt-4.1-mini`, Fast = B1 + RU + `gpt-4o`, Quality = B2 + RU + `gpt-4.1`). При выборе заполнять модель, CEFR, профиль, лимит токенов и `Force generate for flagged`.
- ✅ Базовая версия реализована: селектор пресетов живёт в сайдбаре, переключение на «Custom» возвращает ручной контроль.
- ⏳ TODO: сохранить отредактированные пользователем комбинации — либо через Streamlit `session_state` + `st.experimental_set_query_params`, либо через локальный кэш (cookies / `st.session_state.presets_user`). Нужно придумать удобное UX для «Save as preset».
- ✅ Простой поток внедрён: кнопка «Generate → Preview → Export» сверху над Upload/Manual запускает автогенерацию, а ручные Start/Next/Stop спрятаны в `Advanced run controls`.
- ✅ Onboarding & tips: добавлена карточка «Getting started» при пустом вводе, подсказки в сайдбаре (API, CEFR, профили) переведены на английский, причины `flagged_precheck` теперь раскрываются прямо в превью и в отдельном expander-е.
- **Guided sidebar tooltips** — снабдить ключевые контролы (API key, CEFR, профиль, batch) короткими, «мягкими» подсказками и ссылкой на справку. Добавить мини-список примеров `flagged_precheck` рядом с чекбоксом «Force generate…» (цифры, all caps, EN слова).
- **Simple generation flow** — объединить «Generate → Preview → Export» в один блок с call-to-action на главной панели; расширенные опции спрятать в `Advanced`.
- **Status onboarding** — при отсутствии данных или первом заходе показывать карточку с шагами («1. Подтяните API key», «2. Выберите preset», «3. Загрузите слова»).

### 16.2 Audio panel streamlining

- ✅ Quick TTS mode — кнопка «Quick TTS» синтезирует с дефолтными стилями/голосом, оставляя только чекбоксы «word/sentence»; расширенные настройки скрыты в `Advanced audio options`.
- ✅ ElevenLabs fetch UX — каталог голосов подгружается по кнопке «Fetch ElevenLabs voices», результаты кешируются на сессию без сброса API key.
- ⏳ Audio onboarding — добавить доп. подсказки/мини-гайд по выбору голоса после первого успешного запуска.

### 16.3 Run summary & diagnostics

- ✅ Run report 2.0 — итоговый модуль показывает модели, ретраи, schema-фолбэки, время/скорость, долю ошибок, usage сигнал-слов, оценку стоимости (LLM + аудио) и даёт JSON-экспорт.
- **Skip reasons** — расширить превью/индикаторы, чтобы рядом с `flagged_precheck` выводились короткие объяснения (цифры, EN, all caps) с подсказками, как устранить.

### 16.4 Iteration approach

- Минимизировать одновременно открытые refactor-таски: сначала внедрить preset wizard и simple flow, затем заняться TTS/ElevenLabs UX, после чего подключить Run report.
- Все пользовательские изменения сопровождать smoke-тестами (`streamlit run app/app.py`) и обновлёнными статус-заметками.

---

## 17) Multi-provider strategy (Text + TTS)

**Goal:** support non-OpenAI providers for (a) cheaper text generation and (b) better NL voices, while keeping strict JSON output, correct usage accounting, and predictable billing.

### 17.1 Non-negotiable requirements (acceptance gate)

We only add a provider/model if it supports:

* **Strict JSON output** (schema or equivalent hard JSON mode) suitable for our validator/repair-pass flow.
* **Usage reporting**: input/output tokens (and cached tokens if available). If a provider cannot report usage, it is not eligible for paid usage/budget tracking.
* **Batch stability**: consistent latency, low error rate, and clear retry semantics (429/5xx).

For TTS additionally:

* **Good NL voice quality** and controllable voice selection.
* **Transparent pricing unit** (per characters / per seconds / per audio tokens) + predictable limits.

### 17.2 Architecture (provider-first, without breaking MVP)

* Introduce provider interfaces:
  * `TextProvider` → returns `{card_json, usage, raw_text(optional), provider_request_id}`.
  * `TTSProvider` → returns `{audio_bytes, usage(chars/tokens/seconds), provider_request_id}`.
* Normalize a single **UsageEvent** shape across providers:
  * `provider`, `model`, `input_tokens`, `output_tokens`, `cached_tokens?`, `audio_chars?`, `audio_tokens?`, `seconds?`
  * `raw_cost_usd`, `raw_cost_eur` (FX), `charged_cost_eur` (markup), `pricing_unit`
* Keep current OpenAI path as the default provider; add new providers behind feature flags.

### 17.3 Pricing storage (multi-provider + units)

* Store provider prices by `(provider, model)` with explicit units:
  * text: USD per **1M tokens** (input/cached/output)
  * audio: USD per **1M chars** or per **1M audio tokens** (depending on provider)
* Billing for users is in **EUR**:
  * use a configurable FX rate (manual/periodic) to convert provider USD costs into EUR for ledger calculations.
  * apply a tiered markup multiplier (bigger top-up → smaller multiplier); store multiplier/tier per charge for audit.

### 17.4 First candidates (to evaluate)

**Text (goal: lower cost):**
* Keep OpenAI as baseline.
* Add 1 alternative provider that passes the acceptance gate (strict JSON + usage) and is measurably cheaper on our workload.
* **First pick:** Google **Gemini Flash** (fast/cheap class) — only if strict JSON + usage are reliable.

**TTS (goal: better voice + transparent billing):**
* Consider character-priced TTS providers with strong NL voices (typically cloud TTS offerings).
* **First pick:** **Azure Neural TTS** (NL voices + per-character billing).
* ElevenLabs remains a “premium option”, but its billing model may require either separate paid add-on pricing or BYOK for TTS (not for MVP).

### 17.5 Rollout plan (safe)

1. Build provider scaffolding + normalized UsageEvent + multi-unit pricing table (no UI change).
2. Add TTS provider #2 behind a feature flag; compare NL voice quality + cost/limits.
3. Add Text provider #2 behind a feature flag; compare cost per valid card + repair rate + speed.
4. Expose provider selection in UI only after we have stable metrics and clear pricing.

### 17.6 Evaluation matrix (how we decide)

We evaluate candidates on a fixed test set (e.g. 30–100 rows) and record results in Run report JSON.

**Text provider pass criteria**
* **Strict JSON validity:** ≥99% valid without manual fixes.
* **Repair rate:** ≤5% (or demonstrably lower cost even with repair).
* **Usage availability:** input/output tokens always present; cached tokens if supported, else `null`/0.
* **Stability:** transient error rate low; retries bounded; no systematic schema breaks.
* **Cost:** lower **€ per valid card** than OpenAI baseline on the same dataset/prompt_version.

**TTS provider pass criteria**
* **NL voice quality:** subjectively better than OpenAI baseline on word + sentence samples.
* **Billing transparency:** clear unit (chars/tokens/seconds) and stable pricing.
* **Batch stability:** predictable latency and low failure rate with retries.
* **Export compatibility:** produces MP3 (or converted) compatible with Anki media.

---

## 18) L1 Expansion Roadmap

**Target cohort:** broaden beyond RU/EN/ES/DE to cover key migrant and diaspora languages:

* **Arabic** (focus on Morocco, Syria, Iraq; MSA fallback when dialectal data scarce).
* **Turkish**.
* **Polish**.
* **English** (UI already EN-capable; ensure gloss localisation matches).
* **Ukrainian / Russian** (separate lemmatization rules, shared Cyrillic challenges).
* **Tigrinya**.
* **Farsi / Dari** (Iran / Afghanistan).
* **Amharic**.
* **Surinam languages** (Sranan Tongo, Hindustani/Hindi).
* **Indonesian languages** (Bahasa Indonesia + regional variants).

**Arabic-specific risks**

* Right-to-left rendering in Streamlit widgets and Anki templates; need CSS overrides and `[dir="rtl"]` wrappers.
* Optional diacritics (tashkil) vs. learner expectations; may require prompting LLM for transliteration + undiacritized text.
* Dialect vs. MSA variance: decide whether glosses are dialectal, MSA, or both; may need per-variant label.
* Multi-word gloss length >2 conflicts with current validation → adjust `L1_gloss` rule for Arabic scripts.
* Unicode normalization: merge Arabic presentation forms, Tatweel, Arabic-Indic digits before export.
* Punctuation spacing (، ؟) different from Latin; ensure sanitization preserves them and CSV delimiter replacement is safe.

**Non-Latin scripts considerations**

* Update prompts to clarify transliteration expectations (e.g., provide Latin transliteration alongside native script where helpful).
* CSV exports may need explicit UTF-8 BOM for Excel compatibility; test on Windows.
* Extend `L1_LANGS` config with per-language labels and localized CSV header strings.
* Review Run report UI for font coverage; consider embedding Noto fonts when exporting HTML/Anki.

**Delivery plan**

1. Survey demand/priority → pick pilot languages (likely Arabic + Turkish + Polish).
2. Prototype prompt adjustments and validation rules per script.
3. Add automated tests covering RTL rendering, CSV round-trip, Anki import check.
4. Document language-specific caveats in README and onboarding.

---
