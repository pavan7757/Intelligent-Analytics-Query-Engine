# Intelligent Analytics Query Engine

This application answers natural-language questions about the supplied sales data. A FastAPI backend interprets and validates analytical plans, Pandas calculates the results, and a React/Vite frontend displays the result table, explanation and confidence score.

## Architecture

```text
Question + schema + data dictionary + relevant feedback
  -> Gemini interpretation (or deterministic fallback)
  -> QueryPlan Pydantic validation
  -> semantic completeness and schema validation
  -> Pandas execution
  -> result + generated logic + explanation + confidence + provenance
  -> React frontend
```

Gemini receives the exact `QueryPlan.model_json_schema()` and must return one JSON object with the plan fields at the top level. It interprets the question; it does not calculate answers or supply executable code. Pydantic validation is mandatory. Semantic checks reject plans that omit required filters, change Boolean meaning, use the wrong entity granularity, or cannot represent the requested operation. Pandas executes only supported, validated plans.

If credentials or the SDK are unavailable, or a provider request, JSON parsing or Pydantic validation fails, the backend uses its deterministic interpreter. This path supports a bounded set of analytical language patterns and runs through the same completeness checks. A semantically incomplete interpretation is rejected rather than executed as a simpler question.

Every query response reports `interpreter_source` (`genai` or `deterministic_fallback`) and `fallback_used`. A correct numeric result alone does not establish that Gemini was used.

## Files and data

The supplied dataset files are present in the project root:

| File | Purpose |
| --- | --- |
| `sales_data.csv` | Sales observations and analytical fields |
| `targets.csv` | Additive targets and supported comparison dimensions |
| `data_dictionary.json` | Business terms, aliases and metric definitions |
| `nl_queries.json` | Assignment query corpus and intended logic |
| `feedback_log.csv` | Persisted interpretation corrections |

Both CSV files must exist because the loader reads them on each query. The sales data must contain rows for analytical execution. Missing JSON files load as empty dictionaries; include both supplied JSON files to retain business context and automatic dataset detection. The query corpus describes assignment expectations. Queries are interpreted and calculated from the CSV data; the corpus is never used as a question-to-answer lookup.

With an empty `DATASET_DIR`, configuration searches the project root and other local candidate directories for all four supplied dataset files. Set `DATASET_DIR` explicitly to an absolute directory when using another dataset. The feedback CSV is created there when needed, so this directory must be writable.

```text
project/
  backend/app/          API, configuration, planning, validation and execution
  backend/tests/        Backend regression tests
  backend/requirements.txt
  frontend/src/         React UI and API client
  frontend/package.json
  .env.example          Placeholder configuration
  .gitignore
  sales_data.csv
  targets.csv
  data_dictionary.json
  nl_queries.json
  README.md
```

## Setup

Install Python and Node.js/npm. The commands below use Windows PowerShell, starting in the project root. Keep the backend and frontend running in separate terminals.

### Environment configuration

Create `.env` from the example only if it does not already exist:

```powershell
if (!(Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
notepad .env
```

Before starting the backend, replace or clear every credential placeholder, and replace the example `DATASET_DIR=C:\path\to\dataset`. Leave `DATASET_DIR` empty to use the files already in the project root. A Gemini configuration has this shape:

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=
LLM_API_KEY=
LLM_BASE_URL=
DATASET_DIR=
FRONTEND_ORIGIN=http://localhost:5173
```

Replace the Gemini placeholder locally with your own key. To run without provider calls, leave **all three** credential variables empty; a placeholder is a nonempty credential and will otherwise be sent to the provider.

The backend loads only the project-root `.env`, resolved from `backend/app/config.py`, with `override=False`. Existing process environment variables take precedence over `.env`. Restart the backend after changing configuration.

Optional backups use `OPENROUTER_API_KEY` with `OPENROUTER_MODEL`, `GROQ_API_KEY` with `GROQ_MODEL`, and `XAI_API_KEY` with `XAI_MODEL`. Groq and Grok/xAI are separate providers: a Groq key belongs in `GROQ_API_KEY` and is sent only to the Groq endpoint. Gemini retries HTTP 502, 503 and 504 failures up to three total attempts before trying backups. Provider fallback details appear as a small note below the result.

| Variable | Behavior |
| --- | --- |
| `LLM_PROVIDER` | Defaults to `gemini`; Gemini aliases `google` and `google_gemini` are accepted |
| `LLM_MODEL` | Gemini default is `gemini-3.5-flash`; configured legacy model names are normalized by the application |
| `LLM_API_KEY` | First credential choice, including when Gemini is selected |
| `GEMINI_API_KEY` | Second credential choice |
| `GOOGLE_API_KEY` | Third credential choice; Gemini alias |
| `LLM_BASE_URL` | Optional OpenAI-compatible endpoint override; the Gemini SDK path does not use it |
| `DATASET_DIR` | Absolute dataset directory, or empty for automatic detection |
| `FRONTEND_ORIGIN` | Allowed browser origin; defaults to `http://localhost:5173` |

Credential precedence is **`LLM_API_KEY` > `GEMINI_API_KEY` > `GOOGLE_API_KEY`**, selecting the first nonempty value. Clear unintended overrides when configuring Gemini. Model access and quota depend on the provider account; provider failures activate fallback. No real credentials belong in source files, `.env.example`, screenshots or submissions.

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Reuse an existing virtual environment instead of recreating it. These commands do not require PowerShell script activation. Interactive API documentation is available at `http://127.0.0.1:8000/docs`. `GET /health` confirms that the API process is running; use `GET /schema` to check dataset loading.

### Frontend

In a second terminal, starting from the project root:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. The default API URL is `http://localhost:8000`. For another API address, set `VITE_API_URL` before starting or building Vite. If Vite uses another port or host, set `FRONTEND_ORIGIN` to the browser's exact origin and restart the backend.

## APIs and response states

| Endpoint | Request and behavior |
| --- | --- |
| `GET /health` | Returns `{"status":"ok"}` |
| `GET /schema` | Returns columns, dtypes, dictionary and query corpus; dataset-loading failures return HTTP 503 |
| `POST /query` | JSON body `{"query":"your analytical question"}` |
| `POST /feedback` | Persists a correction using the fields below |

Query responses contain `query`, `generated_logic`, `result`, `confidence_score` and `explanation`, plus `query_plan`, `warnings`, `is_supported`, `unresolved_terms`, `interpreter_source` and `fallback_used`.

- A supported query returns HTTP 200. An empty record/group result is distinct from an unsupported question; an aggregate over an empty selection may still return a scalar result such as a sum of zero.
- An unsupported interpretation returns HTTP 200 with `is_supported=false`, confidence 0, no generated calculation logic, an empty result and an explanation of the unresolved requirements.
- Validation or analytical execution failures return a controlled HTTP 422 with `detail`. API or connectivity errors are displayed separately by the frontend.

The frontend clears previous results when a new query is submitted and after a failed request. It distinguishes unsupported questions from valid queries with no matching records and displays safe backend error details rather than leaving an old result visible.

## Supported analytics

- Sum, mean, count, minimum and maximum over supported metrics.
- Single- and multi-dimensional grouping, including comma-separated dimensions.
- Categorical filtering and numeric `>`, `>=`, `<`, `<=`, `=`, `==` and `!=` predicates. QueryPlan represents equality as `=`.
- Explicit Boolean AND/OR composition for safely representable filter expressions; unrepresentable expressions are rejected.
- Matching-record queries without unintended aggregation.
- Highest/lowest rankings, global Top N and Top N within partitions. Explicit numeric or number-word counts take precedence over an implicit single highest result.
- Contribution/share/percentage-of-total calculations over grouped results.
- Filter + grouping + ranking combinations within one aggregation stage.
- Overall or grouped actual-versus-target comparisons when the target data has compatible fields and additive grain, including below/above/met-target result filtering.
- Aggregate ratios, consecutive-calendar-year growth, and aggregation of ranked selections represented by validated plan fields.
- Month, year, quarter, week and day grouping, explicit year filters, and supported absolute date boundaries.

Fields are resolved against physical columns, detected types and dictionary mappings. Bare customer entities resolve to identity fields when available, while customer segments remain a separate dimension. Revenue can be derived as `quantity * unit_price * (1 - discount)` when absent as a physical column. The engine does not evaluate arbitrary dictionary formulas; a dictionary definition alone does not make every derived metric executable.

## Feedback and confidence

Feedback requires `query`, `incorrect_interpretation` and `correct_interpretation`; `notes` is optional. For example:

```json
{
  "query": "your previous question",
  "incorrect_interpretation": "the interpretation to correct",
  "correct_interpretation": "the intended meaning or field mapping",
  "notes": "optional context"
}
```

`POST /feedback` appends these fields and a timestamp to `DATASET_DIR/feedback_log.csv`. Later interpretation retrieves at most the first five stored rows whose query contains the current query as a case-insensitive substring, and supplies their corrections to the provider prompt. This is limited prompt reuse, not model training, semantic search or guaranteed correction application. Deterministic fallback does **not** apply feedback corrections. CSV storage assumes local use and does not provide a concurrent multi-user storage system.

Confidence is a deterministic score from 0 to 1, not a calibrated probability. It combines interpretation confidence with metric, grouping, filter, date, target and schema checks. Ambiguities reduce the score; empty result lists cap it at 0.5. Unsupported or semantically incomplete plans receive 0 and are not executed. Fallback starts with lower interpretation confidence; successful schema checks do not prove GenAI provenance.

Generated logic and explanations describe the validated, executed filters, Boolean predicates, grouping, ranking, limits, partitions, target comparison and assumptions, including whether interpretation used Gemini or fallback.

## Known limitations and assumptions

- Relative dates such as "last month", "today" and "year to date" are unsupported and explicitly rejected. Use supported absolute dates or years.
- Named months and abbreviations support calendar filtering with an optional year. An omitted year uses the only year in a single-year dataset; otherwise it includes that month across all dataset years, with an explicit assumption. Monthly targets are filtered to the same calendar period.
- Average order value (AOV) calculates aggregate revenue divided by non-null order-ID count, with null for a zero denominator. YoY growth joins consecutive calendar-year totals within each requested group; missing previous years and zero baselines produce null. Filtering an output year retains its prior-year baseline. Arbitrary formula metrics remain unsupported.
- Ranked selection followed by aggregation is supported for requests such as total profit of the top two products per country: first aggregate each entity, rank within its partition, select N, then aggregate selected values by the partition. Other multi-stage pipelines are rejected when they cannot be represented safely. Growth combined with ranking, ratios or target comparison requires further stages and is rejected.
- Explicit "sort/order by" clauses, including sorting by a separate field, are unsupported. Use supported highest/lowest rankings, which order by the analytical metric.
- Median, percentile, variance and other unimplemented aggregations are rejected. Unknown fields, operators and essential constraints must be resolved or rejected.
- Target comparisons require compatible additive totals. Incompatible grouping, filtering or time grain returns a controlled analytical error.
- When multiple datetime columns cannot be resolved uniquely, specify the intended date field. Invalid datetime values can prevent execution.
- Count aggregation counts matching rows rather than distinct entities. Contribution percentages use the grouped, filtered selection as their denominator. Top N uses the requested row count and does not include extra tied rows.
- Pandas loads local CSV data into memory; this is an assignment-scale analytical application, without authentication, a production database or a general SQL/Python execution interface.

## Verification and submission

From the backend directory:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

From the frontend directory:

```powershell
npm run build
```

The backend regression tests exercise deterministic execution and mocked provider paths without requiring a fresh Gemini call. `npm run build` produces the frontend bundle in `frontend/dist`. The frontend has no dedicated test script; verify supported results, unsupported questions, empty results and failed requests in the browser, including that a failed request clears the previous result.

Keep the dataset files, source, tests, lockfile and `.env.example` in the submission. Exclude `.env`, virtual environments (`backend/venv`, `venv`, `.venv`), Python caches, `node_modules` and generated build output. `.gitignore` applies to Git operations; manually created ZIP submissions must also omit these directories and secret files. Recheck the archive itself before uploading.

