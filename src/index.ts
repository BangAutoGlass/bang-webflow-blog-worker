import OpenAI from "openai"

const DEFAULT_EDGE_REQUEST_TIMEOUT_SECONDS = 90
const DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS = 900
const DEFAULT_WEBFLOW_REQUEST_TIMEOUT_SECONDS = 120
const DEFAULT_WORKER_BATCH_SIZE = 10
const DEFAULT_WORKER_CONCURRENCY = 5
const MAX_WORKER_CONCURRENCY = 10
const DEFAULT_STALE_LOCK_MINUTES = 45
const DEFAULT_POLL_INTERVAL_MS = 5000
const DEFAULT_IDLE_POLL_INTERVAL_MS = 60000
const DEFAULT_ERROR_SLEEP_MS = 30000
const DEFAULT_HEARTBEAT_MS = 30000
const DEFAULT_WEBFLOW_API_BASE = "https://api.webflow.com/v2"

const EXPECTED_BODY_MODEL = "gpt-5.5"
const EXPECTED_FIELD_MODEL = "gpt-5.4"
const EXPECTED_BODY_REASONING_EFFORT = "xhigh"
const EXPECTED_FIELD_REASONING_EFFORT = "medium"

type StepKind = "body_generation" | "title_and_field_generation" | "webflow_publish"
type JsonRecord = Record<string, unknown>

type WorkItem = {
  jobId: string
  rowId: string
  stepId: string
  phase: StepKind
  stepNumber: 1 | 2 | 3
  model: string
  modelSecretName: string
  reasoningEffort: string
  webSearchTool: string
  job: JsonRecord
  row: JsonRecord
  request: JsonRecord
}

type EdgeClaimResponse = JsonRecord & {
  success?: boolean
  action?: string
  claimedTotal?: number
  workItems?: WorkItem[]
  message?: string
  job?: JsonRecord
}

type PhaseResult = {
  success: boolean
  jobId: string
  rowId: string
  stepId: string
  phase: StepKind
  durationMs: number
  openaiResponseId?: string
  webflowItemId?: string
  error?: string
}

const shutdownState = {
  shuttingDown: false,
}

function envValue(name: string) {
  const value = process.env[name]
  return value && value.trim() ? value.trim() : ""
}

function getRequiredEnv(name: string) {
  const value = envValue(name)
  if (!value) throw new Error(`Missing required environment variable: ${name}`)
  return value
}

function getRequiredAnyEnv(names: string[]) {
  for (const name of names) {
    const value = envValue(name)
    if (value) return value
  }
  throw new Error(`Missing required environment variable. Set one of: ${names.join(", ")}`)
}

function boolEnv(name: string, fallback: boolean) {
  const value = envValue(name).toLowerCase()
  if (["true", "1", "yes", "y", "on"].includes(value)) return true
  if (["false", "0", "no", "n", "off"].includes(value)) return false
  return fallback
}

function numberEnv(name: string, fallback: number, min: number, max: number) {
  const parsed = Number(process.env[name])
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(min, Math.min(max, Math.floor(parsed)))
}

function firstString(...values: unknown[]) {
  for (const value of values) {
    const text = stringifyValue(value).trim()
    if (text) return text
  }
  return ""
}

function stringifyValue(value: unknown) {
  if (value === null || value === undefined) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return ""
  }
}

function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

function asRecord(value: unknown, fallback: JsonRecord = {}) {
  return isRecord(value) ? value : fallback
}

function cleanPlainObject(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(cleanPlainObject)
  if (!isRecord(value)) return value

  const output: JsonRecord = {}
  for (const [key, innerValue] of Object.entries(value)) {
    if (["__proto__", "constructor", "prototype"].includes(key)) continue
    output[key] = cleanPlainObject(innerValue)
  }
  return output
}

function compactObject(value: JsonRecord) {
  const output: JsonRecord = {}
  for (const [key, innerValue] of Object.entries(value)) {
    if (innerValue === undefined || innerValue === null) continue
    if (typeof innerValue === "string" && innerValue.trim() === "") continue
    if (Array.isArray(innerValue) && innerValue.length === 0) continue
    output[key] = innerValue
  }
  return output
}

function safeErrorMessage(error: unknown, maxLength = 5000) {
  let raw = "Unknown error"
  if (error instanceof Error) raw = error.message
  else if (typeof error === "string") raw = error
  else if (isRecord(error)) {
    const parts = [
      error.message ? `message=${String(error.message)}` : "",
      error.details ? `details=${String(error.details)}` : "",
      error.hint ? `hint=${String(error.hint)}` : "",
      error.code ? `code=${String(error.code)}` : "",
    ].filter(Boolean)

    if (parts.length) raw = parts.join(" | ")
    else {
      try {
        raw = JSON.stringify(error)
      } catch {
        raw = Object.prototype.toString.call(error)
      }
    }
  } else if (error !== undefined && error !== null) raw = String(error)

  return raw.length > maxLength ? `${raw.slice(0, maxLength)}... [truncated]` : raw
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function logJson(payload: JsonRecord, level: "debug" | "info" | "warn" | "error" = "info") {
  const configured = envValue("LOG_LEVEL") || "info"
  const order = { debug: 10, info: 20, warn: 30, error: 40 }
  if (order[level] < (order as Record<string, number>)[configured] && configured !== "debug") return
  console.log(JSON.stringify({ time: new Date().toISOString(), level, ...payload }, null, 2))
}

function buildQuery(params: unknown) {
  const search = new URLSearchParams()
  const record = asRecord(params)

  for (const [key, value] of Object.entries(record)) {
    if (value === undefined || value === null || value === "") continue
    if (Array.isArray(value)) {
      for (const item of value) search.append(key, String(item))
    } else {
      search.set(key, String(value))
    }
  }

  const query = search.toString()
  return query ? `?${query}` : ""
}

function normalizeUrl(base: string, path: string, query?: unknown) {
  const cleanBase = String(base || "").replace(/\/+$/, "")
  const cleanPath = String(path || "").startsWith("/") ? String(path || "") : `/${String(path || "")}`
  return `${cleanBase}${cleanPath}${buildQuery(query)}`
}

async function fetchWithTimeout(url: string, init: RequestInit = {}, label = "request", timeoutMs = 30000) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)

  try {
    return await fetch(url, { ...init, signal: controller.signal })
  } catch (error) {
    if (controller.signal.aborted || /abort|timeout|timed out/i.test(safeErrorMessage(error))) {
      throw new Error(`${label} timed out after ${Math.round(timeoutMs / 1000)}s.`)
    }
    throw error
  } finally {
    clearTimeout(timeout)
  }
}

async function callJson(url: string, init: RequestInit = {}, options: {
  label?: string
  timeoutMs?: number
  retries?: number
  retryStatuses?: number[]
} = {}) {
  const {
    label = "request",
    timeoutMs = 30000,
    retries = 1,
    retryStatuses = [408, 409, 425, 429, 500, 502, 503, 504],
  } = options

  let lastError: unknown = null

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const response = await fetchWithTimeout(url, init, label, timeoutMs)
      const text = await response.text()
      let data: unknown = null

      try {
        data = text ? JSON.parse(text) : null
      } catch {
        data = { raw: text.slice(0, 4000) }
      }

      if (!response.ok) {
        const shouldRetry = attempt < retries - 1 && retryStatuses.includes(response.status)
        if (shouldRetry) {
          const retryAfter = Number(response.headers.get("retry-after") || 0)
          const waitMs = retryAfter
            ? Math.min(retryAfter * 1000, 30000)
            : Math.min(1000 * Math.pow(2, attempt), 15000) + Math.floor(Math.random() * 250)
          await sleep(waitMs)
          continue
        }

        throw new Error(`${label} failed with HTTP ${response.status}: ${JSON.stringify(data).slice(0, 4000)}`)
      }

      return data
    } catch (error) {
      lastError = error
      if (attempt < retries - 1) {
        await sleep(Math.min(1000 * Math.pow(2, attempt), 15000) + Math.floor(Math.random() * 250))
        continue
      }
      throw lastError
    }
  }

  throw lastError || new Error(`${label} failed.`)
}

const config = {
  edgeFunctionUrl: getRequiredEnv("EDGE_FUNCTION_URL"),
  workerSecret: getRequiredAnyEnv(["WORKER_SECRET", "WEBFLOW_BLOG_CREATION_WORKER_SECRET"]),
  workerId: envValue("WORKER_ID") || `render-blog-worker-${process.pid}`,
  openaiApiKey: getRequiredEnv("OPENAI_API_KEY"),
  webflowToken: envValue("WEBFLOW_API_TOKEN") || envValue("WEBFLOW_OAUTH_TOKEN"),
  webflowApiBase: envValue("WEBFLOW_API_BASE") || DEFAULT_WEBFLOW_API_BASE,
  workerEnabled: boolEnv("WORKER_ENABLED", true),
  workerBatchSize: numberEnv("WORKER_BATCH_SIZE", DEFAULT_WORKER_BATCH_SIZE, 1, MAX_WORKER_CONCURRENCY),
  workerConcurrency: numberEnv("WORKER_CONCURRENCY", DEFAULT_WORKER_CONCURRENCY, 1, MAX_WORKER_CONCURRENCY),
  staleLockMinutes: numberEnv("WORKER_STALE_LOCK_MINUTES", DEFAULT_STALE_LOCK_MINUTES, 0, 1440),
  pollIntervalMs: numberEnv("WORKER_POLL_INTERVAL_MS", DEFAULT_POLL_INTERVAL_MS, 250, 300000),
  idlePollIntervalMs: numberEnv("WORKER_IDLE_POLL_INTERVAL_MS", DEFAULT_IDLE_POLL_INTERVAL_MS, 1000, 900000),
  errorSleepMs: numberEnv("WORKER_ERROR_SLEEP_MS", DEFAULT_ERROR_SLEEP_MS, 1000, 900000),
  heartbeatMs: numberEnv("WORKER_HEARTBEAT_MS", DEFAULT_HEARTBEAT_MS, 5000, 300000),
  edgeRequestTimeoutMs: numberEnv("EDGE_REQUEST_TIMEOUT_SECONDS", DEFAULT_EDGE_REQUEST_TIMEOUT_SECONDS, 10, 300) * 1000,
  edgeMaxRetries: numberEnv("EDGE_MAX_RETRIES", 3, 1, 8),
  openaiRequestTimeoutMs: numberEnv("OPENAI_REQUEST_TIMEOUT_SECONDS", DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS, 30, 1800) * 1000,
  openaiMaxRetries: numberEnv("OPENAI_MAX_RETRIES", 3, 1, 8),
  webflowRequestTimeoutMs: numberEnv("WEBFLOW_REQUEST_TIMEOUT_SECONDS", DEFAULT_WEBFLOW_REQUEST_TIMEOUT_SECONDS, 10, 300) * 1000,
  webflowMaxRetries: numberEnv("WEBFLOW_MAX_RETRIES", 4, 1, 10),
  keepAliveWhenDone: boolEnv("KEEP_ALIVE_WHEN_DONE", true),
  stopOnFatalError: boolEnv("STOP_ON_FATAL_ERROR", false),
  storeFullOpenAIResponse: boolEnv("STORE_FULL_OPENAI_RESPONSE", false),
  validateWebflowFields: boolEnv("VALIDATE_WEBFLOW_FIELDS", true),
}

const openai = new OpenAI({ apiKey: config.openaiApiKey })

async function callEdge(action: string, extra: JsonRecord = {}) {
  const body = cleanPlainObject({
    action,
    workerId: config.workerId,
    workerBatchSize: config.workerBatchSize,
    workerConcurrency: config.workerConcurrency,
    resetStaleMinutes: config.staleLockMinutes,
    includeRows: false,
    ...extra,
  })

  const data = await callJson(
    config.edgeFunctionUrl,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-worker-secret": config.workerSecret,
      },
      body: JSON.stringify(body),
    },
    {
      label: `Edge Function ${action}`,
      timeoutMs: config.edgeRequestTimeoutMs,
      retries: config.edgeMaxRetries,
    }
  )

  const record = asRecord(data)
  if (!record || record.success === false) {
    throw new Error(`Edge Function ${action} failed: ${JSON.stringify(record).slice(0, 4000)}`)
  }

  return record
}

async function sendHeartbeat(item: WorkItem) {
  try {
    await callEdge("heartbeat", {
      jobId: item.jobId,
      rowId: item.rowId,
      stepId: item.stepId,
      phase: item.phase,
    })
  } catch (error) {
    logJson({ event: "heartbeat_failed", rowId: item.rowId, phase: item.phase, error: safeErrorMessage(error, 1500) }, "warn")
  }
}

function getOutputText(response: any) {
  if (typeof response?.output_text === "string" && response.output_text.trim()) {
    return response.output_text.trim()
  }

  const parts: string[] = []
  for (const item of response?.output || []) {
    if (item?.type === "message" && Array.isArray(item.content)) {
      for (const content of item.content) {
        if (content?.type === "output_text" && typeof content.text === "string") parts.push(content.text)
      }
    }
  }

  return parts.join("\n").trim()
}

function parseJsonFromText(text: string, label: string) {
  if (!String(text || "").trim()) throw new Error(`${label} returned an empty response.`)

  try {
    return JSON.parse(text)
  } catch {
    const firstBrace = text.indexOf("{")
    const lastBrace = text.lastIndexOf("}")
    if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
      throw new Error(`${label} was not valid JSON.`)
    }
    return JSON.parse(text.slice(firstBrace, lastBrace + 1))
  }
}

function extractToolCalls(response: any) {
  const output = Array.isArray(response?.output) ? response.output : []
  return output
    .filter((item: any) => {
      const type = String(item?.type || "")
      return type.includes("web_search") || type.includes("tool")
    })
    .map((item: any) => cleanPlainObject(item))
}

function makeSafeOpenAIResponsePayload(response: any) {
  if (config.storeFullOpenAIResponse) return cleanPlainObject(response) as JsonRecord

  return cleanPlainObject({
    id: response?.id || null,
    model: response?.model || null,
    created_at: response?.created_at || null,
    status: response?.status || null,
    outputText: getOutputText(response),
    usage: response?.usage || {},
  }) as JsonRecord
}

function expectedReasoningEffortForPhase(phase: StepKind) {
  if (phase === "title_and_field_generation") return EXPECTED_FIELD_REASONING_EFFORT
  if (phase === "body_generation") return EXPECTED_BODY_REASONING_EFFORT
  return firstString(EXPECTED_BODY_REASONING_EFFORT, "medium")
}

function normalizeReasoningForItem(item: WorkItem, request: JsonRecord) {
  const requestReasoning = asRecord(request.reasoning)
  const expectedEffort = expectedReasoningEffortForPhase(item.phase)

  // Cost guardrail: field generation must stay on medium reasoning even if an older
  // Edge claim payload still contains xhigh from a previously queued job.
  if (item.phase === "title_and_field_generation") {
    return {
      ...requestReasoning,
      effort: EXPECTED_FIELD_REASONING_EFFORT,
    }
  }

  return {
    ...requestReasoning,
    effort: firstString(requestReasoning.effort, item.reasoningEffort, expectedEffort),
  }
}

function buildOpenAIRequestPayload(item: WorkItem) {
  const request = asRecord(item.request)
  const payload: JsonRecord = {
    model: item.model,
    instructions: firstString(request.instructions),
    input: firstString(request.input),
    tools: Array.isArray(request.tools) ? request.tools : [{ type: firstString(item.webSearchTool, "web_search_preview") }],
    reasoning: normalizeReasoningForItem(item, request),
    text: isRecord(request.text) ? request.text : undefined,
    store: request.store === true,
  }

  return compactObject(payload)
}

function getOpenAILabel(item: WorkItem) {
  if (item.phase === "body_generation") return "OpenAI blog body generation"
  if (item.phase === "title_and_field_generation") return "OpenAI CMS field generation"
  return "OpenAI request"
}

function validateWorkItemModel(item: WorkItem) {
  if (item.phase === "body_generation" && item.model !== EXPECTED_BODY_MODEL) {
    throw new Error(`Body generation must use ${EXPECTED_BODY_MODEL}; received ${item.model}.`)
  }

  if (item.phase === "title_and_field_generation" && item.model !== EXPECTED_FIELD_MODEL) {
    throw new Error(`Field generation must use ${EXPECTED_FIELD_MODEL}; received ${item.model}.`)
  }
}

function validateOpenAIPhaseConfig(item: WorkItem, payload: JsonRecord) {
  validateWorkItemModel(item)

  const effort = firstString(asRecord(payload.reasoning).effort)
  const expectedEffort = expectedReasoningEffortForPhase(item.phase)

  if (item.phase === "title_and_field_generation" && effort !== EXPECTED_FIELD_REASONING_EFFORT) {
    throw new Error(`Field generation must use ${EXPECTED_FIELD_REASONING_EFFORT} reasoning; received ${effort || "none"}.`)
  }

  if (item.phase === "body_generation" && !effort) {
    throw new Error(`Body generation is missing reasoning effort. Expected ${expectedEffort}.`)
  }
}

async function createOpenAIResponse(payload: JsonRecord, label: string) {
  let lastError: unknown = null

  for (let attempt = 0; attempt < config.openaiMaxRetries; attempt++) {
    try {
      return await openai.responses.create(payload as any, {
        signal: AbortSignal.timeout(config.openaiRequestTimeoutMs),
      } as any)
    } catch (error) {
      lastError = error
      const status = Number((error as any)?.status || (error as any)?.code || (error as any)?.response?.status || 0)
      const message = safeErrorMessage(error)
      const retryable =
        status === 408 ||
        status === 409 ||
        status === 425 ||
        status === 429 ||
        status >= 500 ||
        /rate.?limit|timeout|temporarily|overloaded|aborted/i.test(message)

      if (!retryable || attempt >= config.openaiMaxRetries - 1) break

      const retryAfter = Number(
        (error as any)?.headers?.["retry-after"] ||
          (error as any)?.headers?.get?.("retry-after") ||
          0
      )
      const waitMs = retryAfter
        ? Math.min(retryAfter * 1000, 30000)
        : Math.min(1500 * Math.pow(2, attempt), 30000) + Math.floor(Math.random() * 500)

      logJson({ event: "openai_retry", label, attempt: attempt + 1, waitMs, error: message.slice(0, 1000) }, "warn")
      await sleep(waitMs)
    }
  }

  throw new Error(`${label} failed after retries: ${safeErrorMessage(lastError)}`)
}

function buildResearchPayloadFromOpenAI(item: WorkItem, parsed: JsonRecord, response: any) {
  const request = asRecord(item.request)
  return cleanPlainObject({
    query: firstString(request.researchQuery, asRecord(request.row).finalTitle, asRecord(request.row).csvName),
    provider: "openai_web_search",
    webSearchTool: firstString(item.webSearchTool, "web_search_preview"),
    model: item.model,
    modelSecretName: item.modelSecretName,
    researchSummary: firstString(parsed.researchSummary),
    customerQuestions: Array.isArray(parsed.customerQuestions) ? parsed.customerQuestions.map(String) : [],
    competitorPatterns: Array.isArray(parsed.competitorPatterns) ? parsed.competitorPatterns.map(String) : [],
    citations: Array.isArray(parsed.citations) ? parsed.citations : [],
    toolCalls: extractToolCalls(response),
  }) as JsonRecord
}

async function runOpenAIPhase(item: WorkItem): Promise<PhaseResult> {
  const startedAt = Date.now()
  const label = getOpenAILabel(item)
  const payload = buildOpenAIRequestPayload(item)
  validateOpenAIPhaseConfig(item, payload)

  const response = await createOpenAIResponse(payload, label)
  const outputText = getOutputText(response)
  const parsed = asRecord(parseJsonFromText(outputText, label))

  if (item.phase === "body_generation" && !firstString(parsed.bodyHtml, parsed.body_html)) {
    throw new Error("Body generation returned empty bodyHtml.")
  }

  if (item.phase === "title_and_field_generation") {
    for (const required of ["title", "slug", "postSummary", "metaTitle", "metaDescription", "excerpt"]) {
      if (!firstString(parsed[required])) throw new Error(`Field generation returned missing ${required}.`)
    }
  }

  await callEdge("complete_phase", {
    jobId: item.jobId,
    rowId: item.rowId,
    stepId: item.stepId,
    phase: item.phase,
    requestPayload: {
      ...payload,
      apiKey: undefined,
      workerId: config.workerId,
      modelSecretName: item.modelSecretName,
      reasoningEffort: firstString(asRecord(payload.reasoning).effort),
    },
    responsePayload: makeSafeOpenAIResponsePayload(response),
    parsedOutput: parsed,
    researchPayload: buildResearchPayloadFromOpenAI(item, parsed, response),
    toolCalls: extractToolCalls(response),
    usagePayload: response?.usage || {},
    openaiResponseId: response?.id || "",
    includeRows: false,
  })

  return {
    success: true,
    jobId: item.jobId,
    rowId: item.rowId,
    stepId: item.stepId,
    phase: item.phase,
    durationMs: Date.now() - startedAt,
    openaiResponseId: response?.id || "",
  }
}

function getWebflowItemId(data: any) {
  return firstString(
    data?.data?.id,
    data?.data?._id,
    data?.id,
    data?._id,
    data?.result?.data?.id,
    data?.result?.data?._id,
    data?.result?.id,
    data?.result?._id,
    data?.items?.[0]?.id,
    data?.items?.[0]?._id,
    data?.data?.items?.[0]?.id,
    data?.data?.items?.[0]?._id
  )
}

function getCollectionFields(collectionData: unknown) {
  const data = asRecord(collectionData)
  const candidates = [data.fields, asRecord(data.collection).fields, asRecord(data.data).fields]
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) return candidate
  }
  return []
}

function normalizeLabel(value: unknown) {
  return stringifyValue(value)
    .trim()
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

function buildCollectionFieldIndex(collectionData: unknown) {
  const fields = getCollectionFields(collectionData)
  const validSlugs = new Set<string>(["name", "slug"])
  const labelToSlug = new Map<string, string>([
    ["name", "name"],
    ["slug", "slug"],
  ])

  for (const rawField of fields) {
    const field = asRecord(rawField)
    const slug = firstString(field.slug, field.apiName, field.fieldSlug)
    if (!slug) continue
    validSlugs.add(slug)

    for (const label of [field.displayName, field.name, field.label, slug, slug.replace(/-/g, " ")]) {
      const normalized = normalizeLabel(label)
      if (normalized && !labelToSlug.has(normalized)) labelToSlug.set(normalized, slug)
    }
  }

  return { validSlugs, labelToSlug }
}

function remapOrFilterFieldDataForCollection(fieldData: JsonRecord, collectionData: unknown) {
  const { validSlugs, labelToSlug } = buildCollectionFieldIndex(collectionData)
  const next: JsonRecord = {}
  const skippedFields: string[] = []
  const remappedFields: Array<{ from: string; to: string }> = []

  for (const [key, value] of Object.entries(fieldData)) {
    if (!key.trim()) continue
    if (validSlugs.has(key)) {
      next[key] = value
      continue
    }

    const inferredSlug = labelToSlug.get(normalizeLabel(key))
    if (inferredSlug) {
      next[inferredSlug] = value
      remappedFields.push({ from: key, to: inferredSlug })
      continue
    }

    skippedFields.push(key)
  }

  return {
    fieldData: next,
    skippedFields,
    remappedFields,
    validFieldSlugs: Array.from(validSlugs).sort(),
  }
}

function extractCollectionIdFromPath(path: string) {
  const match = String(path || "").match(/\/collections\/([^/]+)\/items/)
  return match?.[1] || ""
}

async function webflowRequest(options: {
  method: string
  path: string
  query?: unknown
  body?: unknown
  label?: string
}) {
  if (!config.webflowToken) throw new Error("Missing WEBFLOW_API_TOKEN or WEBFLOW_OAUTH_TOKEN for Webflow publish phase.")

  const url = normalizeUrl(config.webflowApiBase, options.path, options.query)
  const headers: Record<string, string> = {
    Authorization: `Bearer ${config.webflowToken}`,
    Accept: "application/json",
  }

  const init: RequestInit = {
    method: options.method,
    headers,
  }

  if (options.body !== undefined && options.body !== null) {
    headers["Content-Type"] = "application/json"
    init.body = JSON.stringify(cleanPlainObject(options.body))
  }

  return await callJson(url, init, {
    label: options.label || `Webflow ${options.method} ${options.path}`,
    timeoutMs: config.webflowRequestTimeoutMs,
    retries: config.webflowMaxRetries,
  })
}

async function prepareWebflowBody(request: JsonRecord) {
  const rawBody = asRecord(request.body)
  const rawFieldData = asRecord(rawBody.fieldData)

  if (!config.validateWebflowFields) {
    return {
      body: rawBody,
      fieldData: rawFieldData,
      skippedFields: [] as string[],
      remappedFields: [] as Array<{ from: string; to: string }>,
    }
  }

  const collectionId = extractCollectionIdFromPath(firstString(request.path))
  if (!collectionId) throw new Error("Could not infer Webflow collection ID from publish request path.")

  const collectionData = await webflowRequest({
    method: "GET",
    path: `/collections/${collectionId}`,
    label: "Webflow GET collection fields",
  })

  const prepared = remapOrFilterFieldDataForCollection(rawFieldData, collectionData)
  if (!firstString(prepared.fieldData.name) || !firstString(prepared.fieldData.slug)) {
    throw new Error("Prepared Webflow fieldData is missing required name or slug after collection validation.")
  }

  return {
    body: {
      ...rawBody,
      fieldData: prepared.fieldData,
    },
    fieldData: prepared.fieldData,
    skippedFields: prepared.skippedFields,
    remappedFields: prepared.remappedFields,
  }
}

async function runWebflowPhase(item: WorkItem): Promise<PhaseResult> {
  const startedAt = Date.now()
  const request = asRecord(item.request)

  if (request.execute === false) {
    const fieldData = asRecord(request.fieldData, asRecord(asRecord(request.body).fieldData))
    await callEdge("complete_phase", {
      jobId: item.jobId,
      rowId: item.rowId,
      stepId: item.stepId,
      phase: item.phase,
      requestPayload: request,
      parsedOutput: { dryRun: true, message: "Dry run enabled. Webflow item was not created." },
      finalFieldData: fieldData,
      webflowResponse: { dryRun: true },
      includeRows: false,
    })

    return {
      success: true,
      jobId: item.jobId,
      rowId: item.rowId,
      stepId: item.stepId,
      phase: item.phase,
      durationMs: Date.now() - startedAt,
      webflowItemId: "",
    }
  }

  if (request.publishMode === "live" && request.allowPublish === false) {
    throw new Error("Publishing is disabled for this job, but publishMode is live.")
  }

  const prepared = await prepareWebflowBody(request)
  const method = firstString(request.method, "POST")
  const path = firstString(request.path)
  if (!path) throw new Error("Webflow publish request is missing path.")

  const response = await webflowRequest({
    method,
    path,
    query: request.query,
    body: prepared.body,
    label: `Webflow ${method} ${path}`,
  })

  const webflowItemId = getWebflowItemId(response)

  await callEdge("complete_phase", {
    jobId: item.jobId,
    rowId: item.rowId,
    stepId: item.stepId,
    phase: item.phase,
    requestPayload: {
      ...request,
      preparedBody: prepared.body,
      skippedFields: prepared.skippedFields,
      remappedFields: prepared.remappedFields,
    },
    parsedOutput: {
      webflowItemId,
      skippedFields: prepared.skippedFields,
      remappedFields: prepared.remappedFields,
    },
    finalFieldData: prepared.fieldData,
    webflowItemId,
    webflowResponse: response as JsonRecord,
    includeRows: false,
  })

  return {
    success: true,
    jobId: item.jobId,
    rowId: item.rowId,
    stepId: item.stepId,
    phase: item.phase,
    durationMs: Date.now() - startedAt,
    webflowItemId,
  }
}

async function failPhase(item: WorkItem, error: unknown) {
  await callEdge("fail_phase", {
    jobId: item.jobId,
    rowId: item.rowId,
    stepId: item.stepId,
    phase: item.phase,
    error: safeErrorMessage(error),
    includeRows: false,
  })
}

async function processWorkItem(item: WorkItem): Promise<PhaseResult> {
  const startedAt = Date.now()
  let heartbeatTimer: NodeJS.Timeout | null = null

  logJson({
    event: "phase_started",
    workerId: config.workerId,
    jobId: item.jobId,
    rowId: item.rowId,
    stepId: item.stepId,
    phase: item.phase,
    model: item.model,
    modelSecretName: item.modelSecretName,
    reasoningEffort: firstString(item.reasoningEffort, expectedReasoningEffortForPhase(item.phase)),
  })

  try {
    heartbeatTimer = setInterval(() => {
      void sendHeartbeat(item)
    }, config.heartbeatMs)

    const result = item.phase === "webflow_publish"
      ? await runWebflowPhase(item)
      : await runOpenAIPhase(item)

    logJson({
      event: "phase_completed",
      workerId: config.workerId,
      jobId: item.jobId,
      rowId: item.rowId,
      stepId: item.stepId,
      phase: item.phase,
      durationMs: result.durationMs,
      openaiResponseId: result.openaiResponseId || null,
      webflowItemId: result.webflowItemId || null,
    })

    return result
  } catch (error) {
    const message = safeErrorMessage(error)
    try {
      await failPhase(item, error)
    } catch (failError) {
      logJson({
        event: "fail_phase_report_failed",
        workerId: config.workerId,
        jobId: item.jobId,
        rowId: item.rowId,
        stepId: item.stepId,
        phase: item.phase,
        originalError: message,
        reportError: safeErrorMessage(failError),
      }, "error")
    }

    logJson({
      event: "phase_failed",
      workerId: config.workerId,
      jobId: item.jobId,
      rowId: item.rowId,
      stepId: item.stepId,
      phase: item.phase,
      durationMs: Date.now() - startedAt,
      error: message,
    }, "error")

    return {
      success: false,
      jobId: item.jobId,
      rowId: item.rowId,
      stepId: item.stepId,
      phase: item.phase,
      durationMs: Date.now() - startedAt,
      error: message,
    }
  } finally {
    if (heartbeatTimer) clearInterval(heartbeatTimer)
  }
}

async function processWithConcurrency<T, R>(items: T[], concurrency: number, worker: (item: T) => Promise<R>) {
  const results: R[] = []
  let nextIndex = 0

  async function run() {
    while (nextIndex < items.length) {
      const currentIndex = nextIndex++
      results[currentIndex] = await worker(items[currentIndex])
    }
  }

  await Promise.all(Array.from({ length: Math.min(Math.max(1, concurrency), items.length) }, () => run()))
  return results
}

function getJobStatus(data: EdgeClaimResponse) {
  return firstString(asRecord(data.job).status, data.status)
}

async function runLoopOnce() {
  const claimData = (await callEdge("claim_work", {
    workerBatchSize: config.workerBatchSize,
    workerConcurrency: config.workerConcurrency,
    resetStaleMinutes: config.staleLockMinutes,
    includeRows: false,
  })) as EdgeClaimResponse

  const workItems = Array.isArray(claimData.workItems) ? claimData.workItems : []
  const claimedTotal = Number(claimData.claimedTotal || workItems.length || 0)

  logJson({
    event: "claim_work_complete",
    workerId: config.workerId,
    claimedTotal,
    returnedWorkItems: workItems.length,
    jobStatus: getJobStatus(claimData),
    message: claimData.message || null,
  })

  if (!workItems.length) {
    return {
      worked: false,
      done: ["completed", "completed_with_errors", "cancelled", "error"].includes(getJobStatus(claimData)),
      results: [] as PhaseResult[],
    }
  }

  const results = await processWithConcurrency(
    workItems,
    Math.min(config.workerConcurrency, workItems.length),
    processWorkItem
  )

  const successCount = results.filter((result) => result.success).length
  const errorCount = results.length - successCount

  logJson({
    event: "batch_processed",
    workerId: config.workerId,
    processedCount: results.length,
    successCount,
    errorCount,
    phases: results.reduce((acc: Record<string, number>, result) => {
      acc[result.phase] = (acc[result.phase] || 0) + 1
      return acc
    }, {}),
  })

  return {
    worked: results.length > 0,
    done: false,
    results,
  }
}

function validateRuntimeConfig() {
  if (!config.edgeFunctionUrl.startsWith("http")) {
    throw new Error("EDGE_FUNCTION_URL must be a full Supabase Edge Function URL.")
  }

  if (config.workerBatchSize > MAX_WORKER_CONCURRENCY || config.workerConcurrency > MAX_WORKER_CONCURRENCY) {
    throw new Error(`Worker batch size and concurrency may not exceed ${MAX_WORKER_CONCURRENCY}.`)
  }

  if (!config.openaiApiKey) throw new Error("Missing OPENAI_API_KEY.")

  logJson({
    event: "worker_config_validated",
    workerId: config.workerId,
    edgeFunctionUrl: config.edgeFunctionUrl,
    workerBatchSize: config.workerBatchSize,
    workerConcurrency: config.workerConcurrency,
    staleLockMinutes: config.staleLockMinutes,
    pollIntervalMs: config.pollIntervalMs,
    idlePollIntervalMs: config.idlePollIntervalMs,
    heartbeatMs: config.heartbeatMs,
    expectedBodyModel: EXPECTED_BODY_MODEL,
    expectedFieldModel: EXPECTED_FIELD_MODEL,
    expectedBodyReasoningEffort: EXPECTED_BODY_REASONING_EFFORT,
    expectedFieldReasoningEffort: EXPECTED_FIELD_REASONING_EFFORT,
    webflowTokenPresent: Boolean(config.webflowToken),
    validateWebflowFields: config.validateWebflowFields,
    storeFullOpenAIResponse: config.storeFullOpenAIResponse,
  })
}

async function main() {
  validateRuntimeConfig()

  process.on("SIGTERM", () => {
    shutdownState.shuttingDown = true
    logJson({ event: "shutdown_signal", signal: "SIGTERM" }, "warn")
  })

  process.on("SIGINT", () => {
    shutdownState.shuttingDown = true
    logJson({ event: "shutdown_signal", signal: "SIGINT" }, "warn")
  })

  logJson({
    event: "worker_started",
    workerId: config.workerId,
    mode: "edge_claim_complete_external_openai_webflow_worker",
    maxParallelPhases: config.workerConcurrency,
  })

  while (!shutdownState.shuttingDown) {
    if (!config.workerEnabled) {
      logJson({ event: "worker_disabled_sleeping", workerId: config.workerId }, "warn")
      await sleep(config.idlePollIntervalMs)
      continue
    }

    try {
      const result = await runLoopOnce()

      if (result.done) {
        logJson({ event: "queue_terminal_or_no_active_work", workerId: config.workerId })
        if (!config.keepAliveWhenDone) break
        await sleep(config.idlePollIntervalMs)
        continue
      }

      await sleep(result.worked ? config.pollIntervalMs : config.idlePollIntervalMs)
    } catch (error) {
      logJson({ event: "worker_loop_error", workerId: config.workerId, error: safeErrorMessage(error) }, "error")
      if (config.stopOnFatalError) throw error
      await sleep(config.errorSleepMs)
    }
  }

  logJson({ event: "worker_stopped", workerId: config.workerId })
}

main().catch((error) => {
  logJson({ event: "fatal_worker_error", error: safeErrorMessage(error) }, "error")
  process.exit(1)
})
