import { createClient } from "@supabase/supabase-js"
import OpenAI from "openai"

const WEBFLOW_API_BASE = "https://api.webflow.com/v2"

const BLOG_CREATION_TABLE = "webflow_blog_creation"
const BLOG_ROWS_TABLE = "webflow_blog_creation_rows"
const BLOG_STEPS_TABLE = "webflow_blog_creation_steps"
const BLOG_RESEARCH_TABLE = "webflow_blog_creation_research"

const WEBFLOW_BLOG_COLLECTION_ID = "69fab35479ce3c7d679fd50b"

const BEST_MODEL_SECRET_NAME = "best_chatgpt_modal"
const EXPECTED_BEST_MODEL = "gpt-5.5-pro"
const REASONING_EFFORT = "xhigh"
const DEFAULT_PUBLISH_MODE = "live"
const DEFAULT_OPENAI_WEB_SEARCH_TOOL = "web_search_preview"

type PublishMode = "live" | "staged"

type StepKind =
  | "title_and_field_generation"
  | "body_generation"
  | "webflow_publish"

type PhaseResult = {
  success: boolean
  rowId?: string
  jobId?: string
  phase?: StepKind
  message?: string
  error?: string
}

type SerpResult = {
  position: number
  title: string
  url: string
  domain: string
  snippet: string
  source: string
}

const DEFAULT_ARTICLE_AUTHOR = "By Bang AutoGlass Editorial Team"

const DEFAULT_FIELD_MAPPING = {
  name: "name",
  slug: "slug",
  articleAuthor: "article-author",
  publicationDate: "publication-date",
  category: "category",
  body: "post-body",
  postSummary: "post-summary",
  excerpt: "post-summary",
  metaTitle: "meta_title",
  metaDescription: "meta_description",
  faqHtml: "faqHtml",
  ctaText: "ctaText",
  imagePrompt: "imagePrompt",
  tags: "tags",
  notes: "notes",
  make: "make",
  model: "model",
  serviceLabel: "service-label",
  city: "city",
  state: "state",
}

const DEFAULT_COMPETITOR_DOMAINS = [
  "safelite.com",
  "glassdoctor.com",
  "autoglassnow.com",
  "windshieldhub.com",
  "gerbercollision.com",
  "caliber.com",
]

const DEFAULT_BODY_INSTRUCTION =
  `Hey so im currently working on my service cms template for my business Bang AutoGlass and our page only really consists of a format text block that makes up the content for the service. There is a hero but it only has our contact information and a quick sign up form and then below the hero section is our actual details for our service in one long block. What i need your help doing is researching online about the service I'm going to share with you that will include a specific car make. model and service label and once you do i want you to give me a full seo blast format text consisting of H2s, H3s, H4 optional and paragraph text. You can include bullet points and lists as needed but do not add more than one list of each if you do. please try to aim for 2000-2500 words and cover all possible subjects of the service. Please do this now and still write the content in normal conversation however again just format the text but dont give it to me in code form. Also you absolutely dont need to use all of this information however i do want to share with you some general business information about our business so that way when you go to write content, if anything relates or any sections mentions details about our business please try to make sure the content not only is seo enriched but aligns with our business practices. here is the reference "we are a mobile service and most glass replacements take 30-45 minutes to be completed and then 1 hour for the glue to dry. We offer next-day appointments and with every replacement offer a lifetime workmanship warranty and OEM-quality materials." please make sure to write thorough content for each section and category relevant to our service we are working on. i want the actual content written in format text and not normal text. Also i dont want the actual h2 or h3 headings to be written i want the heading to be formatted that way but also dont give me code. Also just so you know when it comes to helping a customer file a claim if they haven't already, we dont file the claim on behalf of the customer we help assist them in making the claim. Please make sure to include as many make specific details and try to include as many primary and secondary keywords targeting the car make / service variation. Here is the service we will be working on: [Name]`

const shutdownState = {
  shuttingDown: false,
}

function getRequiredEnv(name: string) {
  const value = process.env[name]

  if (!value || !value.trim()) {
    throw new Error(`Missing required environment variable: ${name}`)
  }

  return value.trim()
}

function getOptionalEnv(name: string) {
  const value = process.env[name]
  return value && value.trim() ? value.trim() : null
}

function getBestModelFromSecret() {
  const model = getRequiredEnv(BEST_MODEL_SECRET_NAME)

  if (model !== EXPECTED_BEST_MODEL) {
    throw new Error(
      `Invalid ${BEST_MODEL_SECRET_NAME}. Expected "${EXPECTED_BEST_MODEL}" but received "${model}".`
    )
  }

  return model
}

function getWorkerEnabled() {
  return (getOptionalEnv("WORKER_ENABLED") || "true").toLowerCase() === "true"
}

function getWorkerConcurrency() {
  return clampInteger(getOptionalEnv("WORKER_CONCURRENCY"), 1, 5, 1)
}

function getPollIntervalMs() {
  return clampInteger(getOptionalEnv("WORKER_POLL_INTERVAL_MS"), 1000, 60000, 5000)
}

function getStaleLockMinutes() {
  return clampInteger(getOptionalEnv("WORKER_STALE_LOCK_MINUTES"), 5, 120, 15)
}

function getOpenAIWebSearchToolType() {
  return getOptionalEnv("OPENAI_WEB_SEARCH_TOOL_TYPE") || DEFAULT_OPENAI_WEB_SEARCH_TOOL
}

function getOpenAIWebSearchTool() {
  return { type: getOpenAIWebSearchToolType() } as any
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

function firstString(...values: unknown[]) {
  for (const value of values) {
    const text = stringifyValue(value).trim()
    if (text) return text
  }

  return ""
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

function asRecord(value: unknown, fallback: Record<string, unknown> = {}) {
  return isRecord(value) ? value : fallback
}

function cleanPlainObject(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(cleanPlainObject)
  if (!isRecord(value)) return value

  const output: Record<string, unknown> = {}

  for (const [key, innerValue] of Object.entries(value)) {
    if (["__proto__", "constructor", "prototype"].includes(key)) continue
    output[key] = cleanPlainObject(innerValue)
  }

  return output
}

function compactObject(value: Record<string, unknown>) {
  const next: Record<string, unknown> = {}

  for (const [key, fieldValue] of Object.entries(value)) {
    if (fieldValue === undefined || fieldValue === null) continue
    if (typeof fieldValue === "string" && fieldValue.trim() === "") continue
    if (Array.isArray(fieldValue) && fieldValue.length === 0) continue
    next[key] = fieldValue
  }

  return next
}

function clampInteger(value: unknown, min: number, max: number, fallback: number) {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) return fallback

  return Math.max(min, Math.min(max, Math.floor(parsed)))
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function safeErrorMessage(error: unknown) {
  if (!error) return "Unknown server error."

  if (error instanceof Error) {
    return error.message.length > 4000
      ? `${error.message.slice(0, 4000)}... [truncated]`
      : error.message
  }

  if (typeof error === "string") {
    return error.length > 4000
      ? `${error.slice(0, 4000)}... [truncated]`
      : error
  }

  try {
    const text = JSON.stringify(error, null, 2)
    return text.length > 4000 ? `${text.slice(0, 4000)}... [truncated]` : text
  } catch {
    return String(error)
  }
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-")
    .slice(0, 120)
}

function normalizeDomain(value: unknown) {
  const raw = stringifyValue(value).trim().toLowerCase()
  if (!raw) return ""

  try {
    const url = raw.includes("://") ? new URL(raw) : new URL(`https://${raw}`)
    return url.hostname.replace(/^www\./, "")
  } catch {
    return raw.replace(/^https?:\/\//, "").replace(/^www\./, "").split("/")[0].trim()
  }
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

function buildQuery(params: Record<string, unknown>) {
  const search = new URLSearchParams()

  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue
    search.set(key, String(value))
  }

  const query = search.toString()
  return query ? `?${query}` : ""
}

function buildPath(path: string, query: Record<string, unknown>) {
  return `${path}${buildQuery(query)}`
}

function cleanBlogTitle(value: string) {
  return String(value || "")
    .replace(/^\s*[-*#`]+\s*/g, "")
    .replace(/^\s*(recommended\s+)?(seo\s+)?(blog\s+)?title\s*\d*\s*[:\-–—]\s*/i, "")
    .replace(/^\s*recommended\s+title\s*[:\-–—]\s*/i, "")
    .replace(/^\s*final\s+title\s*[:\-–—]\s*/i, "")
    .replace(/^\s*name\s*[:\-–—]\s*/i, "")
    .replace(/^\s*["'“”‘’]+|["'“”‘’]+\s*$/g, "")
    .replace(/\s+/g, " ")
    .trim()
}

function replaceVariables(template: string, row: Record<string, unknown>) {
  const rowData = asRecord(row.raw_row_data)

  const values: Record<string, string> = {
    Name: firstString(row.webflow_name, row.selected_title, row.name_seed, rowData.Name),
    Make: firstString(row.make, rowData.Make),
    Model: firstString(row.model, rowData.Model),
    "Service Label": firstString(row.service_label, rowData["Service Label"]),
    City: firstString(row.city, rowData.City),
    State: firstString(row.state, rowData.State),
    primaryKeyword: firstString(row.primary_keyword, rowData.primaryKeyword),
    secondaryKeywords: Array.isArray(row.secondary_keywords)
      ? row.secondary_keywords.join(", ")
      : stringifyValue(row.secondary_keywords),
  }

  return template.replace(/\[([^\]]+)\]/g, (_match, rawKey) => {
    const key = String(rawKey || "").trim()
    if (Object.prototype.hasOwnProperty.call(values, key)) return values[key]

    const lowerKey = key.toLowerCase()
    const matchingKey = Object.keys(values).find((candidate) => candidate.toLowerCase() === lowerKey)
    return matchingKey ? values[matchingKey] : ""
  })
}

function getOpenAIErrorStatus(error: unknown) {
  const candidate = error as any
  return Number(candidate?.status || candidate?.code || candidate?.response?.status || 0)
}

async function withOpenAIRetry<T>(
  operation: () => Promise<T>,
  label: string,
  maxAttempts = 3
) {
  let lastError: unknown = null

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await operation()
    } catch (error) {
      lastError = error

      const status = getOpenAIErrorStatus(error)
      const message = safeErrorMessage(error)

      const retryable =
        status === 429 ||
        status === 408 ||
        status >= 500 ||
        /rate.?limit|timeout|temporarily|overloaded/i.test(message)

      if (!retryable || attempt >= maxAttempts - 1) break

      const retryAfter = Number(
        (error as any)?.headers?.["retry-after"] ||
          (error as any)?.headers?.get?.("retry-after") ||
          0
      )

      const retryMs =
        Number.isFinite(retryAfter) && retryAfter > 0
          ? Math.min(retryAfter * 1000, 15000)
          : Math.min(1500 * Math.pow(2, attempt), 15000) +
            Math.floor(Math.random() * 500)

      console.warn(`${label} retrying in ${retryMs}ms after: ${message}`)
      await sleep(retryMs)
    }
  }

  throw new Error(`${label} failed after retries: ${safeErrorMessage(lastError)}`)
}

function getTextFromResponse(response: any) {
  if (typeof response.output_text === "string" && response.output_text.trim()) {
    return response.output_text.trim()
  }

  const parts: string[] = []

  for (const item of response.output || []) {
    if (item.type === "message" && Array.isArray(item.content)) {
      for (const content of item.content) {
        if (content.type === "output_text" && typeof content.text === "string") {
          parts.push(content.text)
        }
      }
    }
  }

  return parts.join("\n").trim()
}

function parseJsonFromOpenAIResponse(response: any, label: string) {
  const text = getTextFromResponse(response)

  if (!text) throw new Error(`${label} returned an empty response.`)

  try {
    return JSON.parse(text)
  } catch {
    const firstBrace = text.indexOf("{")
    const lastBrace = text.lastIndexOf("}")

    if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
      throw new Error(`${label} response was not valid JSON.`)
    }

    return JSON.parse(text.slice(firstBrace, lastBrace + 1))
  }
}

function titleGenerationSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      title: { type: "string" },
      slug: { type: "string" },
      notes: {
        type: "object",
        additionalProperties: false,
        properties: {
          csvNameIsFinalTitle: { type: "boolean" },
          titleSource: { type: "string" },
          csvName: { type: "string" },
          metadataOnly: { type: "boolean" },
        },
        required: ["csvNameIsFinalTitle", "titleSource", "csvName", "metadataOnly"],
      },
      postSummary: { type: "string" },
      metaTitle: { type: "string" },
      metaDescription: { type: "string" },
      excerpt: { type: "string" },
      bodyHtml: { type: "string" },
      faqHtml: { type: "string" },
      ctaText: { type: "string" },
      imagePrompt: { type: "string" },
      tags: {
        type: "array",
        items: { type: "string" },
      },
      researchSummary: { type: "string" },
      customerQuestions: {
        type: "array",
        items: { type: "string" },
      },
      competitorPatterns: {
        type: "array",
        items: { type: "string" },
      },
    },
    required: [
      "title",
      "slug",
      "notes",
      "postSummary",
      "metaTitle",
      "metaDescription",
      "excerpt",
      "bodyHtml",
      "faqHtml",
      "ctaText",
      "imagePrompt",
      "tags",
      "researchSummary",
      "customerQuestions",
      "competitorPatterns",
    ],
  }
}

function bodyGenerationSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      bodyHtml: { type: "string" },
      researchSummary: { type: "string" },
      customerQuestions: {
        type: "array",
        items: { type: "string" },
      },
      citations: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            title: { type: "string" },
            url: { type: "string" },
            note: { type: "string" },
          },
          required: ["title", "url", "note"],
        },
      },
    },
    required: ["bodyHtml", "researchSummary", "customerQuestions", "citations"],
  }
}

function buildMetadataInstructions(job: Record<string, unknown>) {
  return `
You are the Bang AutoGlass SEO metadata strategist.

This request is responsible for metadata only.

Critical title rule:
- The CSV Name field is the final Webflow blog title.
- Do not rewrite, improve, shorten, expand, rephrase, or replace the CSV Name.
- Do not generate alternate blog titles.
- The returned title field must exactly match the provided CSV Name value.

This request must NOT write the final blog body.

Use any provided SERP context only for metadata and research notes. If make, model, service label, city, or state are blank, do not invent them.

Generate:
- slug based on CSV Name
- postSummary
- metaTitle
- metaDescription
- excerpt equal to postSummary
- concise researchSummary
- customerQuestions
- competitorPatterns

Field-specific instructions:
Post Summary:
${firstString(job.post_summary_instruction)}

Meta Title:
${firstString(job.meta_title_instruction)}

Meta Description:
${firstString(job.meta_description_instruction)}

Output instructions:
- title must exactly equal the CSV Name value.
- bodyHtml must be an empty string.
- faqHtml should be empty unless truly useful.
- ctaText should be empty unless useful.
- imagePrompt should be empty unless useful.
- tags should be an empty array unless useful.
- notes must be:
{
  "csvNameIsFinalTitle": true,
  "titleSource": "csv_name",
  "csvName": "Exact CSV Name value",
  "metadataOnly": true
}

Return valid JSON only.
`.trim()
}

function buildBodyInstructions(job: Record<string, unknown>, row: Record<string, unknown>) {
  const bodyInstruction = firstString(job.body_instruction, DEFAULT_BODY_INSTRUCTION)
  const resolvedBodyInstruction = replaceVariables(bodyInstruction, row)

  return `
You are the Bang AutoGlass SEO blog content writer.

This request is ONLY for writing the final blog body.

Use the selected Webflow Name field as the final blog title. The Webflow Name comes directly from the CSV Name and must not be rewritten.

The final body must be returned as clean Webflow-ready HTML in bodyHtml because Webflow rich text fields need HTML. The rendered Webflow page will show formatted text, not code. Do not wrap the HTML in markdown code fences.

User's editable body instructions:
${resolvedBodyInstruction}

Additional strict rules:
- Do not include an H1. The Webflow Name field is the H1/title.
- Use clean HTML such as <p>, <h2>, <h3>, <h4>, <ul>, <ol>, <li>, and <strong>.
- Do not literally write "H2", "H3", or "H4" before headings.
- Make the content specific to the blog Name/title and available row context.
- Make, Model, Service Label, City, and State are optional. If any of those fields are blank, do not invent them.
- Naturally discuss helpful commercial-intent topics when relevant: repair vs replacement, cost factors without exact prices, safety, timing, mobile service, insurance claim support, OEM-quality glass, ADAS recalibration concerns, warning signs, and what to expect.
- Bang AutoGlass can assist customers with making an insurance claim, but do not say Bang AutoGlass files the claim on behalf of the customer.
- Bang AutoGlass is a mobile service.
- Most glass replacements take about 30 to 45 minutes to complete, followed by about 1 hour for adhesive curing, but do not guarantee this as an exact timeline for every job.
- Bang AutoGlass offers next-day appointments when available.
- Bang AutoGlass offers a lifetime workmanship warranty with replacements.
- Say OEM-quality materials when relevant.
- Do not say aftermarket.
- Do not invent exact prices, laws, insurance affiliations, certifications, or guaranteed timelines.
- Do not overstuff keywords.
- Do not include fake reviews, fake statistics, fake citations, or unverifiable claims.
- Do not mention that you used AI.

Output requirements:
- Return valid JSON only.
- bodyHtml must contain the full blog article HTML.
- Include a concise researchSummary.
- Include customerQuestions as an array.
- Include citations as an array of title/url/note objects when sources are available from web research.
`.trim()
}

async function webflowRequest(options: {
  method: string
  path: string
  body?: unknown
  query?: Record<string, unknown>
  retry?: boolean
}) {
  const { method, path, body, query = {}, retry = true } = options

  const endpoint = `${WEBFLOW_API_BASE}${buildPath(path, query)}`
  const headers: Record<string, string> = {
    Authorization: `Bearer ${getRequiredEnv("WEBFLOW_API_TOKEN")}`,
    Accept: "application/json",
  }

  const init: RequestInit = { method, headers }

  if (body !== undefined && body !== null) {
    headers["Content-Type"] = "application/json"
    init.body = JSON.stringify(cleanPlainObject(body))
  }

  let lastError: Error | null = null

  for (let attempt = 0; attempt < (retry ? 3 : 1); attempt++) {
    try {
      const response = await fetch(endpoint, init)
      const text = await response.text()

      let json: any = null
      try {
        json = text ? JSON.parse(text) : null
      } catch {
        json = text ? { raw: text.slice(0, 2000) } : null
      }

      if (!response.ok) {
        const shouldRetry =
          retry && attempt < 2 && (response.status === 429 || response.status >= 500)

        if (shouldRetry) {
          const retryAfter = Number(response.headers.get("retry-after") || 0)
          const retryMs = retryAfter
            ? Math.min(retryAfter * 1000, 10000)
            : 1000 * Math.pow(2, attempt)

          await sleep(retryMs)
          continue
        }

        throw new Error(
          `Webflow API error ${response.status} ${method} ${path}: ${JSON.stringify(
            json || text.slice(0, 2000)
          )}`
        )
      }

      return {
        endpoint,
        status: response.status,
        data: json,
      }
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))

      if (attempt < 2 && retry) {
        await sleep(1000 * Math.pow(2, attempt))
        continue
      }

      throw lastError
    }
  }

  throw lastError || new Error("Unknown Webflow request failure.")
}

async function fetchSerpApiResults(query: string, limit = 10) {
  const apiKey = getOptionalEnv("SERPAPI_API_KEY")
  if (!apiKey) return null

  const params = new URLSearchParams({
    engine: getOptionalEnv("SERPAPI_ENGINE") || "google",
    q: query,
    num: String(limit),
    api_key: apiKey,
  })

  const location = getOptionalEnv("SERPAPI_LOCATION")
  if (location) params.set("location", location)

  const response = await fetch(`https://serpapi.com/search.json?${params.toString()}`)
  const data: any = await response.json().catch(() => null)

  if (!response.ok) {
    throw new Error(`SerpAPI error ${response.status}: ${JSON.stringify(data).slice(0, 1500)}`)
  }

  const organic = Array.isArray(data?.organic_results) ? data.organic_results : []

  return organic.slice(0, limit).map((item: any, index: number): SerpResult => {
    const url = firstString(item.link, item.url)

    return {
      position: Number(item.position || index + 1),
      title: cleanBlogTitle(firstString(item.title)),
      url,
      domain: normalizeDomain(firstString(item.domain, item.displayed_link, url)),
      snippet: firstString(item.snippet),
      source: "serpapi",
    }
  })
}

async function fetchBingResults(query: string, limit = 10) {
  const apiKey = getOptionalEnv("BING_SEARCH_API_KEY")
  if (!apiKey) return null

  const endpoint =
    getOptionalEnv("BING_SEARCH_ENDPOINT") || "https://api.bing.microsoft.com/v7.0/search"

  const params = new URLSearchParams({
    q: query,
    count: String(limit),
    responseFilter: "Webpages",
  })

  const response = await fetch(`${endpoint}?${params.toString()}`, {
    headers: {
      "Ocp-Apim-Subscription-Key": apiKey,
    },
  })

  const data: any = await response.json().catch(() => null)

  if (!response.ok) {
    throw new Error(`Bing Search error ${response.status}: ${JSON.stringify(data).slice(0, 1500)}`)
  }

  const webPages = Array.isArray(data?.webPages?.value) ? data.webPages.value : []

  return webPages.slice(0, limit).map((item: any, index: number): SerpResult => {
    const url = firstString(item.url)

    return {
      position: index + 1,
      title: cleanBlogTitle(firstString(item.name)),
      url,
      domain: normalizeDomain(url),
      snippet: firstString(item.snippet),
      source: "bing",
    }
  })
}

async function fetchConfiguredSerpResults(query: string, limit = 10) {
  const preferred = (getOptionalEnv("BLOG_SERP_PROVIDER") || "auto").toLowerCase()
  const providers = preferred === "auto" ? ["serpapi", "bing"] : [preferred]

  for (const provider of providers) {
    if (provider === "serpapi") {
      const results = await fetchSerpApiResults(query, limit)
      if (results) return { provider: "serpapi", results }
    }

    if (provider === "bing") {
      const results = await fetchBingResults(query, limit)
      if (results) return { provider: "bing", results }
    }
  }

  return {
    provider: "none",
    results: [] as SerpResult[],
  }
}

function buildResearchQuery(row: Record<string, unknown>) {
  const make = firstString(row.make)
  const model = firstString(row.model)
  const serviceLabel = firstString(row.service_label)
  const city = firstString(row.city)
  const state = firstString(row.state)
  const title = firstString(row.webflow_name, row.selected_title, row.name_seed)
  const primaryKeyword = firstString(row.primary_keyword, title)
  const vehicle = [make, model].filter(Boolean).join(" ").trim()
  const location = [city, state].filter(Boolean).join(" ").trim()

  return firstString(
    [primaryKeyword, location].filter(Boolean).join(" "),
    [vehicle, serviceLabel, location].filter(Boolean).join(" "),
    title
  )
}

async function insertResearchRecord(options: {
  supabase: any
  jobId: string
  rowId: string
  stepId?: string | null
  phase: "title_and_field_generation" | "body_generation"
  provider: string
  query: string
  organicResults?: SerpResult[]
  researchSummary?: string
  customerQuestions?: string[]
  citations?: unknown[]
  rawResponse?: unknown
}) {
  const {
    supabase,
    jobId,
    rowId,
    stepId = null,
    phase,
    provider,
    query,
    organicResults = [],
    researchSummary = "",
    customerQuestions = [],
    citations = [],
    rawResponse = {},
  } = options

  const competitorResults = organicResults.filter((result) => {
    const domain = normalizeDomain(result.domain || result.url)

    return DEFAULT_COMPETITOR_DOMAINS.some((competitor) => {
      const normalizedCompetitor = normalizeDomain(competitor)
      return domain === normalizedCompetitor || domain.endsWith(`.${normalizedCompetitor}`)
    })
  })

  const bangAutoglassResults = organicResults.filter((result) => {
    const domain = normalizeDomain(result.domain || result.url)
    return domain === "bangautoglass.com" || domain.endsWith(".bangautoglass.com")
  })

  const { error } = await supabase.from(BLOG_RESEARCH_TABLE).insert({
    job_id: jobId,
    row_id: rowId,
    step_id: stepId,
    phase,
    provider,
    query,
    organic_results: cleanPlainObject(organicResults),
    competitor_results: cleanPlainObject(competitorResults),
    bang_autoglass_results: cleanPlainObject(bangAutoglassResults),
    customer_questions: cleanPlainObject(customerQuestions),
    research_summary: researchSummary,
    citations: cleanPlainObject(citations),
    raw_response: cleanPlainObject(rawResponse),
  })

  if (error) {
    console.warn(`Could not insert research record: ${safeErrorMessage(error)}`)
  }
}

async function ensureStep(options: {
  supabase: any
  jobId: string
  rowId: string
  stepNumber: number
  stepKind: StepKind
  model: string
}) {
  const { supabase, jobId, rowId, stepNumber, stepKind, model } = options

  const existing = await supabase
    .from(BLOG_STEPS_TABLE)
    .select("*")
    .eq("row_id", rowId)
    .eq("step_kind", stepKind)
    .maybeSingle()

  if (existing.error) {
    throw new Error(`Could not read step ${stepKind}: ${safeErrorMessage(existing.error)}`)
  }

  if (existing.data) return existing.data as Record<string, unknown>

  const inserted = await supabase
    .from(BLOG_STEPS_TABLE)
    .insert({
      job_id: jobId,
      row_id: rowId,
      step_number: stepNumber,
      step_kind: stepKind,
      status: "pending",
      openai_model: model,
      reasoning_effort: REASONING_EFFORT,
    })
    .select("*")
    .single()

  if (inserted.error || !inserted.data) {
    throw new Error(`Could not create step ${stepKind}: ${safeErrorMessage(inserted.error)}`)
  }

  return inserted.data as Record<string, unknown>
}

async function markStep(options: {
  supabase: any
  stepId: string
  status: "pending" | "running" | "success" | "error" | "skipped"
  requestPayload?: unknown
  responsePayload?: unknown
  parsedOutput?: unknown
  researchPayload?: unknown
  toolCalls?: unknown
  usagePayload?: unknown
  openaiResponseId?: string | null
  error?: string | null
}) {
  const {
    supabase,
    stepId,
    status,
    requestPayload,
    responsePayload,
    parsedOutput,
    researchPayload,
    toolCalls,
    usagePayload,
    openaiResponseId,
    error,
  } = options

  const payload: Record<string, unknown> = {
    status,
    updated_at: new Date().toISOString(),
  }

  if (status === "running") payload.started_at = new Date().toISOString()
  if (["success", "error", "skipped"].includes(status)) {
    payload.completed_at = new Date().toISOString()
  }

  if (requestPayload !== undefined) payload.request_payload = cleanPlainObject(requestPayload)
  if (responsePayload !== undefined) payload.response_payload = cleanPlainObject(responsePayload)
  if (parsedOutput !== undefined) payload.parsed_output = cleanPlainObject(parsedOutput)
  if (researchPayload !== undefined) payload.research_payload = cleanPlainObject(researchPayload)
  if (toolCalls !== undefined) payload.tool_calls = cleanPlainObject(toolCalls)
  if (usagePayload !== undefined) payload.usage_payload = cleanPlainObject(usagePayload)
  if (openaiResponseId !== undefined) payload.openai_response_id = openaiResponseId
  if (error !== undefined) payload.error = error

  const { error: updateError } = await supabase
    .from(BLOG_STEPS_TABLE)
    .update(payload)
    .eq("id", stepId)

  if (updateError) {
    throw new Error(`Could not update step: ${safeErrorMessage(updateError)}`)
  }
}

function extractToolCalls(response: any) {
  const output = Array.isArray(response?.output) ? response.output : []

  return output
    .filter((item: any) => String(item?.type || "").includes("web_search"))
    .map((item: any) => cleanPlainObject(item))
}

async function recoverStaleLocks(supabase: any, jobId: string) {
  try {
    const { error } = await supabase.rpc("webflow_blog_creation_recover_stale_locks", {
      p_job_id: jobId,
      p_stale_after: `${getStaleLockMinutes()} minutes`,
    })

    if (error) {
      console.warn(`stale lock recovery skipped: ${safeErrorMessage(error)}`)
    }
  } catch (error) {
    console.warn(`stale lock recovery failed: ${safeErrorMessage(error)}`)
  }
}

async function refreshJobCounts(supabase: any, jobId: string) {
  const { data: rows, error } = await supabase
    .from(BLOG_ROWS_TABLE)
    .select("id,status")
    .eq("job_id", jobId)

  if (error) {
    throw new Error(`Could not refresh job counts: ${safeErrorMessage(error)}`)
  }

  const values = Array.isArray(rows) ? rows : []

  const counts = {
    total_rows: values.length,
    pending_count: values.filter((row) => ["pending", "queued"].includes(row.status)).length,
    running_count: values.filter((row) => ["running", "publishing"].includes(row.status)).length,
    phase_1_complete_count: values.filter((row) => row.status === "phase_1_complete").length,
    phase_2_complete_count: values.filter((row) => row.status === "phase_2_complete").length,
    success_count: values.filter((row) => row.status === "success").length,
    error_count: values.filter((row) => row.status === "error").length,
    skipped_count: values.filter((row) => row.status === "skipped").length,
  }

  const completed =
    counts.success_count + counts.error_count + counts.skipped_count

  let status: string | undefined

  if (counts.total_rows > 0 && completed >= counts.total_rows) {
    status = counts.error_count > 0 ? "completed_with_errors" : "completed"
  } else if (counts.running_count > 0) {
    status = "running"
  } else if (
    counts.pending_count > 0 ||
    counts.phase_1_complete_count > 0 ||
    counts.phase_2_complete_count > 0
  ) {
    status = "queued"
  }

  const updatePayload: Record<string, unknown> = {
    ...counts,
    updated_at: new Date().toISOString(),
  }

  if (status) {
    updatePayload.status = status
    if (["completed", "completed_with_errors"].includes(status)) {
      updatePayload.completed_at = new Date().toISOString()
    }
  }

  const { data: job, error: updateError } = await supabase
    .from(BLOG_CREATION_TABLE)
    .update(updatePayload)
    .eq("id", jobId)
    .select("*")
    .single()

  if (updateError || !job) {
    throw new Error(`Could not update job counts: ${safeErrorMessage(updateError)}`)
  }

  return job as Record<string, unknown>
}

async function findRunnableJob(supabase: any) {
  const { data, error } = await supabase
    .from(BLOG_CREATION_TABLE)
    .select("*")
    .in("status", ["queued", "running"])
    .order("updated_at", { ascending: true })
    .limit(1)

  if (error) {
    throw new Error(`Could not find runnable job: ${safeErrorMessage(error)}`)
  }

  return Array.isArray(data) && data.length ? data[0] as Record<string, unknown> : null
}

async function claimNextRow(options: {
  supabase: any
  jobId: string
  workerId: string
}) {
  const { supabase, jobId, workerId } = options

  const claimableStatuses = ["pending", "queued", "phase_1_complete", "phase_2_complete"]

  for (let attempt = 0; attempt < 25; attempt++) {
    const { data: candidates, error } = await supabase
      .from(BLOG_ROWS_TABLE)
      .select("*")
      .eq("job_id", jobId)
      .in("status", claimableStatuses)
      .order("row_index", { ascending: true })
      .limit(10)

    if (error) {
      throw new Error(`Could not select next row: ${safeErrorMessage(error)}`)
    }

    const rows = Array.isArray(candidates) ? candidates : []
    if (!rows.length) return null

    for (const candidate of rows) {
      const claim = await supabase
        .from(BLOG_ROWS_TABLE)
        .update({
          status: "running",
          locked_at: new Date().toISOString(),
          locked_by: workerId,
          attempt_count: Number(candidate.attempt_count || 0) + 1,
          started_at: candidate.started_at || new Date().toISOString(),
          last_error: null,
        })
        .eq("id", candidate.id)
        .in("status", claimableStatuses)
        .select("*")

      if (claim.error) {
        throw new Error(`Could not claim row: ${safeErrorMessage(claim.error)}`)
      }

      if (Array.isArray(claim.data) && claim.data.length) {
        return claim.data[0] as Record<string, unknown>
      }
    }

    await sleep(75 + Math.floor(Math.random() * 100))
  }

  return null
}

async function reloadRow(supabase: any, rowId: string) {
  const { data, error } = await supabase
    .from(BLOG_ROWS_TABLE)
    .select("*")
    .eq("id", rowId)
    .single()

  if (error || !data) {
    throw new Error(`Could not reload row ${rowId}: ${safeErrorMessage(error)}`)
  }

  return data as Record<string, unknown>
}

async function generateMetadata(options: {
  openai: OpenAI
  supabase: any
  job: Record<string, unknown>
  row: Record<string, unknown>
  model: string
}) {
  const { openai, supabase, job, row, model } = options

  const jobId = firstString(job.id)
  const rowId = firstString(row.id)
  const rowData = asRecord(row.raw_row_data)

  const finalTitle = firstString(row.name_seed, rowData.Name, rowData.name)
  if (!finalTitle) {
    throw new Error("Cannot generate metadata because CSV Name is missing.")
  }

  const finalSlug = slugify(finalTitle)

  const step = await ensureStep({
    supabase,
    jobId,
    rowId,
    stepNumber: 1,
    stepKind: "title_and_field_generation",
    model,
  })

  const stepId = firstString(step.id)

  try {
    await markStep({
      supabase,
      stepId,
      status: "running",
    })

    const researchQuery = buildResearchQuery({
      ...row,
      webflow_name: finalTitle,
      selected_title: finalTitle,
      name_seed: finalTitle,
    })

    const serp = await fetchConfiguredSerpResults(researchQuery, 10)

    await insertResearchRecord({
      supabase,
      jobId,
      rowId,
      stepId,
      phase: "title_and_field_generation",
      provider: serp.provider,
      query: researchQuery,
      organicResults: serp.results,
      rawResponse: {
        provider: serp.provider,
        results: serp.results,
        csvNameIsFinalTitle: true,
      },
    })

    const rowPayload = {
      finalTitle,
      csvName: finalTitle,
      csvNameIsFinalTitle: true,
      generateBlogTitles: false,
      make: row.make,
      model: row.model,
      serviceLabel: row.service_label,
      city: row.city,
      state: row.state,
      category: row.category,
      articleAuthor: row.article_author,
      publicationDate: row.publication_date,
      nameSeed: finalTitle,
      primaryKeyword: row.primary_keyword || finalTitle,
      secondaryKeywords: row.secondary_keywords || [],
      rawRowData: row.raw_row_data || {},
      fieldInstructions: {
        publicationDateInstruction: job.publication_date_instruction,
        postSummaryInstruction: job.post_summary_instruction,
        metaTitleInstruction: job.meta_title_instruction,
        metaDescriptionInstruction: job.meta_description_instruction,
      },
    }

    const requestPayload = {
      model,
      reasoningEffort: REASONING_EFFORT,
      webSearch: false,
      row: rowPayload,
      externalSerpProvider: serp.provider,
      externalSerpResults: serp.results,
      csvNameIsFinalTitle: true,
      generateBlogTitles: false,
    }

    const response: any = await withOpenAIRetry(
      () =>
        openai.responses.create({
          model,
          instructions: buildMetadataInstructions(job),
          input: [
            "Generate metadata fields for this one Bang AutoGlass blog row. Do not rewrite the CSV Name. The CSV Name is the final Webflow blog title.",
            "",
            `CSV Name / final title: ${finalTitle}`,
            "",
            `Row data: ${JSON.stringify(rowPayload, null, 2)}`,
            "",
            `External SERP results, if available: ${JSON.stringify(serp.results, null, 2)}`,
          ].join("\n"),
          reasoning: {
            effort: REASONING_EFFORT,
          },
          text: {
            verbosity: "high",
            format: {
              type: "json_schema",
              name: "bang_autoglass_blog_metadata_generation_preserve_csv_name",
              strict: true,
              schema: titleGenerationSchema(),
            },
          },
          store: false,
        } as any),
      "OpenAI metadata generation"
    )

    const parsed = parseJsonFromOpenAIResponse(response, "OpenAI metadata generation")
    const returnedTitle = firstString(parsed.title)

    const notes = {
      csvNameIsFinalTitle: true,
      titleSource: "csv_name",
      csvName: finalTitle,
      metadataOnly: true,
    }

    const postSummary = firstString(parsed.postSummary, parsed.excerpt)
    const metaTitle = firstString(parsed.metaTitle)
    const metaDescription = firstString(parsed.metaDescription)

    const phaseOneOutput = {
      ...asRecord(parsed),
      title: finalTitle,
      slug: finalSlug,
      notes,
      csvNameIsFinalTitle: true,
      generateBlogTitles: false,
      ignoredModelReturnedTitle: returnedTitle && returnedTitle !== finalTitle ? returnedTitle : "",
    }

    await markStep({
      supabase,
      stepId,
      status: "success",
      requestPayload,
      responsePayload: {
        id: response.id || null,
        usage: response.usage || {},
        output_text: getTextFromResponse(response),
      },
      parsedOutput: phaseOneOutput,
      researchPayload: {
        query: researchQuery,
        provider: serp.provider,
        results: serp.results,
        researchSummary: firstString(parsed.researchSummary),
        customerQuestions: Array.isArray(parsed.customerQuestions) ? parsed.customerQuestions : [],
        competitorPatterns: Array.isArray(parsed.competitorPatterns) ? parsed.competitorPatterns : [],
        csvNameIsFinalTitle: true,
      },
      toolCalls: extractToolCalls(response),
      usagePayload: response.usage || {},
      openaiResponseId: response.id || null,
    })

    await insertResearchRecord({
      supabase,
      jobId,
      rowId,
      stepId,
      phase: "title_and_field_generation",
      provider: "openai_metadata",
      query: researchQuery,
      organicResults: [],
      researchSummary: firstString(parsed.researchSummary),
      customerQuestions: Array.isArray(parsed.customerQuestions)
        ? parsed.customerQuestions.map(String)
        : [],
      rawResponse: {
        openaiResponseId: response.id || null,
        parsed: phaseOneOutput,
      },
    })

    const { error: updateError } = await supabase
      .from(BLOG_ROWS_TABLE)
      .update({
        status: "phase_1_complete",
        current_phase: "body_generation",
        selected_title: finalTitle,
        title_candidates: [],
        title_notes: cleanPlainObject(notes),
        webflow_name: finalTitle,
        webflow_slug: finalSlug,
        meta_title: metaTitle,
        meta_description: metaDescription,
        post_summary: postSummary,
        excerpt: postSummary,
        body_html: "",
        faq_html: firstString(parsed.faqHtml),
        cta_text: firstString(parsed.ctaText),
        image_prompt: firstString(parsed.imagePrompt),
        tags: Array.isArray(parsed.tags) ? parsed.tags.map(String).filter(Boolean) : [],
        notes: cleanPlainObject(notes),
        phase_1_output: cleanPlainObject(phaseOneOutput),
        title_research: cleanPlainObject({
          query: researchQuery,
          provider: serp.provider,
          externalSerpResults: serp.results,
          researchSummary: firstString(parsed.researchSummary),
          customerQuestions: Array.isArray(parsed.customerQuestions) ? parsed.customerQuestions : [],
          competitorPatterns: Array.isArray(parsed.competitorPatterns) ? parsed.competitorPatterns : [],
          csvNameIsFinalTitle: true,
        }),
        title_openai_response_id: response.id || null,
        title_attempt_count: Number(row.title_attempt_count || 0) + 1,
        locked_at: null,
        locked_by: null,
        last_error: null,
      })
      .eq("id", rowId)

    if (updateError) {
      throw new Error(`Could not save metadata output: ${safeErrorMessage(updateError)}`)
    }
  } catch (error) {
    const message = safeErrorMessage(error)

    await markStep({
      supabase,
      stepId,
      status: "error",
      error: message,
    }).catch((stepError: unknown) => {
      console.warn(`Could not mark metadata step error: ${safeErrorMessage(stepError)}`)
    })

    throw error
  }
}

async function generateBody(options: {
  openai: OpenAI
  supabase: any
  job: Record<string, unknown>
  row: Record<string, unknown>
  model: string
}) {
  const { openai, supabase, job, row, model } = options

  const jobId = firstString(job.id)
  const rowId = firstString(row.id)

  const step = await ensureStep({
    supabase,
    jobId,
    rowId,
    stepNumber: 2,
    stepKind: "body_generation",
    model,
  })

  const stepId = firstString(step.id)

  try {
    await markStep({
      supabase,
      stepId,
      status: "running",
    })

    const selectedTitle = firstString(row.webflow_name, row.selected_title, row.name_seed)
    if (!selectedTitle) throw new Error("Cannot generate body without selected title.")

    const researchQuery = [
      selectedTitle,
      row.make,
      row.model,
      row.service_label,
      row.city,
      row.state,
      row.primary_keyword,
    ]
      .map((value) => stringifyValue(value).trim())
      .filter(Boolean)
      .join(" ")

    const serp = await fetchConfiguredSerpResults(researchQuery, 10)

    await insertResearchRecord({
      supabase,
      jobId,
      rowId,
      stepId,
      phase: "body_generation",
      provider: serp.provider,
      query: researchQuery,
      organicResults: serp.results,
      rawResponse: {
        provider: serp.provider,
        results: serp.results,
      },
    })

    const rowPayload = {
      selectedTitle,
      make: row.make,
      model: row.model,
      serviceLabel: row.service_label,
      city: row.city,
      state: row.state,
      category: row.category,
      articleAuthor: row.article_author,
      publicationDate: row.publication_date,
      nameSeed: row.name_seed,
      primaryKeyword: row.primary_keyword,
      secondaryKeywords: row.secondary_keywords || [],
      postSummary: row.post_summary,
      metaTitle: row.meta_title,
      metaDescription: row.meta_description,
      notes: row.notes || {},
      titleResearch: row.title_research || {},
      rawRowData: row.raw_row_data || {},
      csvNameIsFinalTitle: true,
    }

    const requestPayload = {
      model,
      reasoningEffort: REASONING_EFFORT,
      webSearch: true,
      row: rowPayload,
      bodyInstruction: job.body_instruction,
      externalSerpProvider: serp.provider,
      externalSerpResults: serp.results,
      webSearchTool: getOpenAIWebSearchToolType(),
    }

    const response: any = await withOpenAIRetry(
      () =>
        openai.responses.create({
          model,
          instructions: buildBodyInstructions(job, row),
          input: [
            "Write the final blog body HTML for this Bang AutoGlass Webflow CMS blog post.",
            "",
            `Row data: ${JSON.stringify(rowPayload, null, 2)}`,
            "",
            `External SERP results, if available: ${JSON.stringify(serp.results, null, 2)}`,
          ].join("\n"),
          tools: [getOpenAIWebSearchTool()],
          reasoning: {
            effort: REASONING_EFFORT,
          },
          text: {
            verbosity: "high",
            format: {
              type: "json_schema",
              name: "bang_autoglass_blog_body_generation",
              strict: true,
              schema: bodyGenerationSchema(),
            },
          },
          store: false,
        } as any),
      "OpenAI blog body generation"
    )

    const parsed = parseJsonFromOpenAIResponse(response, "OpenAI blog body generation")
    const bodyHtml = firstString(parsed.bodyHtml)

    if (!bodyHtml.trim()) {
      throw new Error("Body generation returned empty bodyHtml.")
    }

    await markStep({
      supabase,
      stepId,
      status: "success",
      requestPayload,
      responsePayload: {
        id: response.id || null,
        usage: response.usage || {},
        output_text: getTextFromResponse(response),
      },
      parsedOutput: parsed,
      researchPayload: {
        query: researchQuery,
        provider: serp.provider,
        results: serp.results,
        researchSummary: firstString(parsed.researchSummary),
        customerQuestions: Array.isArray(parsed.customerQuestions) ? parsed.customerQuestions : [],
        citations: Array.isArray(parsed.citations) ? parsed.citations : [],
      },
      toolCalls: extractToolCalls(response),
      usagePayload: response.usage || {},
      openaiResponseId: response.id || null,
    })

    await insertResearchRecord({
      supabase,
      jobId,
      rowId,
      stepId,
      phase: "body_generation",
      provider: "openai_web_search",
      query: researchQuery,
      organicResults: [],
      researchSummary: firstString(parsed.researchSummary),
      customerQuestions: Array.isArray(parsed.customerQuestions)
        ? parsed.customerQuestions.map(String)
        : [],
      citations: Array.isArray(parsed.citations) ? parsed.citations : [],
      rawResponse: {
        openaiResponseId: response.id || null,
        toolCalls: extractToolCalls(response),
        parsed,
      },
    })

    const { error: updateError } = await supabase
      .from(BLOG_ROWS_TABLE)
      .update({
        status: "phase_2_complete",
        current_phase: "webflow_publish",
        body_html: bodyHtml,
        phase_2_output: cleanPlainObject(parsed),
        body_research: cleanPlainObject({
          query: researchQuery,
          provider: serp.provider,
          externalSerpResults: serp.results,
          researchSummary: firstString(parsed.researchSummary),
          customerQuestions: Array.isArray(parsed.customerQuestions) ? parsed.customerQuestions : [],
          citations: Array.isArray(parsed.citations) ? parsed.citations : [],
        }),
        body_openai_response_id: response.id || null,
        body_attempt_count: Number(row.body_attempt_count || 0) + 1,
        locked_at: null,
        locked_by: null,
        last_error: null,
      })
      .eq("id", rowId)

    if (updateError) {
      throw new Error(`Could not save body generation output: ${safeErrorMessage(updateError)}`)
    }
  } catch (error) {
    const message = safeErrorMessage(error)

    await markStep({
      supabase,
      stepId,
      status: "error",
      error: message,
    }).catch((stepError: unknown) => {
      console.warn(`Could not mark body step error: ${safeErrorMessage(stepError)}`)
    })

    throw error
  }
}

function getCollectionFields(collectionData: any) {
  const candidates = [
    collectionData?.fields,
    collectionData?.collection?.fields,
    collectionData?.data?.fields,
  ]

  for (const candidate of candidates) {
    if (Array.isArray(candidate)) return candidate
  }

  return []
}

function buildCollectionFieldIndex(collectionData: any) {
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

    for (const label of [
      field.displayName,
      field.name,
      field.label,
      slug,
      slug.replace(/-/g, " "),
    ]) {
      const normalized = normalizeLabel(label)
      if (normalized && !labelToSlug.has(normalized)) {
        labelToSlug.set(normalized, slug)
      }
    }
  }

  return { validSlugs, labelToSlug }
}

function remapOrFilterFieldDataForCollection(options: {
  fieldData: Record<string, unknown>
  collectionData: any
}) {
  const { fieldData, collectionData } = options
  const { validSlugs, labelToSlug } = buildCollectionFieldIndex(collectionData)

  const next: Record<string, unknown> = {}
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

async function prepareFieldDataForCollection(fieldData: Record<string, unknown>) {
  const collectionResult = await webflowRequest({
    method: "GET",
    path: `/collections/${WEBFLOW_BLOG_COLLECTION_ID}`,
  })

  const prepared = remapOrFilterFieldDataForCollection({
    fieldData,
    collectionData: collectionResult.data,
  })

  if (!prepared.fieldData.name || !prepared.fieldData.slug) {
    throw new Error(
      "Prepared Webflow fieldData is missing required name or slug after collection field validation."
    )
  }

  return {
    ...prepared,
    collection: collectionResult.data,
  }
}

function buildFinalFieldData(row: Record<string, unknown>, job: Record<string, unknown>) {
  const mapping = {
    ...DEFAULT_FIELD_MAPPING,
    ...asRecord(job.field_mapping),
  } as Record<string, string>

  const notes = asRecord(row.notes)
  const tags = Array.isArray(row.tags) ? row.tags : []

  const fieldData: Record<string, unknown> = {}

  const finalName = firstString(row.webflow_name, row.selected_title, row.name_seed)

  fieldData[mapping.name] = finalName
  fieldData[mapping.slug] = firstString(row.webflow_slug, slugify(finalName))
  fieldData[mapping.articleAuthor] = firstString(row.article_author, job.article_author_default, DEFAULT_ARTICLE_AUTHOR)
  fieldData[mapping.publicationDate] = firstString(row.publication_date)
  fieldData[mapping.category] = firstString(row.category, job.category)
  fieldData[mapping.body] = firstString(row.body_html)
  fieldData[mapping.postSummary] = firstString(row.post_summary, row.excerpt)
  fieldData[mapping.excerpt] = firstString(row.post_summary, row.excerpt)

  if (firstString(row.make)) fieldData[mapping.make] = firstString(row.make)
  if (firstString(row.model)) fieldData[mapping.model] = firstString(row.model)
  if (firstString(row.service_label)) fieldData[mapping.serviceLabel] = firstString(row.service_label)
  if (firstString(row.city)) fieldData[mapping.city] = firstString(row.city)
  if (firstString(row.state)) fieldData[mapping.state] = firstString(row.state)

  if (firstString(row.meta_title)) fieldData[mapping.metaTitle] = firstString(row.meta_title)
  if (firstString(row.meta_description)) fieldData[mapping.metaDescription] = firstString(row.meta_description)
  if (firstString(row.faq_html)) fieldData[mapping.faqHtml] = firstString(row.faq_html)
  if (firstString(row.cta_text)) fieldData[mapping.ctaText] = firstString(row.cta_text)
  if (firstString(row.image_prompt)) fieldData[mapping.imagePrompt] = firstString(row.image_prompt)
  if (Object.keys(notes).length) fieldData[mapping.notes] = JSON.stringify(notes)
  if (tags.length) fieldData[mapping.tags] = tags

  return compactObject(fieldData)
}

function extractWebflowItemId(result: any) {
  return firstString(
    result?.data?.id,
    result?.data?._id,
    result?.data?.items?.[0]?.id,
    result?.data?.items?.[0]?._id,
    result?.result?.data?.id,
    result?.result?.data?._id,
    result?.result?.data?.items?.[0]?.id,
    result?.result?.data?.items?.[0]?._id
  )
}

async function publishRowToWebflow(options: {
  supabase: any
  job: Record<string, unknown>
  row: Record<string, unknown>
  model: string
}) {
  const { supabase, job, row, model } = options

  const jobId = firstString(job.id)
  const rowId = firstString(row.id)

  const step = await ensureStep({
    supabase,
    jobId,
    rowId,
    stepNumber: 3,
    stepKind: "webflow_publish",
    model,
  })

  const stepId = firstString(step.id)

  try {
    await markStep({
      supabase,
      stepId,
      status: "running",
    })

    const rawFieldData = buildFinalFieldData(row, job)
    const prepared = await prepareFieldDataForCollection(rawFieldData)

    const publishMode = (firstString(job.publish_mode, DEFAULT_PUBLISH_MODE) || "live") as PublishMode
    const execute = job.execute !== false
    const allowPublish = job.allow_publish !== false

    if (!execute) {
      await markStep({
        supabase,
        stepId,
        status: "skipped",
        requestPayload: {
          dryRun: true,
          rawFieldData,
          preparedFieldData: prepared.fieldData,
        },
        parsedOutput: {
          message: "Dry run enabled. Webflow item was not created.",
        },
      })

      const { error } = await supabase
        .from(BLOG_ROWS_TABLE)
        .update({
          status: "success",
          current_phase: "done",
          final_field_data: cleanPlainObject(prepared.fieldData),
          completed_at: new Date().toISOString(),
          locked_at: null,
          locked_by: null,
          last_error: null,
        })
        .eq("id", rowId)

      if (error) throw new Error(`Could not save dry-run publish output: ${safeErrorMessage(error)}`)
      return
    }

    if (publishMode === "live" && !allowPublish) {
      throw new Error("Publishing is disabled for this job, but publish_mode is live.")
    }

    const body: Record<string, unknown> = {
      fieldData: prepared.fieldData,
      isArchived: false,
      isDraft: publishMode !== "live",
    }

    const backendConfig = asRecord(job.backend_config)
    const cmsLocaleId = firstString(backendConfig.cmsLocaleId)

    if (cmsLocaleId) body.cmsLocaleId = cmsLocaleId

    const path =
      publishMode === "live"
        ? `/collections/${WEBFLOW_BLOG_COLLECTION_ID}/items/live`
        : `/collections/${WEBFLOW_BLOG_COLLECTION_ID}/items`

    const webflowResult = await webflowRequest({
      method: "POST",
      path,
      query: { skipInvalidFiles: true },
      body,
    })

    const webflowItemId = extractWebflowItemId(webflowResult)

    await markStep({
      supabase,
      stepId,
      status: "success",
      requestPayload: {
        path,
        rawFieldData,
        preparedFieldData: prepared.fieldData,
        skippedFields: prepared.skippedFields,
        remappedFields: prepared.remappedFields,
      },
      responsePayload: webflowResult,
      parsedOutput: {
        webflowItemId,
      },
    })

    const { error } = await supabase
      .from(BLOG_ROWS_TABLE)
      .update({
        status: "success",
        current_phase: "done",
        final_field_data: cleanPlainObject(prepared.fieldData),
        webflow_item_id: webflowItemId || null,
        webflow_response: cleanPlainObject(webflowResult),
        publish_attempt_count: Number(row.publish_attempt_count || 0) + 1,
        completed_at: new Date().toISOString(),
        locked_at: null,
        locked_by: null,
        last_error: null,
      })
      .eq("id", rowId)

    if (error) throw new Error(`Could not save Webflow publish output: ${safeErrorMessage(error)}`)
  } catch (error) {
    const message = safeErrorMessage(error)

    await markStep({
      supabase,
      stepId,
      status: "error",
      error: message,
    }).catch((stepError: unknown) => {
      console.warn(`Could not mark publish step error: ${safeErrorMessage(stepError)}`)
    })

    throw error
  }
}

async function processOnePhase(options: {
  openai: OpenAI
  supabase: any
  job: Record<string, unknown>
  row: Record<string, unknown>
  model: string
}): Promise<PhaseResult> {
  const { openai, supabase, job, row, model } = options
  const rowId = firstString(row.id)
  const jobId = firstString(job.id)

  try {
    let latestJob = job

    const jobCheck = await supabase
      .from(BLOG_CREATION_TABLE)
      .select("*")
      .eq("id", jobId)
      .single()

    if (jobCheck.error || !jobCheck.data) {
      throw new Error(`Could not reload job before processing: ${safeErrorMessage(jobCheck.error)}`)
    }

    latestJob = jobCheck.data as Record<string, unknown>

    if (["paused", "cancelled", "completed", "completed_with_errors", "error"].includes(firstString(latestJob.status))) {
      await supabase
        .from(BLOG_ROWS_TABLE)
        .update({
          status: "queued",
          current_phase: "queued",
          locked_at: null,
          locked_by: null,
          last_error: `Worker released row because job is ${latestJob.status}.`,
          updated_at: new Date().toISOString(),
        })
        .eq("id", rowId)

      return {
        success: false,
        jobId,
        rowId,
        message: `Job is ${latestJob.status}.`,
      }
    }

    let latestRow = await reloadRow(supabase, rowId)

    if (!firstString(latestRow.webflow_name, latestRow.selected_title)) {
      await supabase
        .from(BLOG_ROWS_TABLE)
        .update({
          status: "running",
          current_phase: "title_and_field_generation",
          last_error: null,
          updated_at: new Date().toISOString(),
        })
        .eq("id", rowId)

      await generateMetadata({
        openai,
        supabase,
        job: latestJob,
        row: latestRow,
        model,
      })

      return {
        success: true,
        jobId,
        rowId,
        phase: "title_and_field_generation",
      }
    }

    if (!firstString(latestRow.body_html)) {
      await supabase
        .from(BLOG_ROWS_TABLE)
        .update({
          status: "running",
          current_phase: "body_generation",
          last_error: null,
          updated_at: new Date().toISOString(),
        })
        .eq("id", rowId)

      latestRow = await reloadRow(supabase, rowId)

      await generateBody({
        openai,
        supabase,
        job: latestJob,
        row: latestRow,
        model,
      })

      return {
        success: true,
        jobId,
        rowId,
        phase: "body_generation",
      }
    }

    if (!firstString(latestRow.webflow_item_id)) {
      await supabase
        .from(BLOG_ROWS_TABLE)
        .update({
          status: "publishing",
          current_phase: "webflow_publish",
          last_error: null,
          updated_at: new Date().toISOString(),
        })
        .eq("id", rowId)

      latestRow = await reloadRow(supabase, rowId)

      await publishRowToWebflow({
        supabase,
        job: latestJob,
        row: latestRow,
        model,
      })

      return {
        success: true,
        jobId,
        rowId,
        phase: "webflow_publish",
      }
    }

    await supabase
      .from(BLOG_ROWS_TABLE)
      .update({
        status: "success",
        current_phase: "done",
        locked_at: null,
        locked_by: null,
        completed_at: latestRow.completed_at || new Date().toISOString(),
        last_error: null,
        updated_at: new Date().toISOString(),
      })
      .eq("id", rowId)

    return {
      success: true,
      jobId,
      rowId,
      phase: "webflow_publish",
      message: "Row already had Webflow item ID and was marked success.",
    }
  } catch (error) {
    const message = safeErrorMessage(error)

    await supabase
      .from(BLOG_ROWS_TABLE)
      .update({
        status: "error",
        current_phase: "error",
        last_error: message,
        completed_at: new Date().toISOString(),
        locked_at: null,
        locked_by: null,
        updated_at: new Date().toISOString(),
      })
      .eq("id", rowId)

    return {
      success: false,
      jobId,
      rowId,
      error: message,
    }
  }
}

async function processOneWorker(options: {
  openai: OpenAI
  supabase: any
  model: string
  workerSlot: number
}): Promise<PhaseResult> {
  const { openai, supabase, model, workerSlot } = options

  const job = await findRunnableJob(supabase)

  if (!job) {
    return {
      success: true,
      message: "No runnable jobs found.",
    }
  }

  const jobId = firstString(job.id)
  await recoverStaleLocks(supabase, jobId)

  const currentJob = await supabase
    .from(BLOG_CREATION_TABLE)
    .select("*")
    .eq("id", jobId)
    .single()

  if (currentJob.error || !currentJob.data) {
    throw new Error(`Could not reload job: ${safeErrorMessage(currentJob.error)}`)
  }

  if (["paused", "cancelled", "completed", "completed_with_errors", "error"].includes(firstString(currentJob.data.status))) {
    return {
      success: true,
      jobId,
      message: `Job is ${currentJob.data.status}.`,
    }
  }

  await supabase
    .from(BLOG_CREATION_TABLE)
    .update({
      status: "running",
      started_at: currentJob.data.started_at || new Date().toISOString(),
      updated_at: new Date().toISOString(),
      last_error: null,
    })
    .eq("id", jobId)

  const workerId = `render:blog-worker:${workerSlot}:${process.pid}:${crypto.randomUUID()}`

  const row = await claimNextRow({
    supabase,
    jobId,
    workerId,
  })

  if (!row) {
    await refreshJobCounts(supabase, jobId)

    return {
      success: true,
      jobId,
      message: "No claimable rows found.",
    }
  }

  const result = await processOnePhase({
    openai,
    supabase,
    job: currentJob.data,
    row,
    model,
  })

  await refreshJobCounts(supabase, jobId)

  return result
}

async function workerLoop() {
  const supabaseUrl = getRequiredEnv("SUPABASE_URL")
  const supabaseServiceRoleKey = getRequiredEnv("SUPABASE_SERVICE_ROLE_KEY")
  const openaiApiKey = getRequiredEnv("OPENAI_API_KEY")
  const model = getBestModelFromSecret()

  const supabase = createClient(supabaseUrl, supabaseServiceRoleKey, {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
    },
  })

  const openai = new OpenAI({ apiKey: openaiApiKey })

  console.log("Bang Webflow Blog Worker started.")
  console.log(`Model: ${model}`)
  console.log(`Concurrency: ${getWorkerConcurrency()}`)
  console.log(`Poll interval: ${getPollIntervalMs()}ms`)
  console.log(`Web search tool: ${getOpenAIWebSearchToolType()}`)

  while (!shutdownState.shuttingDown) {
    if (!getWorkerEnabled()) {
      console.log("WORKER_ENABLED is false. Sleeping.")
      await sleep(getPollIntervalMs())
      continue
    }

    const concurrency = getWorkerConcurrency()

    try {
      const results = await Promise.allSettled(
        Array.from({ length: concurrency }, (_, index) =>
          processOneWorker({
            openai,
            supabase,
            model,
            workerSlot: index + 1,
          })
        )
      )

      for (const result of results) {
        if (result.status === "fulfilled") {
          const value = result.value

          if (value.phase) {
            console.log(
              `[processed] job=${value.jobId || ""} row=${value.rowId || ""} phase=${value.phase} success=${value.success}`
            )
          } else if (value.message) {
            console.log(`[idle] ${value.message}`)
          }

          if (value.error) {
            console.error(`[row-error] ${value.error}`)
          }
        } else {
          console.error(`[worker-error] ${safeErrorMessage(result.reason)}`)
        }
      }
    } catch (error) {
      console.error(`[loop-error] ${safeErrorMessage(error)}`)
    }

    await sleep(getPollIntervalMs())
  }

  console.log("Bang Webflow Blog Worker stopped.")
}

process.on("SIGTERM", () => {
  console.log("SIGTERM received. Shutting down after current cycle.")
  shutdownState.shuttingDown = true
})

process.on("SIGINT", () => {
  console.log("SIGINT received. Shutting down after current cycle.")
  shutdownState.shuttingDown = true
})

workerLoop().catch((error) => {
  console.error(`Fatal worker error: ${safeErrorMessage(error)}`)
  process.exit(1)
})
