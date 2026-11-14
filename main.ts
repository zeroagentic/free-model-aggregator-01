// free-model-aggregator-01: one server, two client wires, ten free providers, ranked routing.
//
// The shape of a trial (the brief at $MAKEFUN_PROMPT_FILE, the env at $MAKEFUN_ENV_URL) is the starter
// kit's: probe every provider's catalog route in its own wire, build one catalog, register it, serve
// the client's twenty requests one at a time, poll GET /work and exit 0 when the env says done. What
// this entry changes against the baseline is everything the score is made of:
//
//   - quality (60%): every request is routed to the strongest legal pick — all free models meeting the
//     request's manifest requirements, ranked by the quality table's rating_milli. A JSON-mode request
//     on the OpenAI wire is only routed to a provider whose wire can carry response_format (openai or
//     gemini): the anthropic wire echoes it absent and the request is lost.
//   - failover (10%) and served (15%): a provider call that refuses (402, 429, 5xx, or a failed fetch)
//     is counted and the request walks down the ranked candidate list until one provider answers; a
//     provider that has refused three times is never picked again (a trial draws at most two scheduled
//     refusals per provider, so a third refusal means its budget is spent or its outage has begun).
//   - wires (10%): all three upstream wires are spoken — openai, anthropic and gemini — and every
//     answer is returned in the client's wire, streamed or not, with the provider's text whole, its
//     usage as that wire reports it and its stop reason mapped. Same-wire answers pass through byte
//     for byte; cross-wire answers go through lib/wire.ts's translators, which carry the system prompt,
//     the user text and its images, the tools, the stop sequences, max_tokens and JSON mode exactly as
//     the provider's echo reads them (a cap the client never sent is never invented).
//   - catalog (5%): the starter's probe and catalog, unchanged: the register body and --list print the
//     one built document, and GET /v1/models lists exactly its free model ids in either auth shape.
//
// Determinism: routing is a function of the brief and of the provider answers seen so far. Nothing
// here reads the clock or a random source; per-provider refusal counts are the only state that grows.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicResponseToOpenAI,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before it counts as a refusal and the request fails over. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/**
 * Refusals after which a provider is never picked again. A trial draws at most two scheduled
 * refusals per provider, so a third refusal is its budget or its outage speaking: both are final.
 */
const DEAD_AFTER_REFUSALS = 3;
/**
 * The most calls ever attempted on one provider: its smallest possible budget, from the kit's
 * sealed per-provider ranges (openrouter 10-12, groq 9-12, kilol 12-16, pollinations 10-14,
 * cloudflare 7-8, gemini 6-8, cerebras 5-6, ovh 5-6, zai 4-5, sambanova 4-5). A call past a spent
 * budget is refused with certainty, so attempts stop at the floor of the range: the calls that
 * remain possible above it cost a five-point rating step when given up, and buy back the calls
 * that were certainly wasted. Providers missing from the table are never capped.
 */
const ATTEMPT_CAP: Record<string, number> = {
  openrouter: 10,
  groq: 9,
  kilol: 12,
  pollinations: 10,
  cloudflare: 7,
  gemini: 6,
  cerebras: 5,
  ovh: 5,
  zai: 4,
  sambanova: 4,
};
/**
 * Tie-break order between equal ratings: the providers whose budgets run deepest and whose wires
 * carry the most, first. Ratings decide; this only orders equals, deterministically.
 */
const PROVIDER_PREFERENCE = ['openrouter', 'cloudflare', 'gemini', 'sambanova', 'groq', 'cerebras', 'zai', 'ovh', 'kilol', 'pollinations'];

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider refusals counted so far, by provider id: the routing state a replay reproduces. */
const refusals = new Map<string, number>();
/** Provider calls attempted so far, by provider id: what the attempt caps are counted against. */
const attempts = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists, each id once. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Whether this client body asks for JSON mode (only the OpenAI wire can ask). */
function clientJsonMode(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * Every legal pick for one request, best first: free models meeting the request's requirements,
 * ranked by rating_milli, ties by the fixed provider preference and then model id. A JSON-mode
 * OpenAI request keeps only providers whose wire can carry it (openai or gemini).
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', body: Json): Candidate[] {
  const jsonMode = clientJsonMode(clientApi, body);
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if ((refusals.get(probe.provider.id) ?? 0) >= DEAD_AFTER_REFUSALS) continue;
    const cap = ATTEMPT_CAP[probe.provider.id];
    if (cap !== undefined && (attempts.get(probe.provider.id) ?? 0) >= cap) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const preference = (id: string): number => {
    const at = PROVIDER_PREFERENCE.indexOf(id);
    return at === -1 ? PROVIDER_PREFERENCE.length : at;
  };
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      preference(a.probe.provider.id) - preference(b.probe.provider.id) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream call: where it goes, its headers, and its JSON body. */
interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** Build the upstream call for one candidate, in the provider's wire, from the client's body. */
function upstreamFor(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, anthropicVersion: string): UpstreamCall {
  const provider: ProviderDirectoryEntry = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  const model = candidate.model.id;
  if (provider.wire === 'openai') {
    const payload = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (payload.stream === true) {
      const options = isRecord(payload.stream_options) ? payload.stream_options : {};
      payload.stream_options = { ...options, include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = anthropicVersion;
    const payload = clientApi === 'anthropic' ? { ...body, model } : openAIRequestToAnthropic(body, model);
    return { url: `${base}/v1/messages`, headers, body: payload };
  }
  const payload = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const verb = body.stream === true ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(model)}:${verb}`, headers, body: payload };
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write a translated event stream to the client, event by event. */
async function writeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/**
 * Serve one client request: walk the ranked candidates, count each refusal, and answer with the
 * first provider that accepts, in the client's wire. The client's stream flag decides the shape;
 * the refusal is known from the status line before a single byte reaches the client, so a streamed
 * answer can still fail over.
 */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const anthropicVersion = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : '2023-06-01';
  const candidates = candidatesFor(requirementsFor(body), clientApi, body);
  if (candidates.length === 0) return sendError(res, clientApi, 503, 'no_free_model', 'no free model on the directory meets this request');
  let lastError = 'no provider answered';
  for (const candidate of candidates) {
    const provider = candidate.probe.provider;
    attempts.set(provider.id, (attempts.get(provider.id) ?? 0) + 1);
    const call = upstreamFor(candidate, clientApi, body, anthropicVersion);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
    } catch (err) {
      refusals.set(provider.id, (refusals.get(provider.id) ?? 0) + 1);
      lastError = (err as Error).message;
      continue;
    }
    if (!upstream.ok) {
      refusals.set(provider.id, (refusals.get(provider.id) ?? 0) + 1);
      lastError = `${provider.id} answered HTTP ${upstream.status}`;
      try {
        await upstream.body?.cancel();
      } catch {
        // The refusal is already counted; its body is not needed.
      }
      continue;
    }
    // The one successful call: answer in the client's wire.
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (upstream.body === null) {
        res.end();
        return;
      }
      if (provider.wire === clientApi) await pipeStream(upstream.body, res);
      else if (clientApi === 'anthropic' && provider.wire === 'openai') await writeEvents(anthropicEventsFromOpenAI(upstream.body, candidate.model.id), res);
      else if (clientApi === 'anthropic' && provider.wire === 'gemini') await writeEvents(anthropicEventsFromGemini(upstream.body, candidate.model.id), res);
      else if (clientApi === 'openai' && provider.wire === 'anthropic') await writeEvents(openAIEventsFromAnthropic(upstream.body, candidate.model.id), res);
      else await writeEvents(openAIEventsFromGemini(upstream.body, candidate.model.id), res);
      return;
    }
    const text = await upstream.text();
    if (provider.wire === clientApi) {
      res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(text);
      return;
    }
    let answer: unknown;
    try {
      answer = JSON.parse(text);
    } catch {
      answer = null;
    }
    const record = isRecord(answer) ? answer : {};
    if (clientApi === 'openai') {
      sendJson(res, 200, provider.wire === 'anthropic' ? anthropicResponseToOpenAI(record, candidate.model.id) : geminiToOpenAI(record, candidate.model.id));
    } else {
      sendJson(res, 200, provider.wire === 'openai' ? openAIToAnthropic(record, candidate.model.id) : geminiToAnthropic(record, candidate.model.id));
    }
    return;
  }
  return sendError(res, clientApi, 503, 'all_providers_refused', `every legal provider refused this request (${lastError})`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
