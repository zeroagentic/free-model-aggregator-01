// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** The fixed `created` epoch second every translated OpenAI document carries (no clock reads). */
export const FIXED_CREATED = FIXED_OPENAI_CREATED;

/** Anthropic's or Gemini's stop reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens' || stop === 'MAX_TOKENS' || stop === 'length') return 'length';
  if (stop === 'tool_use' || stop === 'tool_calls' || stop === 'function_call') return 'tool_calls';
  if (stop === 'refusal' || stop === 'SAFETY' || stop === 'content_filter') return 'content_filter';
  return 'stop';
}

/** One OpenAI image_url as an Anthropic image source: a data URL becomes base64, the rest a URL. */
function imageSourceOf(url: string): Json {
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { type: 'base64', media_type: data[1], data: data[2] };
  return { type: 'url', url };
}

/** One OpenAI image_url as a Gemini part: a data URL becomes inlineData, the rest fileData. */
function geminiImagePart(url: string): Json {
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** An Anthropic image block as a Gemini part. */
function geminiImagePartOf(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** A tool's JSON-schema parameters in a provider's spelling: the OpenAI function's or the Anthropic tool's. */
function schemaOf(raw: Json): Json {
  if (isRecord(raw.parameters)) return raw.parameters;
  if (isRecord(raw.input_schema)) return raw.input_schema;
  return { type: 'object', properties: {} };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: the mirror of
 * anthropicToOpenAI. System messages join into the system prompt, image_url parts become image blocks,
 * tool_calls become tool_use blocks, tool messages become tool_result blocks, `stop` becomes
 * stop_sequences, and the output cap passes only when the client sent one (an invented cap would change
 * the provider's echo). JSON mode is never carried: such requests are routed to wires that speak it.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  let system = '';
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'user') {
      if (typeof raw.content === 'string') {
        messages.push({ role: 'user', content: raw.content });
        continue;
      }
      if (!Array.isArray(raw.content)) continue;
      const blocks: Json[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          blocks.push({ type: 'image', source: imageSourceOf(part.image_url.url) });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
    }
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : [];
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: schemaOf(raw.function),
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** A request's stop list and JSON-mode flag, read off an OpenAI body. */
function openAIKnobs(body: Json): { stops: string[]; jsonMode: boolean } {
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : [];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { stops, jsonMode: format === 'json_object' || format === 'json_schema' };
}

/** The Gemini generationConfig both client wires share: cap, stops, JSON mode, sampling, only as sent. */
function geminiConfig(stops: string[], jsonMode: boolean, maxTokens: unknown, temperature: unknown, topP: unknown): Json | null {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  return Object.keys(config).length > 0 ? config : null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. System messages become the
 * systemInstruction, user parts keep their order (text, then the image as inlineData or fileData),
 * assistant turns become model contents with functionCall parts, and tool messages fold into the
 * preceding user content as functionResponse parts so the last user text stays the client's.
 */
export function openAIToGemini(body: Json): Json {
  let system = '';
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') parts.push({ text: raw.content });
      else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') parts.push(geminiImagePart(part.image_url.url));
        }
      }
      contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (typeof call.id === 'string') toolNames.set(call.id, name);
        parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const name = (typeof raw.tool_call_id === 'string' ? toolNames.get(raw.tool_call_id) : undefined) ?? '';
      const part: Json = {
        functionResponse: { name, response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } },
      };
      const last = contents[contents.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.parts)) last.parts.push(part);
      else contents.push({ role: 'user', parts: [part] });
    }
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: schemaOf(raw.function),
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const knobs = openAIKnobs(body);
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens;
  const config = geminiConfig(knobs.stops, knobs.jsonMode, maxTokens, body.temperature, body.top_p);
  if (config !== null) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system prompt becomes the
 * systemInstruction, image blocks become inlineData or fileData, tool_use blocks become functionCall
 * parts and tool_result blocks functionResponse parts inside their own user content, so the last user
 * text and its image are the client's own.
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') parts.push({ text: raw.content });
    else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (!assistant && block.type === 'image') {
          const part = geminiImagePartOf(block);
          if (part !== null) parts.push(part);
        } else if (!assistant && block.type === 'tool_result') {
          const name = (typeof block.tool_use_id === 'string' ? toolNames.get(block.tool_use_id) : undefined) ?? '';
          parts.push({ functionResponse: { name, response: { result: toolResultText(block.content) } } });
        } else if (assistant && block.type === 'tool_use') {
          const name = typeof block.name === 'string' ? block.name : '';
          if (typeof block.id === 'string') toolNames.set(block.id, name);
          parts.push({ functionCall: { name, args: isRecord(block.input) ? block.input : {} } });
        }
      }
    }
    contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: schemaOf(raw) }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s) => typeof s === 'string') : [];
  const config = geminiConfig(stops, false, body.max_tokens, body.temperature, body.top_p);
  if (config !== null) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object naming `model`. */
export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const calls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: typeof body.id === 'string' ? body.id.replace(/^msg_/, 'chatcmpl-') : 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: finishReasonOf(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini answer's text, tool calls, stop reason and usage, read once for both client wires. */
export function geminiAnswerOf(body: Json): { text: string; calls: { name: string; args: Json }[]; stop: string; input: number; output: number } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const calls: { name: string; args: Json }[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      calls.push({
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        args: isRecord(part.functionCall.args) ? part.functionCall.args : {},
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    calls,
    stop: typeof first.finishReason === 'string' ? first.finishReason : 'STOP',
    input: numberOrZero(usage.promptTokenCount),
    output: numberOrZero(usage.candidatesTokenCount),
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: answer.text,
          ...(answer.calls.length > 0
            ? { tool_calls: answer.calls.map((call, i) => ({ id: `call_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args) } })) }
            : {}),
        },
        finish_reason: finishReasonOf(answer.stop),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  answer.calls.forEach((call, i) => content.push({ type: 'tool_use', id: `toolu_${i}`, name: call.name, input: call.args }));
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(answer.stop === 'STOP' ? 'end_turn' : answer.stop === 'MAX_TOKENS' ? 'max_tokens' : answer.stop === 'SAFETY' ? 'refusal' : 'end_turn'),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** One parsed server-sent event frame: its event name (null when unnamed) and its joined data. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Read a provider's event stream frame by frame, in arrival order. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): SseFrame[] => {
    const out: SseFrame[] = [];
    for (const frame of text.split('\n\n')) {
      if (frame.trim() === '') continue;
      let event: string | null = null;
      const data: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      if (data.length > 0) out.push({ event, data: data.join('\n') });
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const parsed of framesOf(frame)) yield parsed;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const parsed of framesOf(buffer)) yield parsed;
  } finally {
    reader.releaseLock();
  }
}

/** One OpenAI stream chunk frame in the shape the provider itself sends. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage?: Json): string {
  const chunk: Json = {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
  if (usage !== undefined) {
    chunk.choices = [];
    chunk.usage = usage;
  }
  return sseEvent(null, chunk);
}

/**
 * An Anthropic event stream as an OpenAI Chat Completions stream: a role chunk, content and tool-call
 * chunks as the blocks move, then the finish chunk, the usage chunk and [DONE], in the order the
 * client's checker wants.
 */
export async function* openAIEventsFromAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  const toolIndexes = new Map<number, number>();
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(model, { role: 'assistant', content: '' }, null)];
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(model, {}, stop ?? 'stop'),
      openAIChunk(model, {}, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }),
      'data: [DONE]\n\n',
    ];
  };
  for await (const frame of sseFrames(stream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      yield* start();
    } else if (frame.event === 'content_block_start') {
      yield* start();
      const index = typeof data.index === 'number' ? data.index : 0;
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (block.type === 'tool_use') {
        const at = toolIndexes.size;
        toolIndexes.set(index, at);
        yield openAIChunk(
          model,
          {
            tool_calls: [
              {
                index: at,
                id: typeof block.id === 'string' ? block.id : `toolu_${at}`,
                type: 'function',
                function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' },
              },
            ],
          },
          null,
        );
      }
    } else if (frame.event === 'content_block_delta') {
      yield* start();
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield openAIChunk(model, { content: delta.text }, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        const index = typeof data.index === 'number' ? data.index : 0;
        const at = toolIndexes.get(index) ?? 0;
        yield openAIChunk(model, { tool_calls: [{ index: at, function: { arguments: delta.partial_json } }] }, null);
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = finishReasonOf(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      yield* start();
      yield* finish();
    }
  }
  yield* start();
  yield* finish();
}

/**
 * A Gemini event stream as an OpenAI Chat Completions stream: text pieces become content chunks, the
 * closing frame's finishReason and usageMetadata become the finish and usage chunks, then [DONE].
 */
export async function* openAIEventsFromGemini(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(model, { role: 'assistant', content: '' }, null)];
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(model, {}, stop ?? 'stop'),
      openAIChunk(model, {}, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }),
      'data: [DONE]\n\n',
    ];
  };
  for await (const frame of sseFrames(stream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    yield* start();
    const answer = geminiAnswerOf(data);
    if (answer.text !== '') yield openAIChunk(model, { content: answer.text }, null);
    for (const [i, call] of answer.calls.entries()) {
      yield openAIChunk(
        model,
        { tool_calls: [{ index: i, id: `call_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args) } }] },
        null,
      );
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    if (typeof first.finishReason === 'string') {
      stop = finishReasonOf(first.finishReason);
      inputTokens = answer.input;
      outputTokens = answer.output;
    }
  }
  yield* start();
  yield* finish();
}

/**
 * A Gemini event stream as the Anthropic event sequence: message_start, one text block carrying every
 * piece, then message_delta with the mapped stop reason and the closing frame's output tokens, and
 * message_stop. Gemini reports usage only in its closing frame, so message_start carries zeroes and
 * the scored output count lands in message_delta.
 */
export async function* anthropicEventsFromGemini(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let blockOpen = false;
  let finished = false;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_gemini',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
    ];
  };
  const openBlock = (): string[] => {
    if (blockOpen) return [];
    blockOpen = true;
    return [sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })];
  };
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (blockOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      blockOpen = false;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };
  for await (const frame of sseFrames(stream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    yield* start();
    const answer = geminiAnswerOf(data);
    if (answer.text !== '') {
      yield* openBlock();
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } });
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    if (typeof first.finishReason === 'string') {
      stopReason = first.finishReason === 'MAX_TOKENS' ? 'max_tokens' : first.finishReason === 'SAFETY' ? 'refusal' : 'end_turn';
      outputTokens = answer.output;
    }
  }
  yield* start();
  yield* finish();
}
