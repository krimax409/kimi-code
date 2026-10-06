import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { foldWireRecords } from '../../../agent/replayBuilder/fold';
import { projectStrict } from '../../../agent/contextProjector/projection';
import type { AnthropicRawStreamEvent } from '#/llm/requester/bases/anthropic/contract';

import { UNKNOWN_CAPABILITY } from '#/llm/capability';
import { providerImagePolicy } from '#/llm/media/image-formats';
import { createMessageAccumulator, type Message } from '#/llm/message';
import { createAnthropicFormat } from '#/llm/requester/bases/anthropic/format';
import { assistantMessageSchema } from '../../agent/historySchema';
import type { LlmModel } from '#/llm/model';
import { lowerMessage } from '#/llm/requester/bases/anthropic/lower';
import { createAnthropicRequester } from '#/llm/requester/bases/anthropic/requester';
import type { LlmClientContext } from '#/llm/requester/requester';
import { kimiAnthropicTrait } from '#/llm-kimi/trait';

const routedModel: LlmModel = {
  provider: 'anthropic',
  model: 'test-model',
  capability: UNKNOWN_CAPABILITY,
  baseUrl: 'https://example.test/v1',
};

const HEIC_URL = 'data:image/heic;base64,AAAA';

const message: Message = {
  role: 'user',
  content: [{ type: 'image_url', imageUrl: { url: HEIC_URL } }],
};

const HEIC_BLOCK = {
  type: 'image',
  source: { type: 'base64', data: 'AAAA', media_type: 'image/heic' },
};

function stubAnthropicClient(): {
  clientFactory: (request: LlmClientContext) => never;
  body: () => Record<string, unknown>;
} {
  const captured: Record<string, unknown>[] = [];
  const events = [
    { type: 'message_start', message: { usage: { input_tokens: 1, output_tokens: 1 } } },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 1 } },
    { type: 'message_stop' },
  ];
  return {
    clientFactory: () =>
      ({
        messages: {
          create: (params: Record<string, unknown>) => {
            captured.push(params);
            return {
              withResponse: async () => ({
                data: {
                  async *[Symbol.asyncIterator]() {
                    for (const event of events) yield event;
                  },
                },
                response: new Response(null),
              }),
            };
          },
        },
      }) as never,
    body: () => {
      const last = captured.at(-1);
      if (last === undefined) throw new Error('expected client to be called');
      return last;
    },
  };
}

describe('anthropic lowering of inline images', () => {
  it('forwards a base64 image the Kimi policy accepts even though the route id is anthropic', () => {
    const wire = lowerMessage(message, providerImagePolicy('kimi').acceptedMimes);
    expect(wire[0]?.content[0]).toEqual(HEIC_BLOCK);
  });

  it('refuses a base64 image outside the baseline set when no trait widens it', () => {
    expect(() => lowerMessage(message, providerImagePolicy().acceptedMimes)).toThrow(
      /Unsupported media type for base64 image: image\/heic/,
    );
  });

  it('sends the HEIC block on the wire when Kimi is reached over the Anthropic protocol', async () => {
    const client = stubAnthropicClient();
    const requester = createAnthropicRequester({
      trait: kimiAnthropicTrait,
      clientFactory: client.clientFactory,
    });
    await requester.generate(
      { model: routedModel },
      { messages: [message] },
      { signal: new AbortController().signal },
    );
    const messages = client.body()['messages'] as { content: unknown[] }[];
    expect(messages[0]?.content[0]).toMatchObject(HEIC_BLOCK);
  });
});

describe('anthropic fallback replay', () => {
  it('keeps the fallback boundary between signed thinking through persisted history', () => {
    const accumulator = createMessageAccumulator();
    const parse = createAnthropicFormat().createStreamParser();
    const sink = { onDelta: (part: Parameters<typeof accumulator.push>[0]) => accumulator.push(part) };
    parse({ type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Before handoff\n' } }, sink);
    parse({ type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'signature-before' } }, sink);
    parse({ type: 'content_block_start', index: 1, content_block: {
      type: 'fallback',
      from: { model: 'claude-opus-5-5' },
      to: { model: 'claude-opus-4-8' },
      trigger: { type: 'refusal', detail: { reason: 'fixture' } },
    } }, sink);
    parse({ type: 'content_block_delta', index: 2, delta: { type: 'thinking_delta', thinking: 'After handoff\n' } }, sink);
    parse({ type: 'content_block_delta', index: 2, delta: { type: 'signature_delta', signature: 'signature-after' } }, sink);
    parse({ type: 'content_block_start', index: 3, content_block: { type: 'tool_use', id: 'tool-fixture', name: 'lookup', input: {} } }, sink);
    parse({ type: 'content_block_delta', index: 3, delta: { type: 'input_json_delta', partial_json: '{"query":"fixture"}' } }, sink);
    const restored = assistantMessageSchema.parse(JSON.parse(JSON.stringify(accumulator.finish())));
    expect(lowerMessage(restored, providerImagePolicy().acceptedMimes)[0]?.content).toEqual([
      { type: 'thinking', thinking: 'Before handoff\n', signature: 'signature-before' },
      { type: 'fallback', from: { model: 'claude-opus-5-5' }, to: { model: 'claude-opus-4-8' }, trigger: { type: 'refusal', detail: { reason: 'fixture' } } },
      { type: 'thinking', thinking: 'After handoff\n', signature: 'signature-after' },
      { type: 'tool_use', id: 'tool-fixture', name: 'lookup', input: { query: 'fixture' } },
    ]);
  });

  it.each([undefined, null, { type: 'refusal', nested: { reason: 'fixture' } }])('preserves fallback payload with trigger %j', (trigger) => {
    const block = {
      type: 'fallback',
      from: { model: 'claude-opus-5-5', extra: 'original' },
      to: { model: 'claude-opus-4-8' },
      trigger,
    };
    const accumulator = createMessageAccumulator();
    const parse = createAnthropicFormat().createStreamParser();
    const sink = { onDelta: (part: Parameters<typeof accumulator.push>[0]) => accumulator.push(part) };
    parse({ type: 'content_block_start', index: 0, content_block: block }, sink);
    parse({ type: 'content_block_start', index: 1, content_block: { ...block, from: block.to, to: block.from } }, sink);
    const restored = assistantMessageSchema.parse(JSON.parse(JSON.stringify(accumulator.finish())));
    expect(lowerMessage(restored, providerImagePolicy().acceptedMimes)[0]?.content).toEqual([
      JSON.parse(JSON.stringify(block)),
      JSON.parse(JSON.stringify({ ...block, from: block.to, to: block.from })),
    ]);
  });

  it('continues over HTTP after fallback, a tool result and wire journal restoration', async () => {
    const fallback = {
      type: 'fallback',
      from: { model: 'claude-opus-5-5' },
      to: { model: 'claude-opus-4-8' },
      trigger: { type: 'refusal' },
    };
    const firstEvents: AnthropicRawStreamEvent[] = [
      { type: 'message_start', message: { id: 'msg_fixture', usage: { input_tokens: 1, output_tokens: 0 } } },
      { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Before\n' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'sig-before' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'content_block_start', index: 1, content_block: fallback },
      { type: 'content_block_stop', index: 1 },
      { type: 'content_block_start', index: 2, content_block: { type: 'thinking', thinking: '' } },
      { type: 'content_block_delta', index: 2, delta: { type: 'thinking_delta', thinking: 'After\n' } },
      { type: 'content_block_delta', index: 2, delta: { type: 'signature_delta', signature: 'sig-after' } },
      { type: 'content_block_stop', index: 2 },
      { type: 'content_block_start', index: 3, content_block: { type: 'tool_use', id: 'tool_fixture', name: 'lookup', input: {} } },
      { type: 'content_block_delta', index: 3, delta: { type: 'input_json_delta', partial_json: '{"query":"fixture"}' } },
      { type: 'content_block_stop', index: 3 },
      { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 10 } },
      { type: 'message_stop' },
    ];
    const expectedAssistant = [
      { type: 'thinking', thinking: 'Before\n', signature: 'sig-before' },
      fallback,
      { type: 'thinking', thinking: 'After\n', signature: 'sig-after' },
      { type: 'tool_use', id: 'tool_fixture', name: 'lookup', input: { query: 'fixture' } },
    ];
    const received: { messages: { role: string; content: unknown[] }[] }[] = [];
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        const body = JSON.parse(Buffer.concat(chunks).toString());
        received.push(body);
        if (received.length === 2 && JSON.stringify(body.messages[1]?.content) !== JSON.stringify(expectedAssistant)) {
          res.writeHead(400, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ type: 'error', error: { type: 'invalid_request_error', message: 'fallback boundary lost' } }));
          return;
        }
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const events = received.length === 1 ? firstEvents : [
          { type: 'message_start', message: { id: 'msg_continued', usage: { input_tokens: 1, output_tokens: 0 } } },
          { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
          { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'CONTINUED' } },
          { type: 'content_block_stop', index: 0 },
          { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 1 } },
          { type: 'message_stop' },
        ];
        for (const event of events) res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
        res.end();
      });
    });
    const directory = await mkdtemp(join(tmpdir(), 'kimi-fallback-test-'));
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    try {
      const address = server.address();
      if (address === null || typeof address === 'string') throw new Error('expected local HTTP address');
      const requester = createAnthropicRequester();
      const model: LlmModel = { ...routedModel, model: 'claude-opus-5-5', apiKey: 'fixture-key', baseUrl: `http://127.0.0.1:${address.port}` };
      const user: Message = { role: 'user', content: [{ type: 'text', text: 'Use lookup' }] };
      const records: Record<string, unknown>[] = [
        { type: 'metadata', protocol_version: '1.5', created_at: 0 },
        { type: 'context.append_message', message: { ...user, toolCalls: [] }, time: 1 },
        { type: 'context.append_loop_event', event: { type: 'step.begin', uuid: 'step_fixture' }, time: 2 },
      ];
      const accumulator = createMessageAccumulator();
      const errors: unknown[] = [];
      await requester.generate({ model }, { messages: [user] }, {
        signal: new AbortController().signal,
        onEvent(event) {
          if (event.type === 'llm.streaming.part') {
            accumulator.push(event.part);

          }
          if (event.type === 'llm.failed.remote' || event.type === 'llm.failed.syntax') errors.push(event.error);
        },
      });
      const firstMessage = accumulator.finish();
      expect(firstMessage.toolCalls[0]?.name).toBe('lookup');
      for (const part of firstMessage.content) {
        records.push({ type: 'context.append_loop_event', event: { type: 'content.part', stepUuid: 'step_fixture', part }, time: 3 });
      }
      records.push(
        { type: 'context.append_loop_event', event: { type: 'tool.call', stepUuid: 'step_fixture', toolCallId: 'tool_fixture', name: 'lookup', args: { query: 'fixture' } }, time: 4 },
        { type: 'context.append_loop_event', event: { type: 'tool.result', toolCallId: 'tool_fixture', result: { output: 'lookup result' } }, time: 5 },
        { type: 'context.append_loop_event', event: { type: 'step.end', uuid: 'step_fixture', finishReason: 'tool_use' }, time: 6 },
      );
      const path = join(directory, 'wire.jsonl');
      await writeFile(path, records.map((record) => JSON.stringify(record)).join('\n'));
      const restored = foldWireRecords((await readFile(path, 'utf8')).split('\n').map((line) => JSON.parse(line)));
      const history = restored.replay.flatMap((record) => record.type === 'message' ? [record.message] : []);
      const projected = projectStrict(history);
      expect(assistantMessageSchema.parse(projected[1]).content).toEqual(firstMessage.content);
      const next = createMessageAccumulator();
      await requester.generate({ model }, { messages: projected }, {
        signal: new AbortController().signal,
        onEvent(event) {
          if (event.type === 'llm.streaming.part') next.push(event.part);
          if (event.type === 'llm.failed.remote' || event.type === 'llm.failed.syntax') errors.push(event.error);
        },
      });
      expect(errors).toEqual([]);
      expect(received).toHaveLength(2);
      expect(received[1]?.messages[1]?.content).toEqual(expectedAssistant);
      expect(received[1]?.messages[2]?.content[0]).toMatchObject({ type: 'tool_result', tool_use_id: 'tool_fixture' });
      expect(next.finish().content).toEqual([{ type: 'text', text: 'CONTINUED' }]);
    } finally {
      await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
      await rm(directory, { recursive: true, force: true });
    }
  });
});
