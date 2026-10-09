import { Type } from "typebox";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import type { Model, ToolCall } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const mockState = vi.hoisted(() => ({
	chunks: [] as unknown[],
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: () => {
					const chunks = mockState.chunks;
					const stream = {
						async *[Symbol.asyncIterator]() {
							for (const chunk of chunks) yield chunk;
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}
	return { default: FakeOpenAI };
});

function buildModel(id: string): Model<"openai-completions"> {
	return {
		id,
		name: id,
		api: "openai-completions",
		provider: "local",
		baseUrl: "http://127.0.0.1:8000/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 4096,
	};
}

const context = normalizeContext({
	messages: [{ role: "user", content: "read it", timestamp: Date.now() }],
	tools: [
		{
			name: "read",
			description: "Read a file",
			parameters: Type.Object({ path: Type.String(), count: Type.Optional(Type.Number()) }),
		},
	],
});

function contentChunks(parts: string[]): unknown[] {
	return [
		...parts.map((content) => ({ id: "c1", choices: [{ index: 0, delta: { content } }] })),
		{ id: "c1", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
	];
}

describe("OpenAI completions MiniCPM5 in-band tool calls", () => {
	beforeEach(() => {
		mockState.chunks = [];
	});

	it("turns <function> text into a tool call and reports toolUse", async () => {
		mockState.chunks = contentChunks([
			"Reading. <func",
			'tion name="read"><param name="path">src/a.ts</pa',
			'ram><param name="count">2</param></function>',
		]);

		const message = await streamOpenAICompletions(buildModel("openbmb/MiniCPM5-8B"), context, {
			apiKey: "test",
		}).result();

		const call = message.content.find((block): block is ToolCall => block.type === "toolCall");
		expect(call?.name).toBe("read");
		expect(call?.arguments).toEqual({ path: "src/a.ts", count: 2 });
		expect(message.content.find((block) => block.type === "text")).toMatchObject({ text: "Reading. " });
		expect(message.stopReason).toBe("toolUse");
	});

	it("leaves the same text alone for other models", async () => {
		mockState.chunks = contentChunks(['<function name="read"><param name="path">a</param></function>']);

		const message = await streamOpenAICompletions(buildModel("some-other-model"), context, {
			apiKey: "test",
		}).result();

		expect(message.content.some((block) => block.type === "toolCall")).toBe(false);
		expect(message.stopReason).toBe("stop");
	});
});
