import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import type { Tool, ToolCall } from "../src/types.ts";
import {
	type MiniCPM5ScanEvent,
	MiniCPM5Scanner,
	renderMiniCPM5ToolCall,
	renderMiniCPM5ToolResponse,
} from "../src/utils/minicpm5-tool-parser.ts";

const TOOLS: Tool[] = [
	{
		name: "read",
		description: "Read a file",
		parameters: Type.Object({ path: Type.String(), count: Type.Optional(Type.Number()) }),
	},
	{
		name: "write",
		description: "Write a file",
		parameters: Type.Object({ path: Type.String(), content: Type.String() }),
	},
];

function scan(text: string, tools: readonly Tool[] = TOOLS): MiniCPM5ScanEvent[] {
	const scanner = new MiniCPM5Scanner({ tools });
	const events: MiniCPM5ScanEvent[] = [];
	for (const char of text) events.push(...scanner.feed(char));
	events.push(...scanner.flush());
	return events;
}

function toolEnds(events: readonly MiniCPM5ScanEvent[]): Extract<MiniCPM5ScanEvent, { type: "toolEnd" }>[] {
	return events.filter((event): event is Extract<MiniCPM5ScanEvent, { type: "toolEnd" }> => event.type === "toolEnd");
}

describe("MiniCPM5 tool parser", () => {
	it("parses the native function/param format", () => {
		const calls = toolEnds(
			scan('<function name="read"><param name="path">src/a.ts</param><param name="count">2</param></function>'),
		);
		expect(calls).toHaveLength(1);
		expect(calls[0]?.name).toBe("read");
		expect(calls[0]?.arguments).toEqual({ path: "src/a.ts", count: 2 });
	});

	it("parses consecutive calls and keeps surrounding text", () => {
		const events = scan(
			'before <function name="read"><param name="path">a</param></function><function name="read"><param name="path">b</param></function> after',
		);
		expect(toolEnds(events).map((call) => call.arguments)).toEqual([{ path: "a" }, { path: "b" }]);
		const text = events.flatMap((event) => (event.type === "text" ? [event.text] : [])).join("");
		expect(text).toBe("before  after");
	});

	it("normalizes tokenizer-space and collapsed attribute forms", () => {
		const calls = toolEnds(scan('<functionĠname="read"><paramname="path">src/a.ts</param></function>'));
		expect(calls[0]?.arguments).toEqual({ path: "src/a.ts" });
	});

	it("preserves CDATA string payloads verbatim", () => {
		const content = "line 1\nif (a < b && c) {}\nline 3";
		const calls = toolEnds(
			scan(
				`<function name="write"><param name="path">out.txt</param><param name="content"><![CDATA[${content}]]></param></function>`,
			),
		);
		expect(calls[0]?.arguments).toEqual({ path: "out.txt", content });
	});

	it("does not normalize tokenizer text inside CDATA", () => {
		const content = '<functionname="not-a-tag">Ġ<paramname="literal">';
		const calls = toolEnds(
			scan(
				`<function name="write"><param name="path">out.txt</param><param name="content"><![CDATA[${content}]]></param></function>`,
			),
		);
		expect(calls[0]?.arguments).toEqual({ path: "out.txt", content });
	});

	it("preserves whitespace in non-CDATA string parameters and decodes entities", () => {
		const calls = toolEnds(scan('<function name="read"><param name="path">  a &amp; b  </param></function>'));
		expect(calls[0]?.arguments).toEqual({ path: "  a & b  " });
	});

	it("passes calls to unknown tools through as text", () => {
		const events = scan('<function name="nope"><param name="x">1</param></function>');
		expect(toolEnds(events)).toHaveLength(0);
		expect(events.some((event) => event.type === "text")).toBe(true);
	});

	it("emits thinking blocks", () => {
		const events = scan("<think>plan</think>done");
		expect(events.find((event) => event.type === "thinkingEnd")).toEqual({ type: "thinkingEnd", thinking: "plan" });
	});

	it("keeps exact raw blocks for diagnostics", () => {
		const raw = '<function name="read"><param name="path">src/a.ts</param></function>';
		expect(toolEnds(scan(raw))[0]?.rawBlock).toBe(raw);
	});

	it("rejects the invoke/parameter hybrid", () => {
		const calls = toolEnds(scan('<function name="read"><parameter name="path">src/a.ts</parameter></function>'));
		expect(calls[0]?.arguments).toEqual({ __parseError: expect.any(String), __rawJson: expect.any(String) });
	});

	it("marks an unfinished call as invalid instead of executing a preview", () => {
		const calls = toolEnds(scan('<function name="read"><param name="path">src/a.ts</param>'));
		expect(calls).toHaveLength(1);
		expect(calls[0]?.arguments).toEqual({ __parseError: expect.any(String), __rawJson: expect.any(String) });
	});

	it("renders calls that the scanner parses back", () => {
		const call: ToolCall = {
			type: "toolCall",
			id: "call_1",
			name: "write",
			arguments: { path: "out.txt", content: "a < b\nsecond line" },
		};
		const rendered = renderMiniCPM5ToolCall(call, TOOLS);
		expect(rendered).toContain('<param name="content"><![CDATA[');
		expect(toolEnds(scan(rendered))[0]?.arguments).toEqual(call.arguments);
	});

	it("renders tool responses", () => {
		expect(renderMiniCPM5ToolResponse("/tmp/project")).toBe("<tool_response>\n/tmp/project\n</tool_response>");
	});
});
