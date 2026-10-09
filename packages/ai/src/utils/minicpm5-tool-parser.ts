import type { Tool, ToolCall } from "../types.ts";

/**
 * Standalone parser for MiniCPM5's native in-band tool-call format:
 *
 *   <function name="read"><param name="path">src/a.ts</param></function>
 *
 * String values are literal text (optionally wrapped in CDATA); other values are JSON.
 * The scanner is streaming: feed it text chunks and it emits text, thinking and tool events.
 */

export type MiniCPM5ScanEvent =
	| { type: "text"; text: string }
	| { type: "thinkingStart" }
	| { type: "thinkingDelta"; delta: string }
	| { type: "thinkingEnd"; thinking: string }
	| { type: "toolStart"; id: string; name: string }
	| { type: "toolEnd"; id: string; name: string; arguments: Record<string, unknown>; rawBlock: string };

export interface MiniCPM5ScannerOptions {
	/** Tool schemas used to decide which parameters are literal strings. Calls to unknown tools are passed through as text. */
	tools?: readonly Tool[];
	/** Emit `<think>` blocks as thinking events instead of visible text. Default true. */
	parseThinking?: boolean;
}

const FUNCTION_OPEN = "<function";
const FUNCTION_CLOSE = "</function>";
const PARAM_CLOSE = "</param>";
const CDATA_OPEN = "<![CDATA[";
const CDATA_CLOSE = "]]>";
const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";
const TOKENIZER_SPACE = "Ġ";
const RAW_BLOCK_LIMIT = 512;

const FUNCTION_OPEN_RE = /^<function\s+name=(["'])([^"']+)\1[^>]*>$/;
const PARAM_OPEN_RE = /^<param\s+name=(["'])([^"']+)\1[^>]*>$/;

type State = "outside" | "thinking" | "tool";

let idCounter = 0;
function mintToolCallId(): string {
	idCounter = (idCounter + 1) % Number.MAX_SAFE_INTEGER;
	return `minicpm5_${Date.now().toString(36)}_${idCounter.toString(36)}`;
}

/** Length of the longest suffix of `text` that is a proper prefix of one of `tags`. */
function partialSuffixOverlap(text: string, tags: readonly string[]): number {
	let best = 0;
	for (const tag of tags) {
		const max = Math.min(text.length, tag.length - 1);
		for (let k = max; k > best; k--) {
			if (text.endsWith(tag.slice(0, k))) {
				best = k;
				break;
			}
		}
	}
	return best;
}

function normalizeTagHeader(text: string): string {
	return text
		.replaceAll(TOKENIZER_SPACE, " ")
		.replaceAll("<functionname=", "<function name=")
		.replaceAll("<paramname=", "<param name=");
}

function decodeXmlEntities(value: string): string {
	return value.replace(/&(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);/g, (match, entity: string) => {
		switch (entity) {
			case "amp":
				return "&";
			case "lt":
				return "<";
			case "gt":
				return ">";
			case "quot":
				return '"';
			case "apos":
				return "'";
			default: {
				const radix = entity.startsWith("#x") ? 16 : 10;
				const codePoint = Number.parseInt(entity.slice(radix === 16 ? 2 : 1), radix);
				try {
					return String.fromCodePoint(codePoint);
				} catch {
					return match;
				}
			}
		}
	});
}

function schemaTypes(schema: unknown, out: Set<string> = new Set(), depth = 0): Set<string> {
	if (depth > 8 || !schema || typeof schema !== "object" || Array.isArray(schema)) return out;
	const node = schema as Record<string, unknown>;
	if (typeof node.type === "string") out.add(node.type);
	else if (Array.isArray(node.type)) for (const t of node.type) if (typeof t === "string") out.add(t);
	if (node.type === undefined && Array.isArray(node.enum)) {
		for (const value of node.enum) out.add(value === null ? "null" : typeof value);
	}
	if (node.type === undefined && "const" in node) out.add(node.const === null ? "null" : typeof node.const);
	for (const key of ["anyOf", "oneOf", "allOf"] as const) {
		const branch = node[key];
		if (Array.isArray(branch)) for (const sub of branch) schemaTypes(sub, out, depth + 1);
	}
	return out;
}

function isStringOnlySchema(schema: unknown): boolean {
	const types = schemaTypes(schema);
	types.delete("null");
	return types.size === 1 && types.has("string");
}

function coerceValue(raw: string, schema: unknown): unknown {
	if (isStringOnlySchema(schema)) return raw;
	if (raw.trim().length === 0) return raw.trim();
	try {
		return JSON.parse(raw) as unknown;
	} catch {
		return raw;
	}
}

function invalidArguments(raw: string, error: Error): Record<string, unknown> {
	const bounded =
		raw.length <= RAW_BLOCK_LIMIT
			? raw
			: `${raw.slice(0, RAW_BLOCK_LIMIT)}… [truncated ${raw.length - RAW_BLOCK_LIMIT} chars]`;
	return { __parseError: error.message, __rawJson: bounded };
}

export class MiniCPM5Scanner {
	#buffer = "";
	#state: State = "outside";
	#id = "";
	#name = "";
	#openTag = "";
	#started = false;
	#thinking = "";
	readonly #parseThinking: boolean;
	readonly #tools = new Map<string, Record<string, unknown>>();

	constructor(options: MiniCPM5ScannerOptions = {}) {
		this.#parseThinking = options.parseThinking !== false;
		for (const tool of options.tools ?? []) {
			const props = (tool.parameters as { properties?: unknown }).properties;
			this.#tools.set(
				tool.name,
				props && typeof props === "object" && !Array.isArray(props) ? (props as Record<string, unknown>) : {},
			);
		}
	}

	feed(text: string): MiniCPM5ScanEvent[] {
		if (text.length === 0) return [];
		this.#buffer += text;
		return this.#consume(false);
	}

	flush(): MiniCPM5ScanEvent[] {
		return this.#consume(true);
	}

	#consume(final: boolean): MiniCPM5ScanEvent[] {
		const events: MiniCPM5ScanEvent[] = [];
		while (this.#buffer.length > 0) {
			if (this.#state === "outside") {
				this.#consumeOutside(final, events);
				if (this.#state === "outside") break;
			} else if (this.#state === "thinking") {
				this.#consumeThinking(final, events);
				if (this.#state === "thinking") break;
			} else {
				this.#consumeTool(final, events);
				if (this.#state === "tool") break;
			}
		}
		if (final && this.#state === "thinking") this.#endThinking(events);
		if (final && this.#state === "tool" && this.#started) this.#endUnterminatedTool(events);
		return events;
	}

	#consumeOutside(final: boolean, events: MiniCPM5ScanEvent[]): void {
		const tool = this.#buffer.indexOf(FUNCTION_OPEN);
		const think = this.#parseThinking ? this.#buffer.indexOf(THINK_OPEN) : -1;
		let start = tool;
		let isThink = false;
		if (think !== -1 && (start === -1 || think < start)) {
			start = think;
			isThink = true;
		}

		if (start === -1) {
			const tags = this.#parseThinking ? [FUNCTION_OPEN, THINK_OPEN] : [FUNCTION_OPEN];
			const hold = final ? 0 : partialSuffixOverlap(this.#buffer, tags);
			const emit = this.#buffer.slice(0, this.#buffer.length - hold);
			if (emit.length > 0) events.push({ type: "text", text: emit });
			this.#buffer = this.#buffer.slice(this.#buffer.length - hold);
			return;
		}

		if (start > 0) events.push({ type: "text", text: this.#buffer.slice(0, start) });
		this.#buffer = this.#buffer.slice(start);
		if (isThink) {
			this.#buffer = this.#buffer.slice(THINK_OPEN.length);
			this.#state = "thinking";
			this.#thinking = "";
			events.push({ type: "thinkingStart" });
			return;
		}

		this.#state = "tool";
		this.#id = mintToolCallId();
		this.#name = "";
		this.#openTag = "";
		this.#started = false;
	}

	#consumeThinking(final: boolean, events: MiniCPM5ScanEvent[]): void {
		const close = this.#buffer.indexOf(THINK_CLOSE);
		if (close === -1) {
			const hold = final ? 0 : partialSuffixOverlap(this.#buffer, [THINK_CLOSE]);
			this.#emitThinkingDelta(this.#buffer.slice(0, this.#buffer.length - hold), events);
			this.#buffer = this.#buffer.slice(this.#buffer.length - hold);
			if (final) this.#endThinking(events);
			return;
		}

		this.#emitThinkingDelta(this.#buffer.slice(0, close), events);
		this.#buffer = this.#buffer.slice(close + THINK_CLOSE.length);
		this.#endThinking(events);
	}

	#consumeTool(final: boolean, events: MiniCPM5ScanEvent[]): void {
		if (!this.#started) {
			const openEnd = this.#buffer.indexOf(">");
			if (openEnd === -1) {
				if (final) {
					events.push({ type: "text", text: this.#buffer });
					this.#buffer = "";
					this.#resetTool();
				}
				return;
			}

			const openTag = normalizeTagHeader(this.#buffer.slice(0, openEnd + 1));
			const name = FUNCTION_OPEN_RE.exec(openTag)?.[2]?.trim() ?? "";
			if (!name || (this.#tools.size > 0 && !this.#tools.has(name))) {
				events.push({ type: "text", text: openTag });
				this.#buffer = this.#buffer.slice(openEnd + 1);
				this.#resetTool();
				return;
			}

			this.#name = name;
			this.#openTag = openTag;
			this.#started = true;
			this.#buffer = this.#buffer.slice(openEnd + 1);
			events.push({ type: "toolStart", id: this.#id, name });
		}

		const close = this.#buffer.indexOf(FUNCTION_CLOSE);
		if (close === -1) return;

		const body = this.#buffer.slice(0, close);
		const rawBlock = `${this.#openTag}${body}${FUNCTION_CLOSE}`;
		const parsed = this.#parseArguments(body);
		events.push({
			type: "toolEnd",
			id: this.#id,
			name: this.#name,
			arguments: parsed instanceof Error ? invalidArguments(rawBlock, parsed) : parsed,
			rawBlock,
		});
		this.#buffer = this.#buffer.slice(close + FUNCTION_CLOSE.length);
		this.#resetTool();
	}

	/** The stream ended inside a call: report invalid arguments rather than executing a partial preview. */
	#endUnterminatedTool(events: MiniCPM5ScanEvent[]): void {
		const rawBlock = `${this.#openTag}${this.#buffer}`;
		events.push({
			type: "toolEnd",
			id: this.#id,
			name: this.#name,
			arguments: invalidArguments(rawBlock, new Error("Unterminated MiniCPM5 <function> call")),
			rawBlock,
		});
		this.#buffer = "";
		this.#resetTool();
	}

	#parseArguments(body: string): Record<string, unknown> | Error {
		const properties = this.#tools.get(this.#name) ?? {};
		const result: Record<string, unknown> = {};
		const seen = new Set<string>();
		let cursor = 0;

		while (true) {
			const paramStart = body.indexOf("<param", cursor);
			if (paramStart === -1) break;
			const openEnd = body.indexOf(">", paramStart + "<param".length);
			if (openEnd === -1) return new Error("Unterminated MiniCPM5 <param> tag");

			const openTag = normalizeTagHeader(body.slice(paramStart, openEnd + 1));
			const key = PARAM_OPEN_RE.exec(openTag)?.[2]?.trim() ?? "";
			if (!key) return new Error("MiniCPM5 <param> is missing a valid name");
			if (seen.has(key)) return new Error(`Duplicate MiniCPM5 parameter: ${key}`);
			seen.add(key);

			const valueStart = openEnd + 1;
			let closeStart: number;
			let rawValue: string;
			if (body.startsWith(CDATA_OPEN, valueStart)) {
				const cdataEnd = body.indexOf(CDATA_CLOSE, valueStart + CDATA_OPEN.length);
				if (cdataEnd === -1) return new Error(`Unterminated CDATA for MiniCPM5 parameter: ${key}`);
				const afterCdata = cdataEnd + CDATA_CLOSE.length;
				closeStart = body.indexOf(PARAM_CLOSE, afterCdata);
				if (closeStart === -1 || body.slice(afterCdata, closeStart).trim().length > 0) {
					return new Error(`Invalid CDATA framing for MiniCPM5 parameter: ${key}`);
				}
				rawValue = body.slice(valueStart + CDATA_OPEN.length, cdataEnd);
			} else {
				closeStart = body.indexOf(PARAM_CLOSE, valueStart);
				if (closeStart === -1) return new Error(`Unterminated MiniCPM5 parameter: ${key}`);
				rawValue = decodeXmlEntities(body.slice(valueStart, closeStart));
			}

			result[key] = coerceValue(rawValue, properties[key]);
			cursor = closeStart + PARAM_CLOSE.length;
		}

		return result;
	}

	#emitThinkingDelta(delta: string, events: MiniCPM5ScanEvent[]): void {
		if (delta.length === 0) return;
		this.#thinking += delta;
		events.push({ type: "thinkingDelta", delta });
	}

	#endThinking(events: MiniCPM5ScanEvent[]): void {
		events.push({ type: "thinkingEnd", thinking: this.#thinking });
		this.#thinking = "";
		this.#state = "outside";
	}

	#resetTool(): void {
		this.#state = "outside";
		this.#id = "";
		this.#name = "";
		this.#openTag = "";
		this.#started = false;
	}
}

function escapeXmlText(value: string): string {
	return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function escapeXmlAttr(value: string): string {
	return escapeXmlText(value).replaceAll('"', "&quot;");
}

function renderParamValue(value: unknown, isString: boolean): string {
	const raw = isString && typeof value === "string" ? value : JSON.stringify(value);
	if (raw.includes("]]>")) return escapeXmlText(raw);
	if (raw.includes("<") || raw.includes("&") || raw.includes("\n") || raw.includes("\r")) {
		return `${CDATA_OPEN}${raw}${CDATA_CLOSE}`;
	}
	return escapeXmlText(raw);
}

/** Render a tool call in MiniCPM5's native format, for replaying history to the model. */
export function renderMiniCPM5ToolCall(call: ToolCall, tools: readonly Tool[] = []): string {
	const tool = tools.find((t) => t.name === call.name);
	const properties = (tool?.parameters as { properties?: Record<string, unknown> } | undefined)?.properties ?? {};
	let out = `<function name="${escapeXmlAttr(call.name)}">`;
	for (const key in call.arguments) {
		const value = call.arguments[key];
		const isString = tool ? isStringOnlySchema(properties[key]) : typeof value === "string";
		out += `<param name="${escapeXmlAttr(key)}">${renderParamValue(value, isString)}</param>`;
	}
	return `${out}</function>`;
}

/** Render a tool result in MiniCPM5's `<tool_response>` envelope. */
export function renderMiniCPM5ToolResponse(text: string): string {
	return `<tool_response>\n${text}\n</tool_response>`;
}
