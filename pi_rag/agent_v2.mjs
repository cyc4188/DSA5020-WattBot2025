import { Agent } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { createInterface } from "node:readline";

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
const inbox = lines[Symbol.asyncIterator]();
const send = (event) => process.stdout.write(JSON.stringify(event) + "\n");
const receive = async () => JSON.parse((await inbox.next()).value);
const settings = await receive();
settings.maxCalls = settings.maxCalls ?? 8;
settings.thinking = Boolean(settings.thinking);
const model = { id: "DeepSeek-V4.1-Flash", name: "DeepSeek Flash 1017",
  provider: "hkustgz", api: "openai-completions",
  baseUrl: "https://aigc-api.hkust-gz.edu.cn/api/v2",
  input: ["text"], reasoning: Boolean(settings.thinking),
  maxTokens: settings.maxTokens || (settings.thinking ? 6144 : 2400), contextWindow: 1000000,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
const text = (content) => typeof content === "string" ? content
  : content.filter(x => x.type === "text").map(x => x.text).join("\n");
function wireMessage(message) {
  if (message.role === "toolResult") {
    return { role: "tool", tool_call_id: message.toolCallId, content: text(message.content) };
  }
  const item = { role: message.role, content: text(message.content) };
  if (message.role === "assistant") {
    const thinking = Array.isArray(message.content)
      ? message.content.filter(x => x.type === "thinking").map(x => x.thinking).join("\n")
      : "";
    if (settings.thinking && thinking) item.reasoning_content = thinking;
    const calls = Array.isArray(message.content)
      ? message.content.filter(x => x.type === "toolCall") : [];
    if (calls.length) {
      item.tool_calls = calls.map(x => ({ id: x.id, type: "function",
        function: { name: x.name, arguments: JSON.stringify(x.arguments) } }));
    }
  }
  return item;
}

let calls = 0;
function streamFn(_, context) {
  const stream = createAssistantMessageEventStream();
  const assistant = { role: "assistant", content: [], api: model.api,
    provider: model.provider, model: model.id, stopReason: "stop", timestamp: Date.now(),
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: model.cost } };
  (async () => {
    try {
      const attempt = ++calls;
      if (attempt > settings.maxCalls) throw new Error("model turn budget exhausted");
      const messages = context.messages.map(wireMessage);
      const active = attempt === settings.maxCalls ? tools.filter(x => x.name === "submit_answer") : tools;
      const definitions = active.map(({ name, description, parameters }) =>
        ({ type: "function", function: { name, description, parameters } }));
      send({ event: "llm_request", stage: "pi_agent", attempt, model: model.id, messages,
        tools: definitions.map(x => x.function.name) });
      const started = Date.now();
      let response;
      let body;
      for (let retry = 0; retry < 6; retry++) {
        try {
          response = await fetch(model.baseUrl + "/chat/completions", {
            method: "POST", signal: AbortSignal.timeout(settings.thinking ? 150000 : 90000),
            headers: { Authorization: "Bearer " + process.env.WATTBOT_CAMPUS_KEY, "Content-Type": "application/json" },
            body: JSON.stringify({ model: model.id, messages, tools: definitions, temperature: 0,
              max_tokens: model.maxTokens, parallel_tool_calls: settings.parallelToolCalls !== false,
              tool_choice: attempt === settings.maxCalls ? {type:"function",function:{name:"submit_answer"}} : "auto",
              chat_template_kwargs: { enable_thinking: Boolean(settings.thinking) } }),
          });
          if (response.ok) {
            body = await response.json();
            break;
          }
          send({ event: "llm_http_error", status: response.status, attempt, retry });
          if (![408, 429, 500, 502, 503, 504].includes(response.status) || retry === 5) {
            throw new Error("Campus API HTTP " + response.status);
          }
          await response.body?.cancel();
          const retryAfter = Number(response.headers.get("retry-after"));
          const delay = response.headers.has("retry-after") && Number.isFinite(retryAfter)
            ? retryAfter * 1000
            : Math.min(30000, 1500 * 2 ** retry) + Math.random() * 500;
          await new Promise(resolve => setTimeout(resolve, delay));
        } catch (error) {
          if (error.message?.startsWith("Campus API HTTP ")) throw error;
          send({ event: "llm_network_error", message: error.message, attempt, retry });
          if (retry === 5) throw error;
          await new Promise(resolve =>
            setTimeout(resolve, Math.min(30000, 1500 * 2 ** retry) + Math.random() * 500));
        }
      }
      const choice = body.choices?.[0];
      const message = choice?.message;
      send({ event: "llm_response", stage: "pi_agent", model: body.model,
        system_fingerprint: body.system_fingerprint, response_id: body.id,
        usage: body.usage, elapsed_ms: Date.now() - started, content: message?.content,
        reasoning_content: message?.reasoning_content ?? null,
        tool_calls: message?.tool_calls, finish_reason: choice?.finish_reason });
      if (![model.id, "DeepSeek-V4.1-Flash-1017"].includes(body.model) || !message) {
        throw new Error("Unexpected model or missing response");
      }
      if (message.content) assistant.content.push({ type: "text", text: message.content });
      if (typeof message.reasoning_content === "string" && message.reasoning_content.length > 0) {
        assistant.content.push({ type: "thinking", thinking: message.reasoning_content });
      }
      for (const call of message.tool_calls ?? []) {
        let args;
        try { args = JSON.parse(call.function.arguments); } catch { args = {}; }
        assistant.content.push({ type: "toolCall", id: call.id, name: call.function.name, arguments: args });
      }
      assistant.stopReason = choice.finish_reason === "length" ? "length"
        : assistant.content.some(x => x.type === "toolCall") ? "toolUse" : "stop";
      stream.push({type: "start", partial: assistant});
      stream.push({type: "done", reason: assistant.stopReason, message: assistant});
    } catch (error) {
      assistant.stopReason = "error";
      assistant.errorMessage = error.message;
      stream.push({type: "error", reason: "error", error: assistant});
    }
  })();
  return stream;
}

const string = { type: "string" };
const integer = { type: "integer" };
const object = (properties, required = Object.keys(properties)) =>
  ({type: "object", properties, required, additionalProperties: false});
const evidence = object({node_id: string, quote: string});
const tools = [];
let finalized = false;
function register(name, description, parameters) {
  tools.push({ name, label: name, description, parameters, async execute(id, args) {
    send({ event: "tool_request", tool: name, tool_id: id, arguments: args,
      attempt: calls, max_calls: settings.maxCalls });
    const result = await receive();
    if (result.terminate === true) finalized = true;
    return { content: [{ type: "text", text: JSON.stringify(result.data) }],
      details: {}, isError: !result.ok, terminate: result.terminate ?? false };
  } });
}
if (settings.mode === "pi") {
  register("list_documents",
    "List all competition documents with their ref_id, title, and page count.",
    object({}, []));
  register("search_corpus", "Search all competition documents with a focused query.",
    object({ query: string,
      ref_id: {...string, description: "Restrict search to one document."},
      top_k: {...integer, minimum: 1, maximum: 10, default: 6,
        description: "Number of results to return (1-10, default 6)."} }, ["query"]));
  register("read_page",
    "Read a full page (with clean Markdown tables) from any competition document.",
    object({ ref_id: string, page: integer }));
}
const operand = object({ value: string, node_id: string, quote: string, calculation_id: string }, []);
register("eval_math",
  "Evaluate a multi-step arithmetic expression with Python Decimal and return the numeric result. Put corpus numbers in named_operands keyed by a short variable name used inside expression, e.g. expression: '(gpu_hours / num_gpus) / 24', named_operands: { gpu_hours: { value: '125000', node_id: '...', quote: '...' }, num_gpus: { value: '8', node_id: '...', quote: '...' } }. Each operand is either a cited source number (supply node_id and the exact quote) or a prior calculation_id returned by eval_math. Standard conversion and math constants may be used directly in expression without quoting: 0, 0.5, 1, 2, 10, 100, 1000, 1e6, 1e9, 1e12, 60, 3600, 24, 7, 12, 52, 365, 365.25, 8760, 86400, 2.20462, 0.453592, 453.592, 2000, 2204.62, 3.78541, 0.264172, 3.6, 1024, and named constants HOURS_PER_DAY, DAYS_PER_YEAR, MILLION, BILLION, LB_PER_KG, KG_PER_LB, L_PER_GAL (and similar). Supported operations: +, -, *, /, **, parentheses, abs(), min(), max(). Always set result_unit.",
  object({ expression: {...string,
      description: "Multi-step Python Decimal arithmetic expression, e.g. '(gpu_hours / num_gpus) / 24' or 'cost_m * 1000000' or '(old - new) / old * 100'."},
    named_operands: { type: "object", additionalProperties: operand,
      description: "Map of variable names used in expression to cited operands or prior calculations." },
    result_unit: string }, ["expression", "named_operands", "result_unit"]));
register("submit_answer",
  "Submit a final grounded answer or abstain. Always supply answer_value and answer_unit, both is_blank when abstaining. For arithmetic also include a calculation_id returned by eval_math. Cite exact quotes with shown node IDs. In pi mode, do not abstain before searching the corpus.",
  object({ answerable: {type: "boolean",
      description: "True if evidence determines the answer, including when it proves the statement false. False only if evidence is insufficient."},
    answer: string,
    answer_value: {...string, description: "For true/false questions, 1 means true and 0 means false. Use is_blank only when abstaining."},
    answer_unit: {...string, description: "Unit for numeric answers; is_blank if inapplicable or abstaining, including all true/false questions."},
    explanation: string, evidence: {type:"array",items:evidence}, calculation_id: string,
    reason: {...string, description: "When abstaining, explain what evidence is missing."} }, ["answerable", "answer_value", "answer_unit"]));

const agent = new Agent({ initialState: { model, tools, systemPrompt: settings.prompt },
  streamFn, toolExecution: "sequential" });
agent.finishTurn = async ({ message }) =>
  finalized || (calls >= settings.maxCalls && message.stopReason !== "error") ? { action: "end" } : undefined;
await agent.prompt(JSON.stringify(settings.input));
const last = agent.state.messages.filter(m => m.role === "assistant").at(-1);
send({ event: "agent_done", stop_reason: last?.stopReason, error: last?.errorMessage });
lines.close();
