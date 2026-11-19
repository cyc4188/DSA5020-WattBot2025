import { Agent } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { createInterface } from "node:readline";

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
const inbox = lines[Symbol.asyncIterator]();
const send = (event) => process.stdout.write(JSON.stringify(event) + "\n");
const receive = async () => JSON.parse((await inbox.next()).value);
const settings = await receive();
const model = { id: "DeepSeek-V4.1-Flash", name: "DeepSeek Flash 1017",
  provider: "hkustgz", api: "openai-completions",
  baseUrl: "https://aigc-api.hkust-gz.edu.cn/api/v2",
  input: ["text"], reasoning: false, maxTokens: 2400, contextWindow: 1000000,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
const text = (content) => typeof content === "string" ? content
  : content.filter(x => x.type === "text").map(x => x.text).join("\n");
function wireMessage(message) {
  if (message.role === "toolResult") {
    return { role: "tool", tool_call_id: message.toolCallId, content: text(message.content) };
  }
  const item = { role: message.role, content: text(message.content) };
  if (message.role === "assistant") {
    const calls = message.content.filter(x => x.type === "toolCall");
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
      for (let retry = 0; retry < 3; retry++) {
        response = await fetch(model.baseUrl + "/chat/completions", {
          method: "POST", signal: AbortSignal.timeout(90000),
          headers: { Authorization: "Bearer " + process.env.WATTBOT_CAMPUS_KEY, "Content-Type": "application/json" },
          body: JSON.stringify({ model: model.id, messages, tools: definitions, temperature: 0,
            max_tokens: 2400, parallel_tool_calls: false,
            tool_choice: attempt === settings.maxCalls ? {type:"function",function:{name:"submit_answer"}} : "auto",
            chat_template_kwargs: { enable_thinking: false } }),
        });
        if (response.ok) break;
        send({ event: "llm_http_error", status: response.status, attempt, retry });
        if (![408, 429, 500, 502, 503, 504].includes(response.status) || retry === 2) {
          throw new Error("Campus API HTTP " + response.status);
        }
        await response.body?.cancel();
        await new Promise(resolve => setTimeout(resolve, 1000 * (retry + 1)));
      }
      const body = await response.json();
      const choice = body.choices?.[0];
      const message = choice?.message;
      send({ event: "llm_response", stage: "pi_agent", model: body.model,
        system_fingerprint: body.system_fingerprint, response_id: body.id,
        usage: body.usage, elapsed_ms: Date.now() - started, content: message?.content,
        tool_calls: message?.tool_calls, finish_reason: choice?.finish_reason });
      if (![model.id, "DeepSeek-V4.1-Flash-1017"].includes(body.model) || !message) {
        throw new Error("Unexpected model or missing response");
      }
      if (message.content) assistant.content.push({ type: "text", text: message.content });
      for (const call of message.tool_calls ?? []) {
        let args;
        try { args = JSON.parse(call.function.arguments); } catch { args = {}; }
        assistant.content.push({ type: "toolCall", id: call.id, name: call.function.name, arguments: args });
      }
      assistant.stopReason = assistant.content.some(x => x.type === "toolCall") ? "toolUse"
        : choice.finish_reason === "length" ? "length" : "stop";
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
function register(name, description, parameters) {
  tools.push({ name, label: name, description, parameters, async execute(id, args) {
    send({ event: "tool_request", tool: name, tool_id: id, arguments: args });
    const result = await receive();
    return { content: [{ type: "text", text: JSON.stringify(result.data) }],
      details: {}, isError: !result.ok, terminate: result.terminate ?? false };
  } });
}
if (settings.mode === "pi") {
  register("search_corpus", "Search all competition documents with a focused query.",
    object({query: string}));
  register("read_page", "Read a page and its extracted table rows in a discovered document.",
    object({ref_id: string, page: integer}));
}
register("calculate", "Compute using cited operands. Operations: sum, subtract, multiply, divide, percent_change (old,new; signed), percent_decrease (old,new; positive magnitude), percent_of (part,whole), convert. For convert include from_unit and to_unit. For other non-percentage operations supply result_unit. An operand can be a cited source number or a prior calculation_id. Units supported for energy: Wh,kWh,MWh,GWh; time: seconds,hours,days; mass: grams,kg,tonnes; magnitude: thousand,million,billion,one.",
  object({ operation: string, operands: {type:"array", items: object({value: string, node_id: string, quote: string, calculation_id: string}, [])},
    from_unit: string, to_unit: string, result_unit: string }, ["operation", "operands"]));
register("submit_answer", "Submit a final grounded answer or abstain. Always supply answer_value and answer_unit, both is_blank when abstaining. For arithmetic also include a calculation_id returned by calculate. Cite exact quotes with shown node IDs.",
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
  calls >= settings.maxCalls && message.stopReason !== "error" ? { action: "end" } : undefined;
await agent.prompt(JSON.stringify(settings.input));
const last = agent.state.messages.filter(m => m.role === "assistant").at(-1);
send({ event: "agent_done", stop_reason: last?.stopReason, error: last?.errorMessage });
lines.close();
