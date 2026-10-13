// Streams a chat completion from the server and hands back each text delta.

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export async function streamChat(
  messages: Message[],
  maxTokens: number,
  temperature: number,
  onDelta: (text: string) => void,
  signal: AbortSignal,
): Promise<string> {
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "tinyserve", messages, max_tokens: maxTokens, temperature, stream: true }),
    signal,
  });
  if (!response.ok || !response.body) throw new Error(`${response.status} ${await response.text()}`);

  // Server-Sent Events: "data: {json}\n\n" per chunk, then "data: [DONE]".
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let finishReason = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? ""; // keep a partial event for the next read
    for (const event of events) {
      const data = event.replace(/^data: /, "");
      if (data === "[DONE]") return finishReason;
      const chunk = JSON.parse(data);
      const choice = chunk.choices?.[0];
      if (choice?.delta?.content) onDelta(choice.delta.content);
      if (choice?.finish_reason) finishReason = choice.finish_reason;
    }
  }
  return finishReason;
}
