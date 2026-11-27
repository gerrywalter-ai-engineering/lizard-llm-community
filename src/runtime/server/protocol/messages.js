import { ProtocolHttpError } from "./errors.js";

function invalid(param, message, code = "invalid_parameter") {
  throw new ProtocolHttpError(400, code, message, { param });
}

function normalizeContent(content, param) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content) || content.length === 0) {
    invalid(param, "Message content must be a string or a non-empty array of text parts.");
  }
  return content.map((part, index) => {
    if (!part || typeof part !== "object" || Array.isArray(part)) {
      invalid(`${param}.${index}`, "Content parts must be objects.");
    }
    if (part.type !== "text") {
      invalid(`${param}.${index}.type`, `Content type '${part.type ?? "unknown"}' is not supported.`, "unsupported_content_type");
    }
    if (typeof part.text !== "string") {
      invalid(`${param}.${index}.text`, "Text content parts require a string text field.");
    }
    const unsupported = Object.keys(part).find((field) => !["type", "text"].includes(field));
    if (unsupported) invalid(`${param}.${index}.${unsupported}`, `${unsupported} is not supported.`, "unsupported_parameter");
    return part.text;
  }).join("");
}

export function normalizeChatMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0) {
    invalid("messages", "messages must be a non-empty array.");
  }
  const normalized = messages.map((message, index) => {
    const param = `messages.${index}`;
    if (!message || typeof message !== "object" || Array.isArray(message)) {
      invalid(param, "Each message must be an object.");
    }
    if (!["system", "user", "assistant"].includes(message.role)) {
      invalid(`${param}.role`, `Message role '${message.role ?? "unknown"}' is not supported.`, "unsupported_message_role");
    }
    const unsupported = Object.keys(message).find((field) => !["role", "content"].includes(field));
    if (unsupported) invalid(`${param}.${unsupported}`, `${unsupported} is not supported.`, "unsupported_parameter");
    const content = normalizeContent(message.content, `${param}.content`);
    if (!content.trim()) invalid(`${param}.content`, "Message content must not be empty.");
    return { role: message.role, content };
  });
  const lastConversationMessage = [...normalized].reverse().find((message) => message.role !== "system");
  if (lastConversationMessage?.role !== "user") {
    invalid("messages", "The final non-system message must have role 'user'.");
  }
  return normalized;
}

