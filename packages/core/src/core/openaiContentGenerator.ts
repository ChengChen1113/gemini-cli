import { fetch } from 'undici';
import {
  Content,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

interface OpenAIConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
}

export class OpenAIContentGenerator implements ContentGenerator {
  constructor(private readonly config: OpenAIConfig) {}

  private contentToMessages(contents: Content[]): { role: string; content: string }[] {
    return contents.flatMap((c) => {
      const text = (c.parts || []).map((p) => p.text).filter(Boolean).join('');
      if (!text) return [];
      const role = c.role === 'model' ? 'assistant' : 'user';
      return [{ role, content: text }];
    });
  }

  private toResponse(text: string): GenerateContentResponse {
    return {
      candidates: [
        {
          content: { role: 'model', parts: [{ text }] },
          index: 0,
          finishReason: 'stop',
          safetyRatings: [],
        },
      ],
      promptFeedback: { safetyRatings: [] },
    } as GenerateContentResponse;
  }

  async generateContent(request: GenerateContentParameters): Promise<GenerateContentResponse> {
    const messages = this.contentToMessages(request.contents || []);
    const res = await fetch(`${this.config.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({ model: this.config.model, messages }),
    });
    const data = await res.json();
    const text = data.choices?.[0]?.message?.content || '';
    return this.toResponse(text);
  }

  async *generateContentStream(
    request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    const messages = this.contentToMessages(request.contents || []);
    const res = await fetch(`${this.config.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({ model: this.config.model, messages, stream: true }),
    });

    const reader = res.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice('data:'.length).trim();
        if (payload === '[DONE]') continue;
        const json = JSON.parse(payload);
        const delta = json.choices?.[0]?.delta?.content;
        if (delta) {
          yield this.toResponse(delta);
        }
      }
    }
  }

  async countTokens(_req: CountTokensParameters): Promise<CountTokensResponse> {
    return { totalTokens: 0 } as CountTokensResponse;
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const res = await fetch(`${this.config.baseUrl}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({ model: this.config.model, input: request.content }),
    });
    return (await res.json()) as EmbedContentResponse;
  }
}
