import crypto from 'crypto';
import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { EventEmitter } from 'stream';
import {
  getAvailableChatModelProviders,
  getAvailableEmbeddingModelProviders,
} from '@/lib/providers';
import { and, eq, gt } from 'drizzle-orm';
import { getFileDetails } from '@/lib/utils/files';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import { searchHandlers } from '@/lib/search';
import { createCustomModel, validateCustomModel } from '@/lib/providers/customModels';
import { loadPersistence } from '@/lib/persistence';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Hard-coded timeout limit for API responses (120 seconds)
const API_RESPONSE_TIMEOUT_MS = 120000; // 120 seconds

type Message = {
  messageId: string;
  chatId: string;
  content: string;
};

type ChatModel = {
  provider: string;
  model?: string;
  name?: string;
  apiKey?: string;
  baseUrl?: string;
};

type Body = {
  message: Message;
  optimizationMode: 'speed' | 'balanced' | 'quality';
  focusMode: string;
  history: Array<[string, string]>;
  files: Array<string>;
  chatModel: ChatModel;
  systemInstructions: string;
  maxSources?: number;
  maxToken?: number;
  includeImages?: boolean;
  includeVideos?: boolean;
};

const getRequestedModelKey = (chatModel?: ChatModel) =>
  chatModel?.model || chatModel?.name;

const getStreamErrorMessage = (data: unknown) => {
  if (typeof data === 'string') {
    try {
      const parsed = JSON.parse(data);
      if (typeof parsed?.data === 'string') {
        return parsed.data;
      }
      if (typeof parsed?.message === 'string') {
        return parsed.message;
      }
    } catch {
      return data;
    }
  }

  if (data instanceof Error) {
    return data.message;
  }

  if (
    typeof data === 'object' &&
    data !== null &&
    'message' in data &&
    typeof (data as { message?: unknown }).message === 'string'
  ) {
    return (data as { message: string }).message;
  }

  return 'An error occurred while processing the chat request';
};

const handleEmitterEvents = async (
  stream: EventEmitter,
  writer: WritableStreamDefaultWriter,
  encoder: TextEncoder,
  aiMessageId: string,
  chatId: string,
) => {
  let recievedMessage = '';
  let sources: any[] = [];
  let writerClosed = false;
  let timeoutId: NodeJS.Timeout;

  // Set up timeout to force close after 120 seconds
  timeoutId = setTimeout(() => {
    if (!writerClosed) {
      console.log(`[Chat Timeout] Forcing stream closure after ${API_RESPONSE_TIMEOUT_MS / 1000}s`);

      // Send timeout notification
      try {
        writer.write(
          encoder.encode(
            JSON.stringify({
              type: 'timeout',
              data: 'Response timeout reached - stopping generation',
              messageId: aiMessageId,
              partial: true,
              timeout: API_RESPONSE_TIMEOUT_MS,
            }) + '\n',
          ),
        ).catch(() => { });

        writer.write(
          encoder.encode(
            JSON.stringify({
              type: 'messageEnd',
              messageId: aiMessageId,
              partial: true,
            }) + '\n',
          ),
        ).then(() => {
          writer.close().catch(() => { });
        }).catch(() => { });
      } catch (error) {
        // Writer already closed
      }

      writerClosed = true;
      stream.removeAllListeners();

      // Save partial message to database
      if (recievedMessage) {
        void (async () => {
          const persistence = await loadPersistence();
          if (!persistence) return;

          persistence.db
            .insert(persistence.messages)
            .values({
              content: recievedMessage + '\n\n[Response truncated due to timeout]',
              chatId: chatId,
              messageId: aiMessageId,
              role: 'assistant',
              metadata: JSON.stringify({
                createdAt: new Date(),
                partial: true,
                timeout: true,
                ...(sources && sources.length > 0 && { sources }),
              }),
            })
            .execute();
        })();
      }
    }
  }, API_RESPONSE_TIMEOUT_MS);

  stream.on('data', (data) => {
    if (writerClosed) return;

    try {
      const parsedData = JSON.parse(data);
      if (parsedData.type === 'response') {
        writer.write(
          encoder.encode(
            JSON.stringify({
              type: 'message',
              data: parsedData.data,
              messageId: aiMessageId,
            }) + '\n',
          ),
        ).catch(() => {
          // Writer already closed by client disconnect
          writerClosed = true;
        });

        recievedMessage += parsedData.data;
      } else if (parsedData.type === 'sources') {
        writer.write(
          encoder.encode(
            JSON.stringify({
              type: 'sources',
              data: parsedData.data,
              messageId: aiMessageId,
            }) + '\n',
          ),
        ).catch(() => {
          // Writer already closed by client disconnect
          writerClosed = true;
        });

        sources = parsedData.data;
      }
    } catch (error) {
      console.error('Error processing stream data:', error);
    }
  });

  stream.on('end', () => {
    clearTimeout(timeoutId); // Clear timeout if stream ends normally

    if (!writerClosed) {
      try {
        writer.write(
          encoder.encode(
            JSON.stringify({
              type: 'messageEnd',
              messageId: aiMessageId,
            }) + '\n',
          ),
        ).then(() => {
          if (!writerClosed) {
            writer.close().catch(() => {
              // Already closed
            });
            writerClosed = true;
          }
        }).catch(() => {
          // Writer already closed
          writerClosed = true;
        });
      } catch (error) {
        // Writer already closed
        writerClosed = true;
      }
    }

    // Only save to DB if not already saved by timeout
    if (!writerClosed || recievedMessage) {
      void (async () => {
        const persistence = await loadPersistence();
        if (!persistence) return;

        persistence.db
          .insert(persistence.messages)
          .values({
            content: recievedMessage,
            chatId: chatId,
            messageId: aiMessageId,
            role: 'assistant',
            metadata: JSON.stringify({
              createdAt: new Date(),
              ...(sources && sources.length > 0 && { sources }),
            }),
          })
          .execute();
      })();
    }
  });
  stream.on('error', (data) => {
    clearTimeout(timeoutId); // Clear timeout on error

    if (!writerClosed) {
      try {
        const errorMessage = getStreamErrorMessage(data);
        writer.write(
          encoder.encode(
            JSON.stringify({
              type: 'error',
              data: errorMessage,
            }) + '\n',
          ),
        ).then(() => {
          if (!writerClosed) {
            writer.close().catch(() => {
              // Already closed
            });
            writerClosed = true;
          }
        }).catch(() => {
          // Writer already closed
          writerClosed = true;
        });
      } catch (error) {
        console.error('Error handling stream error:', error);
        writerClosed = true;
      }
    }
  });
};

const handleHistorySave = async (
  message: Message,
  humanMessageId: string,
  focusMode: string,
  files: string[],
) => {
  const persistence = await loadPersistence();
  if (!persistence) return;

  const chat = await persistence.db.query.chats.findFirst({
    where: eq(persistence.chats.id, message.chatId),
  });

  if (!chat) {
    await persistence.db
      .insert(persistence.chats)
      .values({
        id: message.chatId,
        title: message.content,
        createdAt: new Date().toString(),
        focusMode: focusMode,
        files: files.map(getFileDetails),
      })
      .execute();
  }

  const messageExists = await persistence.db.query.messages.findFirst({
    where: eq(persistence.messages.messageId, humanMessageId),
  });

  if (!messageExists) {
    await persistence.db
      .insert(persistence.messages)
      .values({
        content: message.content,
        chatId: message.chatId,
        messageId: humanMessageId,
        role: 'user',
        metadata: JSON.stringify({
          createdAt: new Date(),
        }),
      })
      .execute();
  } else {
    await persistence.db
      .delete(persistence.messages)
      .where(
        and(
          gt(persistence.messages.id, messageExists.id),
          eq(persistence.messages.chatId, message.chatId),
        ),
      )
      .execute();
  }
};

export const POST = async (req: Request) => {
  try {
    const { verifyApiToken } = await import('@/lib/authMiddleware');
    const auth = verifyApiToken(req);
    if (!auth.authorized) return auth.response!;

    const body = (await req.json()) as Body;
    const { message } = body;

    if (message.content === '') {
      return Response.json(
        {
          message: 'Please provide a message to process',
        },
        { status: 400 },
      );
    }

    const chatModelProviders = await getAvailableChatModelProviders();

    let llm: BaseChatModel | undefined;

    // Check if custom model configuration is provided
    if (body.chatModel?.apiKey && getRequestedModelKey(body.chatModel)) {
      const customConfig = {
        provider: body.chatModel.provider,
        model: getRequestedModelKey(body.chatModel) || '',
        apiKey: body.chatModel.apiKey,
        baseUrl: body.chatModel.baseUrl,
      };

      const validation = validateCustomModel(customConfig);
      if (!validation.isValid) {
        return Response.json({ error: validation.error }, { status: 400 });
      }

      llm = createCustomModel(customConfig);
    } else {
      // Use default configured models
      const chatModelProvider =
        chatModelProviders[
        body.chatModel?.provider || Object.keys(chatModelProviders)[0]
        ];
      const chatModel =
        chatModelProvider?.[
        getRequestedModelKey(body.chatModel) ||
        Object.keys(chatModelProvider || {})[0]
        ];

      if (chatModelProvider && chatModel) {
        llm = chatModel.model;
      }
    }

    if (!llm) {
      return Response.json({ error: 'Invalid chat model configuration' }, { status: 400 });
    }

    const humanMessageId =
      message.messageId ?? crypto.randomBytes(7).toString('hex');
    const aiMessageId = crypto.randomBytes(7).toString('hex');

    const history: BaseMessage[] = body.history.map((msg) => {
      if (msg[0] === 'human') {
        return new HumanMessage({
          content: msg[1],
        });
      } else {
        return new AIMessage({
          content: msg[1],
        });
      }
    });

    const handler = searchHandlers[body.focusMode];

    if (!handler) {
      return Response.json(
        {
          message: 'Invalid focus mode',
        },
        { status: 400 },
      );
    }

    // Get system-configured embedding model for reranking
    let embeddings: Embeddings | null = null;
    if (body.optimizationMode === 'balanced') {
      const embeddingProviders = await getAvailableEmbeddingModelProviders();

      // Try to get the first available embedding model from system configuration
      for (const provider of Object.keys(embeddingProviders)) {
        const models = embeddingProviders[provider];
        if (models && Object.keys(models).length > 0) {
          const firstModel = Object.keys(models)[0];
          embeddings = models[firstModel].model;
          break;
        }
      }
    }

    const stream = await handler.searchAndAnswer(
      message.content,
      history,
      llm,
      embeddings,
      body.optimizationMode,
      body.files,
      body.systemInstructions || '',
      body.maxSources,
      body.maxToken || 4000,
      body.includeImages,
      body.includeVideos,
    );

    const responseStream = new TransformStream();
    const writer = responseStream.writable.getWriter();
    const encoder = new TextEncoder();

    handleEmitterEvents(stream, writer, encoder, aiMessageId, message.chatId);
    handleHistorySave(message, humanMessageId, body.focusMode, body.files);

    return new Response(responseStream.readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        Connection: 'keep-alive',
        'Cache-Control': 'no-cache, no-transform',
      },
    });
  } catch (err) {
    console.error('An error occurred while processing chat request:', err);
    return Response.json(
      { message: 'An error occurred while processing chat request' },
      { status: 500 },
    );
  }
};
