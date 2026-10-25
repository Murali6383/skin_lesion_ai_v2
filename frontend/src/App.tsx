import React, {
  useEffect,
  useRef,
  useState,
} from "react";

const API_URL = "http://127.0.0.1:8001";
const STORAGE_KEY = "skinguardian_chats";

const STREAM_UPDATE_INTERVAL = 40;
const VOICE_SILENCE_DELAY = 1500;

// =====================================================
// TYPESCRIPT VOICE RECOGNITION TYPES
// =====================================================

type VoiceRecognitionResult = {
  isFinal: boolean;
  [index: number]: {
    transcript: string;
    confidence: number;
  };
};

type VoiceRecognitionEvent = {
  resultIndex: number;
  results: ArrayLike<VoiceRecognitionResult>;
};

type VoiceRecognitionErrorEvent = {
  error: string;
  message?: string;
};

type VoiceRecognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;

  onstart: (() => void) | null;
  onresult:
    | ((event: VoiceRecognitionEvent) => void)
    | null;
  onerror:
    | ((event: VoiceRecognitionErrorEvent) => void)
    | null;
  onend: (() => void) | null;

  start: () => void;
  stop: () => void;
  abort: () => void;
};

type VoiceRecognitionConstructor = new () => VoiceRecognition;

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  image?: string | null;
  prediction?: {
    disease: string;
    confidence: number | null;
    isNormal: boolean;
    gradcamUrl: string | null;
  } | null;
  note?: string | null;
  error?: boolean;
  streaming?: boolean;
  validation?: unknown;
  normalSkin?: unknown;
  createdAt: string;
};

type Chat = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
};

// =====================================================
// APP
// =====================================================

function App() {
  // =====================================================
  // STATE
  // =====================================================

  const [chats, setChats] = useState<Chat[]>(() => {
    try {
      const saved =
        localStorage.getItem(STORAGE_KEY);

      if (!saved) {
        return [];
      }

      const parsed = JSON.parse(saved);

      return Array.isArray(parsed)
        ? parsed
        : [];
    } catch (error) {
      console.warn(
        "Unable to load saved chats:",
        error
      );

      return [];
    }
  });

  const [activeChatId, setActiveChatId] =
    useState<string | null>(null);

  const [prompt, setPrompt] = useState("");

  const [image, setImage] = useState<File | null>(
    null
  );

  const [preview, setPreview] = useState<
    string | null
  >(null);

  const [loading, setLoading] = useState(false);

  // =====================================================
  // NEW: VOICE TO TEXT STATE
  // =====================================================

  const [voiceListening, setVoiceListening] =
    useState(false);

  const [voiceError, setVoiceError] =
    useState("");

  const fileInputRef =
    useRef<HTMLInputElement | null>(null);

  const textareaRef =
    useRef<HTMLTextAreaElement | null>(null);

  const bottomRef =
    useRef<HTMLDivElement | null>(null);

  // =====================================================
  // NEW: VOICE RECOGNITION REFERENCES
  // =====================================================

  const recognitionRef =
    useRef<VoiceRecognition | null>(null);

  const voiceTranscriptRef = useRef("");

  const voiceSilenceTimerRef =
    useRef<ReturnType<typeof setTimeout> | null>(
      null
    );

  const voiceShouldSendRef = useRef(false);

  const voiceFinalTranscriptRef = useRef("");

  // =====================================================
  // STREAMING REFERENCES
  // =====================================================

  const streamBufferRef = useRef("");

  const streamFlushTimerRef =
    useRef<ReturnType<typeof setTimeout> | null>(
      null
    );

  const streamAssistantIdRef =
    useRef<string | null>(null);

  const streamChatIdRef =
    useRef<string | null>(null);

  // =====================================================
  // SAVE CHATS SAFELY
  // =====================================================

  useEffect(() => {
    try {
      const lightweightChats =
        chats.map((chat) => ({
          id: chat.id,
          title: chat.title,
          createdAt: chat.createdAt,
          updatedAt: chat.updatedAt,

          messages: (chat.messages || []).map(
            (message) => ({
              id: message.id,
              role: message.role,
              text: message.text || "",

              prediction:
                message.prediction || null,

              note: message.note || null,

              error: message.error || false,

              streaming: false,

              createdAt: message.createdAt,

              image: message.image || null,
            })
          ),
        }));

      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(lightweightChats)
      );
    } catch (error) {
      console.warn(
        "Chat history could not be saved:",
        error
      );

      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // Ignore storage cleanup failure
      }
    }
  }, [chats]);

  // =====================================================
  // AUTO SCROLL
  // =====================================================

  useEffect(() => {
    if (!bottomRef.current) {
      return;
    }

    bottomRef.current.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [chats, loading]);

  // =====================================================
  // CLEANUP STREAM AND VOICE TIMERS
  // =====================================================

  useEffect(() => {
    return () => {
      if (streamFlushTimerRef.current) {
        clearTimeout(
          streamFlushTimerRef.current
        );
      }

      if (voiceSilenceTimerRef.current) {
        clearTimeout(
          voiceSilenceTimerRef.current
        );
      }

      if (recognitionRef.current) {
        voiceShouldSendRef.current = false;

        try {
          recognitionRef.current.abort();
        } catch {
          // Ignore cleanup errors
        }

        recognitionRef.current = null;
      }
    };
  }, []);

  // =====================================================
  // ACTIVE CHAT
  // =====================================================

  const activeChat = chats.find(
    (chat) => chat.id === activeChatId
  );

  // =====================================================
  // AUTO RESIZE TEXTAREA
  // =====================================================

  const autoResizeTextarea = (
    element: HTMLTextAreaElement | null
  ) => {
    if (!element) {
      return;
    }

    element.style.height = "auto";

    const newHeight = Math.min(
      element.scrollHeight,
      180
    );

    element.style.height = `${newHeight}px`;
  };

  const handlePromptChange = (
    event: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setPrompt(event.target.value);

    autoResizeTextarea(event.target);
  };

  const resetTextarea = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  // =====================================================
  // NEW: CLEAR VOICE SILENCE TIMER
  // =====================================================

  const clearVoiceSilenceTimer = () => {
    if (voiceSilenceTimerRef.current) {
      clearTimeout(
        voiceSilenceTimerRef.current
      );

      voiceSilenceTimerRef.current = null;
    }
  };

  // =====================================================
  // NEW: VOICE TO TEXT + AUTOMATIC SEND
  // =====================================================

  const handleVoiceInput = () => {
    if (loading) {
      return;
    }

    const browserWindow = window as Window & {
      SpeechRecognition?: VoiceRecognitionConstructor;
      webkitSpeechRecognition?: VoiceRecognitionConstructor;
    };

    const SpeechRecognitionAPI =
      browserWindow.SpeechRecognition ||
      browserWindow.webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      const message =
        "Voice recognition is not supported in this browser. Please use Google Chrome.";

      setVoiceError(message);

      alert(message);

      return;
    }

    // -----------------------------------------------
    // STOP CURRENT VOICE SESSION
    // -----------------------------------------------

    if (recognitionRef.current) {
      clearVoiceSilenceTimer();

      // Stop should send the final recognized text.
      voiceShouldSendRef.current = true;

      try {
        recognitionRef.current.stop();
      } catch (error) {
        console.warn(
          "Unable to stop voice recognition:",
          error
        );
      }

      return;
    }

    // -----------------------------------------------
    // START A NEW VOICE SESSION
    // -----------------------------------------------

    setVoiceError("");

    voiceTranscriptRef.current = "";
    voiceFinalTranscriptRef.current = "";

    voiceShouldSendRef.current = true;

    const recognition = new SpeechRecognitionAPI();

    recognition.lang = "en-IN";

    recognition.continuous = true;

    recognition.interimResults = true;

    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setVoiceListening(true);
      setVoiceError("");

      console.log("Voice recognition started.");
    };

    recognition.onresult = (
      event: VoiceRecognitionEvent
    ) => {
      let finalTranscript = "";
      let interimTranscript = "";

      for (
        let i = 0;
        i < event.results.length;
        i++
      ) {
        const result = event.results[i];

        const resultText =
          result[0]?.transcript || "";

        if (result.isFinal) {
          finalTranscript +=
            resultText + " ";
        } else {
          interimTranscript +=
            resultText + " ";
        }
      }

      // ---------------------------------------------
      // PRESERVE FINAL SPEECH RECOGNITION RESULTS
      // ---------------------------------------------

      const previousFinal =
        voiceFinalTranscriptRef.current.trim();

      const currentFinal =
        finalTranscript.trim();

      if (currentFinal) {
        voiceFinalTranscriptRef.current =
          currentFinal;
      }

      const finalText =
        voiceFinalTranscriptRef.current.trim();

      const displayedTranscript = [
        finalText,
        interimTranscript.trim(),
      ]
        .filter(Boolean)
        .join(" ")
        .trim();

      voiceTranscriptRef.current =
        displayedTranscript;

      // ---------------------------------------------
      // UPDATE TEXTAREA
      // ---------------------------------------------

      setPrompt(displayedTranscript);

      if (textareaRef.current) {
        textareaRef.current.value =
          displayedTranscript;

        autoResizeTextarea(textareaRef.current);
      }

      // ---------------------------------------------
      // AUTO SEND AFTER SPEECH PAUSE
      // ---------------------------------------------

      clearVoiceSilenceTimer();

      if (displayedTranscript.trim()) {
        voiceSilenceTimerRef.current =
          setTimeout(() => {
            if (
              recognitionRef.current ===
              recognition
            ) {
              voiceShouldSendRef.current = true;

              try {
                recognition.stop();
              } catch (error) {
                console.warn(
                  "Voice recognition stop failed:",
                  error
                );
              }
            }
          }, VOICE_SILENCE_DELAY);
      }
    };

    // -----------------------------------------------
    // VOICE RECOGNITION ERROR
    // -----------------------------------------------

    recognition.onerror = (
      event: VoiceRecognitionErrorEvent
    ) => {
      console.error(
        "Voice recognition error:",
        event.error,
        event.message
      );

      if (
        event.error === "not-allowed" ||
        event.error === "service-not-allowed"
      ) {
        voiceShouldSendRef.current = false;

        setVoiceError(
          "Microphone permission denied. Allow microphone access in your browser settings."
        );
      } else if (event.error === "no-speech") {
        // Do not send an empty voice message.
        if (
          !voiceFinalTranscriptRef.current.trim()
        ) {
          voiceShouldSendRef.current = false;
        }
      } else if (event.error === "audio-capture") {
        voiceShouldSendRef.current = false;

        setVoiceError(
          "No microphone was detected. Please check your microphone."
        );
      } else if (event.error === "network") {
        setVoiceError(
          "Voice recognition network error. Please try again."
        );
      } else if (event.error === "aborted") {
        // Aborted sessions should not auto-send.
        voiceShouldSendRef.current = false;
      }
    };

    // -----------------------------------------------
    // VOICE RECOGNITION FINISHED
    // -----------------------------------------------

    recognition.onend = () => {
      clearVoiceSilenceTimer();

      if (
        recognitionRef.current === recognition
      ) {
        recognitionRef.current = null;
      }

      setVoiceListening(false);

      const finalTranscript =
        voiceFinalTranscriptRef.current.trim();

      const transcript =
        (
          finalTranscript ||
          voiceTranscriptRef.current
        ).trim();

      const shouldSend =
        voiceShouldSendRef.current;

      voiceShouldSendRef.current = false;

      console.log(
        "Voice recognition completed:",
        transcript
      );

      // ---------------------------------------------
      // AUTOMATICALLY SEND RECOGNIZED TEXT
      // ---------------------------------------------

      if (shouldSend && transcript) {
        setPrompt(transcript);

        // Pass the transcript directly.
        // This avoids relying on asynchronous React state.
        void sendMessage(transcript);
      }
    };

    recognitionRef.current = recognition;

    try {
      recognition.start();
    } catch (error) {
      console.error(
        "Unable to start voice recognition:",
        error
      );

      recognitionRef.current = null;

      voiceShouldSendRef.current = false;

      setVoiceListening(false);

      setVoiceError(
        "Unable to start voice recognition. Please try again."
      );
    }
  };

  // =====================================================
  // NEW: STOP VOICE INPUT
  // =====================================================

  const stopVoiceInput = () => {
    clearVoiceSilenceTimer();

    if (!recognitionRef.current) {
      setVoiceListening(false);
      return;
    }

    voiceShouldSendRef.current = true;

    try {
      recognitionRef.current.stop();
    } catch (error) {
      console.warn(
        "Unable to stop voice input:",
        error
      );
    }
  };

  // =====================================================
  // NEW CHAT
  // =====================================================

  const createNewChat = () => {
    if (loading) {
      return;
    }

    if (recognitionRef.current) {
      voiceShouldSendRef.current = false;

      clearVoiceSilenceTimer();

      try {
        recognitionRef.current.abort();
      } catch {
        // Ignore cleanup failure
      }

      recognitionRef.current = null;
    }

    setVoiceListening(false);
    setVoiceError("");

    setActiveChatId(null);

    setPrompt("");

    setImage(null);

    setPreview(null);

    resetTextarea();

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // =====================================================
  // OPEN CHAT
  // =====================================================

  const openChat = (chatId: string) => {
    if (loading) {
      return;
    }

    if (recognitionRef.current) {
      voiceShouldSendRef.current = false;

      clearVoiceSilenceTimer();

      try {
        recognitionRef.current.abort();
      } catch {
        // Ignore cleanup failure
      }

      recognitionRef.current = null;
    }

    setVoiceListening(false);
    setVoiceError("");

    setActiveChatId(chatId);

    setPrompt("");

    setImage(null);

    setPreview(null);

    resetTextarea();

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // =====================================================
  // DELETE CHAT
  // =====================================================

  const deleteChat = (
    chatId: string,
    event: React.MouseEvent<HTMLButtonElement>
  ) => {
    event.stopPropagation();

    if (loading) {
      return;
    }

    setChats((oldChats) =>
      oldChats.filter(
        (chat) => chat.id !== chatId
      )
    );

    if (activeChatId === chatId) {
      setActiveChatId(null);

      setPrompt("");

      setImage(null);

      setPreview(null);
    }
  };

  // =====================================================
  // FILE TO DATA URL
  // =====================================================

  const fileToDataURL = (
    file: File
  ): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => {
        resolve(String(reader.result));
      };

      reader.onerror = reject;

      reader.readAsDataURL(file);
    });
  };

  // =====================================================
  // DATA URL TO FILE
  // =====================================================

  const dataURLToFile = async (
    dataURL: string,
    filename = "skin_lesion.jpg"
  ): Promise<File | null> => {
    try {
      const response = await fetch(dataURL);

      const blob = await response.blob();

      return new File(
        [blob],
        filename,
        {
          type: blob.type || "image/jpeg",
        }
      );
    } catch (error) {
      console.error(
        "Image restore error:",
        error
      );

      return null;
    }
  };

  // =====================================================
  // IMAGE UPLOAD
  // =====================================================

  const handleImage = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      alert("Please select an image.");

      return;
    }

    try {
      const dataURL = await fileToDataURL(file);

      setImage(file);

      setPreview(dataURL);
    } catch (error) {
      console.error(
        "Image processing error:",
        error
      );

      alert("Unable to process image.");
    }
  };

  // =====================================================
  // REMOVE IMAGE
  // =====================================================

  const removeImage = () => {
    if (loading) {
      return;
    }

    setImage(null);

    setPreview(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // =====================================================
  // CHAT TITLE
  // =====================================================

  const generateTitle = (text: string) => {
    const cleanText = text.trim();

    if (!cleanText) {
      return "Skin Lesion Analysis";
    }

    return cleanText.length > 35
      ? cleanText.substring(0, 35) + "..."
      : cleanText;
  };

  // =====================================================
  // GET IMAGE FOR CHAT
  // =====================================================

  const getImageForChat =
    async (): Promise<File | null> => {
      if (image) {
        return image;
      }

      if (preview) {
        return await dataURLToFile(
          preview,
          "skin_lesion.jpg"
        );
      }

      return null;
    };

  // =====================================================
  // FLUSH STREAM BUFFER
  // =====================================================

  const flushStreamBuffer = () => {
    const chatId = streamChatIdRef.current;

    const assistantId =
      streamAssistantIdRef.current;

    const textToAdd = streamBufferRef.current;

    if (!chatId || !assistantId || !textToAdd) {
      return;
    }

    streamBufferRef.current = "";

    setChats((oldChats) =>
      oldChats.map((chat) => {
        if (chat.id !== chatId) {
          return chat;
        }

        return {
          ...chat,

          updatedAt: new Date().toISOString(),

          messages: chat.messages.map(
            (message) => {
              if (message.id !== assistantId) {
                return message;
              }

              return {
                ...message,

                text:
                  (message.text || "") +
                  textToAdd,

                streaming: true,
              };
            }
          ),
        };
      })
    );
  };

  // =====================================================
  // SCHEDULE STREAM FLUSH
  // =====================================================

  const scheduleStreamFlush = () => {
    if (streamFlushTimerRef.current) {
      return;
    }

    streamFlushTimerRef.current =
      setTimeout(() => {
        streamFlushTimerRef.current = null;

        flushStreamBuffer();
      }, STREAM_UPDATE_INTERVAL);
  };

  // =====================================================
  // FINISH STREAM
  // =====================================================

  const finishStream = () => {
    if (streamFlushTimerRef.current) {
      clearTimeout(
        streamFlushTimerRef.current
      );

      streamFlushTimerRef.current = null;
    }

    flushStreamBuffer();

    const chatId = streamChatIdRef.current;

    const assistantId =
      streamAssistantIdRef.current;

    if (chatId && assistantId) {
      setChats((oldChats) =>
        oldChats.map((chat) => {
          if (chat.id !== chatId) {
            return chat;
          }

          return {
            ...chat,

            updatedAt: new Date().toISOString(),

            messages: chat.messages.map(
              (message) => {
                if (message.id !== assistantId) {
                  return message;
                }

                return {
                  ...message,

                  streaming: false,
                };
              }
            ),
          };
        })
      );
    }

    streamChatIdRef.current = null;

    streamAssistantIdRef.current = null;

    streamBufferRef.current = "";
  };

  // =====================================================
  // SEND MESSAGE
  // =====================================================

  const sendMessage = async (
    voiceText?: string
  ) => {
    if (loading || isStreaming) {
      return;
    }

    // NEW: Accept voice transcript directly.
    const question = (
      typeof voiceText === "string"
        ? voiceText
        : prompt
    ).trim();

    if (!question && !image && !preview) {
      alert(
        "Please upload a skin lesion image first."
      );

      return;
    }

    const finalQuestion =
      question ||
      "Please analyze this skin lesion image.";

    let chatId = activeChatId;

    // =================================================
    // CREATE NEW CHAT
    // =================================================

    if (!chatId) {
      chatId = Date.now().toString();

      const newChat: Chat = {
        id: chatId,

        title: generateTitle(finalQuestion),

        createdAt: new Date().toISOString(),

        updatedAt: new Date().toISOString(),

        messages: [],
      };

      setChats((oldChats) => [
        newChat,
        ...oldChats,
      ]);

      setActiveChatId(chatId);
    }

    // =================================================
    // CURRENT CHAT
    // =================================================

    const currentChat = chats.find(
      (chat) => chat.id === chatId
    );

    const firstMessage =
      !currentChat ||
      currentChat.messages.length === 0;

    // =================================================
    // FIRST MESSAGE IMAGE
    // =================================================

    let imageFile: File | null = null;

    if (firstMessage) {
      imageFile = await getImageForChat();

      if (!imageFile) {
        alert(
          "Please upload a skin lesion image first."
        );

        return;
      }
    }

    // =================================================
    // USER MESSAGE
    // =================================================

    const userMessage: ChatMessage = {
      id: Date.now().toString(),

      role: "user",

      text: finalQuestion,

      image: firstMessage
        ? preview || null
        : null,

      createdAt: new Date().toISOString(),
    };

    // =================================================
    // ADD USER MESSAGE
    // =================================================

    setChats((oldChats) =>
      oldChats.map((chat) => {
        if (chat.id !== chatId) {
          return chat;
        }

        return {
          ...chat,

          title:
            chat.messages.length === 0
              ? generateTitle(finalQuestion)
              : chat.title,

          updatedAt: new Date().toISOString(),

          messages: [
            ...chat.messages,
            userMessage,
          ],
        };
      })
    );

    // =================================================
    // CLEAR INPUT
    // =================================================

    setPrompt("");

    if (textareaRef.current) {
      textareaRef.current.value = "";
    }

    resetTextarea();

    // =================================================
    // START LOADING
    // =================================================

    setLoading(true);

    try {
      // ===============================================
      // FORM DATA
      // ===============================================

      const formData = new FormData();

      // ===============================================
      // IMAGE ONLY FIRST MESSAGE
      // ===============================================

      if (firstMessage && imageFile) {
        formData.append("file", imageFile);
      }

      // ===============================================
      // QUESTION
      // ===============================================

      formData.append(
        "prompt",
        finalQuestion
      );

      // ===============================================
      // FIRST MESSAGE
      // ===============================================

      formData.append(
        "first_message",
        firstMessage ? "true" : "false"
      );

      // ===============================================
      // FOLLOW-UP
      // ===============================================

      if (!firstMessage) {
        const latestChat = chats.find(
          (chat) => chat.id === chatId
        );

        const previousAssistantMessage = [
          ...(latestChat?.messages || []),
        ]
          .reverse()
          .find(
            (message) =>
              message.role === "assistant" &&
              message.prediction
          );

        if (previousAssistantMessage?.prediction) {
          const previousPrediction =
            previousAssistantMessage.prediction;

          formData.append(
            "previous_disease",
            previousPrediction.disease
          );

          // Normal skin has confidence = null.
          // Do NOT send the string "null".

          if (
            previousPrediction.confidence !== null &&
            previousPrediction.confidence !== undefined
          ) {
            formData.append(
              "previous_confidence",
              String(
                previousPrediction.confidence
              )
            );
          } else {
            formData.append(
              "previous_confidence",
              ""
            );
          }
        }
      }

      // ===============================================
      // DEBUG
      // ===============================================

      console.log("================================");

      console.log("Sending message...");

      console.log("Question:", finalQuestion);

      console.log("First message:", firstMessage);

      if (imageFile) {
        console.log("Image:", imageFile.name);
      } else {
        console.log("Follow-up: image not sent");
      }

      console.log("================================");

      // ===============================================
      // FETCH
      // ===============================================

      const response = await fetch(
        `${API_URL}/chat`,
        {
          method: "POST",
          body: formData,
        }
      );

      // ===============================================
      // BACKEND ERROR
      // ===============================================

      if (!response.ok) {
        let errorText =
          "Unable to process the request.";

        try {
          const errorData = await response.json();

          if (
            typeof errorData?.detail === "string"
          ) {
            errorText = errorData.detail;
          } else if (
            errorData?.detail?.message
          ) {
            errorText =
              errorData.detail.message;
          } else if (
            Array.isArray(errorData?.detail)
          ) {
            errorText = errorData.detail
              .map(
                (item: {
                  msg?: string;
                  [key: string]: unknown;
                }) =>
                  item.msg ||
                  JSON.stringify(item)
              )
              .join(", ");
          }
        } catch (parseError) {
          console.error(
            "Backend error parsing failed:",
            parseError
          );
        }

        throw new Error(errorText);
      }

      // ===============================================
      // STREAM CHECK
      // ===============================================

      if (!response.body) {
        throw new Error(
          "Streaming response is not available."
        );
      }

      // ===============================================
      // ASSISTANT MESSAGE
      // ===============================================

      const assistantId =
        (Date.now() + 1).toString();

      const assistantMessage: ChatMessage = {
        id: assistantId,

        role: "assistant",

        text: "",

        prediction: null,

        note: null,

        validation: null,

        normalSkin: null,

        streaming: true,

        createdAt: new Date().toISOString(),
      };

      // ===============================================
      // SAVE STREAM REFERENCES
      // ===============================================

      streamChatIdRef.current = chatId;

      streamAssistantIdRef.current =
        assistantId;

      streamBufferRef.current = "";

      // ===============================================
      // ADD ASSISTANT MESSAGE
      // ===============================================

      setChats((oldChats) =>
        oldChats.map((chat) => {
          if (chat.id !== chatId) {
            return chat;
          }

          return {
            ...chat,

            updatedAt: new Date().toISOString(),

            messages: [
              ...chat.messages,
              assistantMessage,
            ],
          };
        })
      );

      // ===============================================
      // STREAM READER
      // ===============================================

      const reader = response.body.getReader();

      const decoder = new TextDecoder("utf-8");

      let buffer = "";

      let receivedFirstToken = false;

      // ===============================================
      // READ STREAM
      // ===============================================

      while (true) {
        const { value, done } =
          await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, {
          stream: true,
        });

        const lines = buffer.split("\n");

        buffer = lines.pop() || "";

        // =============================================
        // PROCESS NDJSON
        // =============================================

        for (const line of lines) {
          const cleanLine = line
            .trim()
            .replace(/^data:\s*/, "");

        if (!cleanLine || cleanLine === "[DONE]") {
          continue;
        }

          let event: Record<string, any>;

          try {
            event = JSON.parse(cleanLine);
          } catch (parseError) {
            console.warn(
              "Invalid stream JSON:",
              cleanLine
            );

            continue;
          }

          // =========================================
          // PREDICTION
          // =========================================

          if (event.type === "prediction") {
            console.log(
              "Prediction event:",
              event
            );

            setChats((oldChats) =>
              oldChats.map((chat) => {
                if (chat.id !== chatId) {
                  return chat;
                }

                return {
                  ...chat,

                  messages: chat.messages.map(
                    (message) => {
                      if (
                        message.id !== assistantId
                      ) {
                        return message;
                      }

                      // =================================
                      // PREDICTION OBJECT
                      // =================================

                      const prediction =
                        event.prediction_performed
                          ? {
                              disease:
                                event.disease ||
                                "Unknown",

                              confidence:
                                typeof event.confidence ===
                                "number"
                                  ? event.confidence
                                  : null,

                              isNormal:
                                event.disease ===
                                "No Disease Detected",

                              // Grad-CAM URL
                              gradcamUrl:
                                event.gradcam_url ||
                                null,
                            }
                          : null;

                      return {
                        ...message,

                        prediction,

                        note:
                          event.prediction_performed
                            ? (
                                event.disease ===
                                "No Disease Detected"
                                  ? "The research model did not detect a clear abnormality. This does not confirm that the skin is medically healthy."
                                  : "The image classification result is an AI research prediction and not a confirmed medical diagnosis."
                              )
                            : null,

                        validation:
                          event.validation || null,

                        normalSkin:
                          event.normal_skin || null,
                      };
                    }
                  ),
                };
              })
            );
          }

          // =========================================
          // TOKEN
          // =========================================

          if (event.type === "token") {
            const token =
              event.content ||
              event.token ||
              "";

            if (!token) {
              continue;
            }

            if (!receivedFirstToken) {
              receivedFirstToken = true;

              setLoading(false);
            }

            streamBufferRef.current += token;

            scheduleStreamFlush();
          }

          // =========================================
          // DONE
          // =========================================

          if (event.type === "done") {
            console.log(
              "Groq stream completed."
            );

            finishStream();
          }

          // =========================================
          // ERROR
          // =========================================

          if (event.type === "error") {
            throw new Error(
              event.message ||
              "Groq streaming failed."
            );
          }
        }
      }

      // ===============================================
      // FINAL FLUSH
      // ===============================================

      // Process any remaining NDJSON record.
      const remainingLine = buffer.trim();

      if (remainingLine) {
        try {
          const event = JSON.parse(remainingLine);

          if (event.type === "token") {
            streamBufferRef.current +=
              event.content || event.token || "";
          }

          if (event.type === "error") {
            throw new Error(
              event.message ||
              "Groq streaming failed."
            );
          }

          if (event.type === "prediction") {
            setChats((oldChats) =>
              oldChats.map((chat) => {
                if (chat.id !== chatId) {
                  return chat;
                }

                return {
                  ...chat,

                  messages: chat.messages.map(
                    (message) => {
                      if (
                        message.id !== assistantId
                      ) {
                        return message;
                      }

                      return {
                        ...message,

                        prediction:
                          event.prediction_performed
                            ? {
                                disease:
                                  event.disease ||
                                  "Unknown",

                                confidence:
                                  typeof event.confidence ===
                                  "number"
                                    ? event.confidence
                                    : null,

                                isNormal:
                                  event.disease ===
                                  "No Disease Detected",

                                gradcamUrl:
                                  event.gradcam_url ||
                                  null,
                              }
                            : null,

                        note:
                          event.prediction_performed
                            ? (
                                event.disease ===
                                "No Disease Detected"
                                  ? "The research model did not detect a clear abnormality. This does not confirm that the skin is medically healthy."
                                  : "The image classification result is an AI research prediction and not a confirmed medical diagnosis."
                              )
                            : null,

                        validation:
                          event.validation || null,

                        normalSkin:
                          event.normal_skin || null,
                      };
                    }
                  ),
                };
              })
            );
          }
        } catch (parseError) {
          if (parseError instanceof Error &&
              (
                parseError.message ===
                "Groq streaming failed."
              )) {
            throw parseError;
          }

          console.warn(
            "Unable to parse remaining stream data:",
            parseError
          );
        }
      }

      finishStream();
    } catch (error) {
      console.error("CHAT ERROR:", error);

      // ===============================================
      // STOP STREAM TIMER
      // ===============================================

      if (streamFlushTimerRef.current) {
        clearTimeout(
          streamFlushTimerRef.current
        );

        streamFlushTimerRef.current = null;
      }

      streamBufferRef.current = "";

      streamChatIdRef.current = null;

      streamAssistantIdRef.current = null;

      // ===============================================
      // USER FRIENDLY ERROR
      // ===============================================

      const errorText =
        error instanceof TypeError &&
        error.message.toLowerCase().includes("fetch")
    ? `Cannot connect to the API at ${API_URL}. Check the backend, CORS settings, and browser Network tab.`
    : error instanceof Error
      ? error.message
      : "Unable to connect to SkinGuardian AI.";

console.error("CHAT ERROR DETAILS:", {
  error,
  apiUrl: API_URL,
});

      const errorMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),

        role: "assistant",

        text: `❌ ${errorText}`,

        error: true,

        streaming: false,

        createdAt: new Date().toISOString(),
      };

      setChats((oldChats) =>
        oldChats.map((chat) => {
          if (chat.id !== chatId) {
            return chat;
          }

          return {
            ...chat,

            updatedAt: new Date().toISOString(),

            messages: [
              ...chat.messages,
              errorMessage,
            ],
          };
        })
      );
    } finally {
      setLoading(false);

      setImage(null);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  // =====================================================
  // ENTER / SHIFT ENTER
  // =====================================================

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      void sendMessage();
    }
  };

  // =====================================================
  // SUGGESTION
  // =====================================================

  const useSuggestion = (text: string) => {
    setPrompt(text);

    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();

        autoResizeTextarea(textareaRef.current);
      }
    }, 0);
  };

  // =====================================================
  // MESSAGES
  // =====================================================

  const messages =
    activeChat?.messages || [];

  // =====================================================
  // STREAMING CHECK
  // =====================================================

  const isStreaming = messages.some(
    (message) => message.streaming === true
  );

  // =====================================================
  // UI
  // =====================================================

  return (
    <div className="app">

      {/* ==================================================
          SIDEBAR
      ================================================== */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">
            🩺
          </div>

          <div>

            <div className="brand-title">
              SkinGuardian
            </div>

            <div className="brand-subtitle">
              AI Medical Assistant
            </div>

          </div>

        </div>

        <button
          className="new-chat-button"
          onClick={createNewChat}
          disabled={loading}
        >

          <span>＋</span>

          New Chat

        </button>

        <div className="recent-header">

          <span>
            Recent Chats
          </span>

          <span className="chat-count">
            {chats.length}
          </span>

        </div>

        <div className="chat-history">

          {chats.length === 0 ? (

            <div className="no-chats">

              <div className="no-chats-icon">
                💬
              </div>

              <p>
                No previous chats
              </p>

              <small>
                Your conversations
                will appear here.
              </small>

            </div>

          ) : (

            chats.map((chat) => (

              <div
                key={chat.id}
                className={`history-item ${
                  activeChatId === chat.id
                    ? "active"
                    : ""
                }`}
                onClick={() => openChat(chat.id)}
              >

                <div className="history-icon">
                  💬
                </div>

                <div className="history-text">

                  <div className="history-title">
                    {chat.title}
                  </div>

                  <div className="history-date">

                    {new Date(
                      chat.updatedAt
                    ).toLocaleDateString()}

                  </div>

                </div>

                <button
                  className="delete-chat"
                  onClick={(event) =>
                    deleteChat(chat.id, event)
                  }
                  disabled={loading}
                >
                  🗑
                </button>

              </div>

            ))

          )}

        </div>

        <div className="sidebar-bottom">

          <div className="ai-info">
            🧠
            <span>
              Skin Lesion AI
            </span>
          </div>

          <div className="ai-info">
            ⚡
            <span>
              Groq GPT-OSS-20B
            </span>
          </div>

          <div className="online-status">

            <span></span>

            AI Online

          </div>

        </div>

      </aside>

      {/* ==================================================
          MAIN
      ================================================== */}

      <main className="main">

        {/* HEADER */}

        <header className="header">

          <div className="header-left">

            <div className="mobile-logo">
              🩺
            </div>

            <div>

              <div className="header-title">

                {activeChat
                  ? activeChat.title
                  : "SkinGuardian AI"}

              </div>

              <div className="header-status">

                <span>
                  ●
                </span>

                AI Assistant Online

              </div>

            </div>

          </div>

          <button
            className="header-new-chat"
            onClick={createNewChat}
            disabled={loading}
          >
            ＋ New Chat
          </button>

        </header>

        {/* ==================================================
            CHAT AREA
        ================================================== */}

        <section className="chat-area">

          {!activeChat ? (

            <div className="welcome">

              <div className="welcome-icon">
                🩺
              </div>

              <h1>
                How can I help you?
              </h1>

              <p>
                Upload a skin lesion
                image and ask
                SkinGuardian AI about it.
              </p>

              <div className="suggestions">

                <button
                  onClick={() =>
                    useSuggestion(
                      "What disease is this?"
                    )
                  }
                >

                  🔍

                  <div>

                    <b>
                      What disease is this?
                    </b>

                    <small>
                      Analyze the skin lesion
                    </small>

                  </div>

                </button>

                <button
                  onClick={() =>
                    useSuggestion(
                      "What symptoms should I watch for?"
                    )
                  }
                >

                  ⚠️

                  <div>

                    <b>
                      What symptoms should I watch for?
                    </b>

                    <small>
                      Learn about warning signs
                    </small>

                  </div>

                </button>

                <button
                  onClick={() =>
                    useSuggestion(
                      "What safety precautions should I follow?"
                    )
                  }
                >

                  🛡️

                  <div>

                    <b>
                      What safety precautions should I follow?
                    </b>

                    <small>
                      Get general safety information
                    </small>

                  </div>

                </button>

                <button
                  onClick={() =>
                    useSuggestion(
                      "When should I see a dermatologist?"
                    )
                  }
                >

                  👨‍⚕️

                  <div>

                    <b>
                      When should I see a dermatologist?
                    </b>

                    <small>
                      Know when to seek professional care
                    </small>

                  </div>

                </button>

              </div>

            </div>

          ) : (

            <div className="messages">

              {messages.map((message) => (

                <div
                  className={`message-row ${
                    message.role
                  }`}
                  key={message.id}
                >

                  <div className="avatar">

                    {message.role === "user"
                      ? "👤"
                      : "🤖"}

                  </div>

                  <div className="message-body">

                    <div className="message-name">

                      {message.role === "user"
                        ? "You"
                        : "SkinGuardian AI"}

                    </div>

                    {/* IMAGE */}

                    {message.image && (

                      <img
                        src={message.image}
                        className="chat-image"
                        alt="Skin lesion"
                      />

                    )}

                    {/* ==================================================
                        PREDICTION CARD
                    ================================================== */}

                    {message.prediction && (

                      <div className="prediction-card">

                        <div className="prediction-header">
                          🧠 Skin AI Prediction
                        </div>

                        <div className="prediction-grid">

                          {/* PREDICTED CONDITION */}

                          <div className="prediction-item">

                            <span>
                              Predicted Condition
                            </span>

                            <strong>
                              {message.prediction.disease}
                            </strong>

                          </div>

                          {/* CONFIDENCE */}

                          {!message.prediction.isNormal && (

                            <div className="prediction-item">

                              <span>
                                Confidence
                              </span>

                              <strong>

                                {message.prediction.confidence !==
                                null
                                  ? `${message.prediction.confidence}%`
                                  : "N/A"}

                              </strong>

                            </div>

                          )}

                        </div>

                        {/* NORMAL SKIN NOTE */}

                        {message.prediction.isNormal && (

                          <div className="normal-result-note">

                            ✓ No clear abnormality was detected
                            by the research model.

                          </div>

                        )}

                        {/* ==================================================
                            GRAD-CAM
                        ================================================== */}

                        {message.prediction.gradcamUrl &&
                          !message.prediction.isNormal && (

                          <div className="gradcam-section">

                            <div className="gradcam-title">
                              🔥 AI Attention / Grad-CAM
                            </div>

                            <p className="gradcam-description">
                              Highlighted areas show regions
                              that contributed to the
                              CNN-based prediction.
                            </p>

                            <img
                              src={`${API_URL}${message.prediction.gradcamUrl}`}
                              className="gradcam-image"
                              alt="Grad-CAM explanation"
                              onError={(event) => {
                                console.error(
                                  "Grad-CAM image failed to load:",
                                  event
                                );

                                event.currentTarget.style.display =
                                  "none";
                              }}
                            />

                            <small className="gradcam-warning">
                              Grad-CAM is an explainability
                              visualization. It does not
                              confirm that the highlighted
                              area is medically diseased.
                            </small>

                          </div>

                        )}

                      </div>

                    )}

                    {/* AI RESPONSE */}

                    <div
                      className={`message-text ${
                        message.error
                          ? "error"
                          : ""
                      }`}
                    >

                      {message.text}

                      {message.streaming && (

                        <span className="streaming-cursor">
                          ▌
                        </span>

                      )}

                    </div>

                    {/* MEDICAL NOTE */}

                    {message.note && (

                      <div className="medical-note">

                        ⚠️{" "}
                        {message.note}

                      </div>

                    )}

                  </div>

                </div>

              ))}

              {/* LOADING */}

              {loading &&
                !isStreaming && (

                  <div className="message-row assistant">

                    <div className="avatar">
                      🤖
                    </div>

                    <div className="message-body">

                      <div className="message-name">
                        SkinGuardian AI
                      </div>

                      <div className="typing">

                        <span></span>
                        <span></span>
                        <span></span>

                      </div>

                    </div>

                  </div>

                )}

              <div ref={bottomRef}></div>

            </div>

          )}

        </section>

        {/* ==================================================
            INPUT
        ================================================== */}

        <div className="input-section">

          {/* IMAGE PREVIEW */}

          {preview && (

            <div className="preview-wrapper">

              <div className="preview-card">

                <img
                  src={preview}
                  alt="Selected lesion"
                />

                <button
                  className="remove-image"
                  onClick={removeImage}
                  disabled={loading}
                >
                  ×
                </button>

              </div>

            </div>

          )}

          {/* INPUT BOX */}

          <div className="input-box">

            <button
              className="attach-button"
              onClick={() =>
                fileInputRef.current?.click()
              }
              title="Upload image"
              disabled={loading}
            >
              📎
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.webp"
              onChange={handleImage}
              hidden
            />

            {/* ==================================================
                NEW: VOICE INPUT BUTTON
            ================================================== */}

            <button
              type="button"
              className={`voice-button ${
                voiceListening ? "recording" : ""
              }`}
              onClick={
                voiceListening
                  ? stopVoiceInput
                  : handleVoiceInput
              }
              disabled={loading}
              title={
                voiceListening
                  ? "Stop voice input and send"
                  : "Speak your message"
              }
              aria-label={
                voiceListening
                  ? "Stop voice input and send"
                  : "Start voice input"
              }
            >

              {voiceListening ? "⏹️" : "🎤"}

            </button>

            <textarea
              ref={textareaRef}
              value={prompt}
              onChange={handlePromptChange}
              onKeyDown={handleKeyDown}
              placeholder={
                voiceListening
                  ? "Listening... Speak now"
                  : activeChat
                    ? "Ask about this skin lesion..."
                    : "Upload an image and ask SkinGuardian AI..."
              }
              rows={1}
              disabled={loading}
            />

            <button
              className="send-button"
              onClick={() => void sendMessage()}
              disabled={
                loading ||
                (
                  !image &&
                  !preview &&
                  !prompt.trim()
                )
              }
              title="Send"
            >
              ➤
            </button>

          </div>

          {/* ==================================================
              NEW: VOICE STATUS
          ================================================== */}

          {voiceListening && (

            <div className="voice-status">
              <span className="voice-status-dot"></span>
              Listening... Speak your message.
              Pause to send automatically.
            </div>

          )}

          {voiceError && (

            <div className="voice-error">
              {voiceError}
            </div>

          )}

          <div className="input-hint">

            Enter to send • Shift + Enter
            for new line • 🎤 Speak to send

          </div>

          <div className="disclaimer">

            SkinGuardian AI provides
            research-oriented information
            and is not a substitute for
            professional medical diagnosis.

          </div>

        </div>

      </main>

    </div>
  );
}

export default App;