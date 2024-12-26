use std::sync::RwLock;

use lru::LruCache;
use reqwest::Client;
use schemars::{schema_for, transform::Transform, JsonSchema, Schema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

use crate::schema::OpenAiTransform;

/// To use this library, you need to create a [`ChatClient`]. This contains various information needed to interact with the ChatGPT API,
/// such as the API key, the model to use, and the URL of the API.
///
/// ```rust
/// # use tysm::ChatClient;
/// // Create a client with your API key and model
/// let client = ChatClient::new("sk-1234567890", "gpt-4o");
/// ```
///
/// ```rust
/// # use tysm::ChatClient;
/// // Create a client using an API key stored in an `OPENAI_API_KEY` environment variable.
/// // (This will also look for an `.env` file in the current directory.)
/// let client = ChatClient::from_env("gpt-4o").unwrap();
/// ```
pub struct ChatClient {
    /// The API key to use for the ChatGPT API.
    pub api_key: String,
    /// The URL of the ChatGPT API. Customize this if you are using a custom API that is compatible with OpenAI's.
    pub url: String,
    /// The model to use for the ChatGPT API.
    pub model: String,
    /// A cache of the few responses. Stores the last 1024 responses by default.
    pub lru: RwLock<LruCache<String, String>>,
    /// This client's token consumption (as reported by the API).
    pub usage: RwLock<ChatUsage>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "system")]
    System,
}

/// A message to send to the ChatGPT API.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatMessage {
    /// The role of user sending the message.
    pub role: Role,
    /// The content of the message. It is a vector of [`ChatMessageContent`]s,
    /// which allows you to include images in the message.
    pub content: Vec<ChatMessageContent>,
}

/// The content of a message.
///
/// Currently, only text and image URLs are supported.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatMessageContent {
    /// A textual message.
    Text {
        /// The text of the message.
        text: String,
    },
    /// An image URL.
    /// The image URL can also be a base64 encoded image.
    /// example:
    /// ```rust
    /// use tysm::{ChatMessageContent, ImageUrl};
    ///
    /// let base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
    /// let content = ChatMessageContent::ImageUrl {
    ///     image: ImageUrl {
    ///         url: format!("data:image/png;base64,{base64_image}"),
    ///     },
    /// };
    /// ```
    ImageUrl {
        /// The image URL.
        #[serde(rename = "image_url")]
        image: ImageUrl,
    },
}

/// An image URL. OpenAI will accept a link to an image, or a base64 encoded image.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageUrl {
    /// The image URL.
    pub url: String,
}

/// A request to the ChatGPT API. You probably will not need to use this directly,
/// but it is public because it is still exposed in errors.
#[derive(Serialize, Clone, Debug)]
pub struct ChatRequest {
    /// The model to use for the ChatGPT API.
    pub model: String,
    /// The messages to send to the API.
    pub messages: Vec<ChatMessage>,
    /// The response format to use for the ChatGPT API.
    pub response_format: ResponseFormat,
}

#[derive(Serialize, Debug, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub json_schema: JsonSchemaFormat,
}

#[derive(Serialize, Debug, Clone)]
pub struct JsonSchemaFormat {
    name: String,
    strict: bool,
    schema: SchemaFormat,
}

#[derive(Serialize, Debug, Clone)]
pub struct SchemaFormat {
    #[serde(rename = "additionalProperties")]
    additional_properties: bool,
    #[serde(flatten)]
    schema: Schema,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatMessageResponse {
    pub role: Role,
    pub content: String,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    #[expect(unused)]
    id: String,
    #[expect(unused)]
    object: String,
    #[expect(unused)]
    created: u64,
    #[expect(unused)]
    model: String,
    #[expect(unused)]
    system_fingerprint: Option<String>,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    #[expect(unused)]
    index: u8,
    message: ChatMessageResponse,
    #[expect(unused)]
    logprobs: Option<serde_json::Value>,
    #[expect(unused)]
    finish_reason: String,
}

#[derive(Deserialize, Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl std::ops::AddAssign for ChatUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.total_tokens += rhs.total_tokens;
    }
}

/// An error that occurs when the OpenAI API key is not found in the environment.
#[derive(Debug)]
pub struct OpenAiApiKeyError(#[expect(unused)] std::env::VarError);
impl std::fmt::Display for OpenAiApiKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unable to find the OpenAI API key in the environment. Please set the OPENAI_API_KEY environment variable. API keys can be found at <https://platform.openai.com/api-keys>.")
    }
}
impl std::error::Error for OpenAiApiKeyError {}

fn api_key() -> Result<String, OpenAiApiKeyError> {
    #[cfg(feature = "dotenv")]
    {
        use dotenv::dotenv;
        dotenv().ok();
    }
    std::env::var("OPENAI_API_KEY").map_err(OpenAiApiKeyError)
}

/// Errors that can occur when interacting with the ChatGPT API.
#[derive(Error, Debug)]
pub enum ChatError {
    /// An error occurred when sending the request to the API.
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when serializing the request to JSON.
    #[error("JSON serialization error: {0}")]
    JsonSerializeError(serde_json::Error, ChatRequest),

    /// An error occurred when deserializing the response from the API.
    #[error("API returned an error response: {0} \nresponse: {1} \nrequest: {2}")]
    ApiResponseError(serde_json::Error, String, String),

    /// The API returned a response that was not a valid JSON object.
    #[error("API returned a response that was not a valid JSON object: {0} \nresponse: {1}")]
    InvalidJson(serde_json::Error, String),

    /// The API did not return any choices.
    #[error("No choices returned from API")]
    NoChoices,
}

impl ChatClient {
    /// Create a new [`ChatClient`].
    /// If the API key is in the environment, you can use the [`Self::from_env`] method instead.
    ///
    /// ```rust
    /// use tysm::ChatClient;
    ///
    /// let client = ChatClient::new("sk-1234567890", "gpt-4o");
    /// ```
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        use std::num::NonZeroUsize;

        Self {
            api_key: api_key.into(),
            url: "https://api.openai.com/v1/chat/completions".into(),
            model: model.into(),
            lru: RwLock::new(LruCache::new(NonZeroUsize::new(1024).unwrap())),
            usage: RwLock::new(ChatUsage::default()),
        }
    }

    /// Create a new [`ChatClient`].
    /// This will use the `OPENAI_API_KEY` environment variable to set the API key.
    /// It will also look in the `.env` file for an `OPENAI_API_KEY` variable (using dotenv).
    ///
    /// ```rust
    pub fn from_env(model: impl Into<String>) -> Result<Self, OpenAiApiKeyError> {
        Ok(Self::new(api_key()?, model))
    }

    /// Send a chat message to the API and deserialize the response into the given type.
    ///
    /// ```rust
    /// # use tysm::ChatClient;
    /// #  let client = {
    /// #     let my_api = "https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/chat/completions".to_string();
    /// #     ChatClient {
    /// #         url: my_api,
    /// #         ..ChatClient::from_env("gpt-4o").unwrap()
    /// #     }
    /// # };
    ///
    /// #[derive(serde::Deserialize, Debug, schemars::JsonSchema)]
    /// struct CityName {
    ///     english: String,
    ///     local: String,
    /// }
    ///
    /// # tokio_test::block_on(async {
    /// let response: CityName = client.chat("What is the capital of Portugal?").await.unwrap();
    ///
    /// assert_eq!(response.english, "Lisbon");
    /// assert_eq!(response.local, "Lisboa");
    /// # })
    /// ```
    ///
    /// The last 1024 Responses are cached in the client, so sending the same request twice
    /// will return the same response.
    pub async fn chat<T: DeserializeOwned + JsonSchema>(
        &self,
        prompt: impl Into<String>,
    ) -> Result<T, ChatError> {
        self.chat_with_system_prompt(prompt, "").await
    }

    /// Send a chat message to the API and deserialize the response into the given type.
    /// The system prompt is used to set the context of the conversation.
    pub async fn chat_with_system_prompt<T: DeserializeOwned + JsonSchema>(
        &self,
        prompt: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Result<T, ChatError> {
        let prompt = prompt.into();
        let system_prompt = system_prompt.into();

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: vec![ChatMessageContent::Text {
                    text: system_prompt,
                }],
            },
            ChatMessage {
                role: Role::User,
                content: vec![ChatMessageContent::Text { text: prompt }],
            },
        ];
        self.chat_with_messages::<T>(messages).await
    }

    /// Send a sequence of chat messages to the API and deserialize the response into the given type.
    /// This is useful for more advanced use cases like chatbots, multi-turn conversations, or when you need to use [Vision](https://platform.openai.com/docs/guides/vision).
    pub async fn chat_with_messages<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<T, ChatError> {
        let mut schema = schema_for!(T);
        OpenAiTransform.transform(&mut schema);

        let ty_name = tynm::type_name::<T>();
        let ty_name = if ty_name.is_empty() {
            "response".to_string()
        } else {
            ty_name
        };

        let chat_request = ChatRequest {
            model: self.model.clone(),
            messages,
            response_format: ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: JsonSchemaFormat {
                    name: ty_name,
                    strict: true,
                    schema: SchemaFormat {
                        additional_properties: false,
                        schema,
                    },
                },
            },
        };

        let chat_request_str = serde_json::to_string(&chat_request).unwrap();

        let chat_response = if let Some(cached_response) = self.chat_cached(&chat_request).await {
            let chat_response: ChatResponse =
                serde_json::from_str(&cached_response).map_err(|e| {
                    ChatError::ApiResponseError(e, cached_response.clone(), chat_request_str)
                })?;
            chat_response
        } else {
            let chat_response = self.chat_uncached(&chat_request).await?;
            let chat_response: ChatResponse =
                serde_json::from_str(&chat_response).map_err(|e| {
                    ChatError::ApiResponseError(e, chat_response.clone(), chat_request_str)
                })?;
            if let Ok(mut usage) = self.usage.write() {
                *usage += chat_response.usage;
            }
            chat_response
        };
        let chat_response = chat_response
            .choices
            .first()
            .ok_or(ChatError::NoChoices)?
            .message
            .content
            .clone();

        let chat_response: T = serde_json::from_str(&chat_response)
            .map_err(|e| ChatError::InvalidJson(e, chat_response.clone()))?;

        Ok(chat_response)
    }

    async fn chat_cached(&self, chat_request: &ChatRequest) -> Option<String> {
        let chat_request = serde_json::to_string(chat_request).ok()?;

        let mut lru = self.lru.write().ok()?;

        lru.get(&chat_request).cloned()
    }

    async fn chat_uncached(&self, chat_request: &ChatRequest) -> Result<String, ChatError> {
        let reqwest_client = Client::new();

        let response = reqwest_client
            .post(self.url.clone())
            .header("Authorization", format!("Bearer {}", self.api_key.clone()))
            .header("Content-Type", "application/json")
            .json(chat_request)
            .send()
            .await?
            .text()
            .await?;

        let chat_request = serde_json::to_string(chat_request)
            .map_err(|e| ChatError::JsonSerializeError(e, chat_request.clone()))?;

        self.lru
            .write()
            .ok()
            .unwrap()
            .put(chat_request, response.clone());

        Ok(response)
    }

    /// Returns how many tokens have been used so far.
    ///
    /// Does not double-count tokens used in cached responses.
    pub fn usage(&self) -> ChatUsage {
        *self.usage.read().unwrap()
    }
}

#[test]
fn test_deser() {
    let s = r#"
{
    "choices": [
        {
        "finish_reason": "stop",
        "index": 0,
        "logprobs": null,
        "message": {
            "content": "Hey there! When replying to someone on a dating app who's asked about what you're studying, it's all about how you present it. Even if you think math might sound boring, you can share why you find it interesting or how it applies to everyday life. Try saying something like, \"I'm actually diving into the world of math! It's fascinating because [insert a fun fact about your studies or why you chose it]. What about you? What are you passionate about?\" This way, you're flipping the script from just stating your major to sharing your enthusiasm, which can be very attractive!",
            "role": "assistant"
        }
        }
    ],
    "created": 1714696172,
    "id": "chatcmpl-9Kb5oqHOdNRLuFJHCTQFOeU516mU8",
    "model": "gpt-4-0125-preview",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 123,
        "prompt_tokens": 188,
        "total_tokens": 311
    }
}
"#;
    let _chat_response: ChatResponse = serde_json::from_str(&s).unwrap();
}
