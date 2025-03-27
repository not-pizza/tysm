//! Chat completions are the most common way to interact with the OpenAI API.
//! This module provides a client for interacting with the ChatGPT API.

use std::sync::RwLock;

use lru::LruCache;
use reqwest::Client;
use schemars::{schema_for, transform::Transform, JsonSchema, Schema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

use crate::schema::OpenAiTransform;
use crate::utils::{api_key, OpenAiApiKeyError};

/// To use this library, you need to create a [`ChatClient`]. This contains various information needed to interact with the ChatGPT API,
/// such as the API key, the model to use, and the URL of the API.
///
/// ```rust
/// # use tysm::chat_completions::ChatClient;
/// // Create a client with your API key and model
/// let client = ChatClient::new("sk-1234567890", "gpt-4o");
/// ```
///
/// ```rust
/// # use tysm::chat_completions::ChatClient;
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

/// The role of a message.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Role {
    /// The user is sending the message.
    #[serde(rename = "user")]
    User,
    /// The assistant is sending the message.
    #[serde(rename = "assistant")]
    Assistant,
    /// The system is sending the message.
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
    /// use tysm::chat_completions::{ChatMessageContent, ImageUrl};
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
    #[allow(private_interfaces)]
    pub response_format: ResponseFormat,
}

/// An object specifying the format that the model must output.
/// `ResponseFormat::JsonSchema` enables Structured Outputs which ensures the model will match your supplied JSON schema
#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    /// The model is constrained to return a JSON object of the specified schema.
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// The schema.
        /// Often generated with `JsonSchemaFormat::new()`.
        json_schema: JsonSchemaFormat,
    },

    /// The model is constrained to return a JSON object, but the schema is not enforced.
    #[serde(rename = "json_object")]
    JsonObject,

    /// The model is not constrained to any specific format.
    #[serde(rename = "text")]
    Text,
}

/// The format of a JSON schema.
#[derive(Serialize, Debug, Clone)]
pub struct JsonSchemaFormat {
    /// The name of the schema. It's not clear whether this is actually used anywhere by OpenAI.
    pub name: String,
    /// Whether the schema is strict. (For openai, you always want this to be true.)
    pub strict: bool,
    /// The schema.
    pub schema: SchemaFormat,
}

impl JsonSchemaFormat {
    /// Create a new `JsonSchemaFormat`.
    pub fn new<T: JsonSchema>() -> Self {
        let mut schema = schema_for!(T);
        let name = tynm::type_name::<T>();
        let name = if name.is_empty() {
            "response".to_string()
        } else {
            name
        };

        OpenAiTransform.transform(&mut schema);

        Self::from_schema(schema, &name)
    }

    /// Create a new `JsonSchemaFormat` from a `Schema`.
    pub fn from_schema(schema: Schema, ty_name: &str) -> Self {
        Self {
            name: ty_name.to_string(),
            strict: true,
            schema: SchemaFormat {
                additional_properties: false,
                schema,
            },
        }
    }
}

/// A JSON schema with an "additionalProperties" field (expected by OpenAI).
#[derive(Serialize, Debug, Clone)]
pub struct SchemaFormat {
    /// Whether additional properties are allowed. For OpenAI, you always want this to be false.
    #[serde(rename = "additionalProperties")]
    pub additional_properties: bool,

    /// The schema.
    #[serde(flatten)]
    pub schema: Schema,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct ChatMessageResponse {
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

/// The token consumption of the chat-completions API.
#[derive(Deserialize, Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ChatUsage {
    /// The number of tokens used for the prompt.
    pub prompt_tokens: u32,
    /// The number of tokens used for the completion.
    pub completion_tokens: u32,
    /// The total number of tokens used.
    pub total_tokens: u32,

    /// Details about the prompt tokens (such as whether they were cached).
    #[serde(default)]
    pub prompt_token_details: Option<PromptTokenDetails>,
    /// Details about the completion tokens for reasoning models
    #[serde(default)]
    pub completion_token_details: Option<CompletionTokenDetails>,
}

/// Includes details about the prompt tokens.
/// Currently, only contains the number of cached tokens.
#[derive(Deserialize, Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct PromptTokenDetails {
    cached_tokens: u32,
}

/// Includes details about the completion tokens for reasoning models
#[derive(Deserialize, Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct CompletionTokenDetails {
    reasoning_tokens: u32,
    accepted_prediction_tokens: u32,
    rejected_prediction_tokens: u32,
}

impl std::ops::AddAssign for ChatUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.total_tokens += rhs.total_tokens;
    }
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
    /// use tysm::chat_completions::ChatClient;
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
    /// # use tysm::chat_completions::ChatClient;
    /// let client = ChatClient::from_env("gpt-4o").unwrap();
    /// ```
    pub fn from_env(model: impl Into<String>) -> Result<Self, OpenAiApiKeyError> {
        Ok(Self::new(api_key()?, model))
    }

    /// Send a chat message to the API and deserialize the response into the given type.
    ///
    /// ```rust
    /// # use tysm::chat_completions::ChatClient;
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
    ///
    /// **Important:** The response type must implement the `JsonSchema` trait
    /// from an in-development version of the `schemars` crate. The version of `schemars` published on crates.io will not work.
    /// Add the in-development version to your Cargo.toml like this:
    /// ```rust,ignore
    /// [dependencies]
    /// schemars = { git = "https://github.com/GREsau/schemars.git", version = "1.0.0-alpha.17", features = [
    ///     "preserve_order",
    /// ] }
    /// ```
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
        let json_schema = JsonSchemaFormat::new::<T>();

        let response_format = ResponseFormat::JsonSchema { json_schema };

        let chat_response = self
            .chat_with_messages_raw::<T>(messages, response_format)
            .await?;

        let chat_response: T = serde_json::from_str(&chat_response)
            .map_err(|e| ChatError::InvalidJson(e, chat_response.clone()))?;

        Ok(chat_response)
    }

    /// Send a sequence of chat messages to the API. It's called "chat_with_messages_raw" because it allows you to specify any response format, and doesn't attempt to deserialize the chat completion.
    pub async fn chat_with_messages_raw<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: Vec<ChatMessage>,
        response_format: ResponseFormat,
    ) -> Result<String, ChatError> {
        let chat_request = ChatRequest {
            model: self.model.clone(),
            messages,
            response_format,
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
