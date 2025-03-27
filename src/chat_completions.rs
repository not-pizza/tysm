//! Chat completions are the most common way to interact with the OpenAI API.
//! This module provides a client for interacting with the ChatGPT API.
//!
//! It also provides a batch API for processing large numbers of requests asynchronously.

use std::sync::RwLock;

use lru::LruCache;
use reqwest::Client;
use schemars::{schema_for, transform::Transform, JsonSchema, Schema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

use crate::schema::OpenAiTransform;
use crate::utils::{api_key, OpenAiApiKeyError};
use crate::OpenAiError;

/// Batch API for processing large numbers of requests asynchronously.
///
/// The Batch API allows you to send asynchronous groups of requests with 50% lower costs,
/// a separate pool of significantly higher rate limits, and a clear 24-hour turnaround time.
///
/// # Example
///
/// ```rust,no_run
/// # use tysm::chat_completions::{ChatClient, batch::BatchRequestItem};
/// # use serde_json::json;
/// # use tokio_test::block_on;
/// # block_on(async {
/// let client = ChatClient::from_env("gpt-4o").unwrap();
///
/// // Create batch requests
/// let requests = vec![
///     BatchRequestItem::new_chat(
///         "request-1",
///         "gpt-4o",
///         vec![
///             json!({"role": "system", "content": "You are a helpful assistant."}),
///             json!({"role": "user", "content": "Hello world!"}),
///         ],
///     ),
///     BatchRequestItem::new_chat(
///         "request-2",
///         "gpt-4o",
///         vec![
///             json!({"role": "system", "content": "You are an unhelpful assistant."}),
///             json!({"role": "user", "content": "Hello world!"}),
///         ],
///     ),
/// ];
///
/// // Create a batch file
/// let file_id = client.create_batch_file("my_batch.jsonl", &requests).await.unwrap();
///
/// // Create a batch
/// let batch = client.create_batch(file_id, "/v1/chat/completions").await.unwrap();
/// println!("Batch created: {}", batch.id);
///
/// // Check batch status
/// let status = client.get_batch_status(&batch.id).await.unwrap();
/// println!("Batch status: {}", status.status);
///
/// // Wait for batch to complete
/// let completed_batch = client.wait_for_batch(&batch.id).await.unwrap();
///
/// // Get batch results
/// let results = client.get_batch_results(&completed_batch).await.unwrap();
/// for result in results {
///     println!("Result for {}: {}", result.custom_id, result.response.unwrap().body);
/// }
/// # });
/// ```
pub mod batch {
    use reqwest::Client;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::io::Write;

    use std::time::Duration;
    use thiserror::Error;
    use tokio::time::sleep;

    use super::*;
    use crate::files::{FilePurpose, FilesClient, FilesError};
    use crate::utils::remove_trailing_slash;
    use crate::OpenAiError;

    /// Errors that can occur when interacting with the Batch API.
    #[derive(Error, Debug)]
    pub enum BatchError {
        /// An error occurred when sending the request to the API.
        #[error("Request error: {0}")]
        RequestError(#[from] reqwest::Error),

        /// An error occurred when serializing or deserializing JSON.
        #[error("JSON error when parsing {1}")]
        JsonParseError(#[source] serde_json::Error, String),

        /// An error occurred with file operations.
        #[error("File error: {0}")]
        FileError(#[from] std::io::Error),

        /// An error occurred with the Files API.
        #[error("Files API error: {0}")]
        FilesApiError(#[from] FilesError),

        /// An error occurred with the Chat API.
        #[error("Chat API error: {0}")]
        ChatError(#[from] ChatError),

        /// An error occurred with the OpenAI API.
        #[error("OpenAI API error: {0}")]
        OpenAiError(#[from] OpenAiError),

        /// The batch has expired.
        #[error("Batch expired: {0}")]
        BatchExpired(String),

        /// The batch has failed.
        #[error("Batch failed: {0}")]
        BatchFailed(String),

        /// The batch was cancelled.
        #[error("Batch cancelled: {0}")]
        BatchCancelled(String),

        /// Timeout waiting for batch to complete.
        #[error("Timeout waiting for batch to complete: {0}")]
        BatchTimeout(String),

        /// Other batch error.
        #[error("Batch error: {0}")]
        Other(String),
    }

    /// A request item for a batch.
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct BatchRequestItem {
        /// A unique identifier for this request.
        pub custom_id: String,
        /// The HTTP method to use for this request.
        pub method: String,
        /// The URL to send this request to.
        pub url: String,
        /// The body of the request.
        pub body: Value,
    }

    /// A response item from a batch.
    #[derive(Deserialize, Debug, Clone)]
    pub struct BatchResponseItem {
        /// The ID of this response.
        pub id: String,
        /// The custom ID that was provided in the request.
        pub custom_id: String,
        /// The response from the API.
        pub response: Option<BatchItemResponse>,
        /// The error from the API, if any.
        pub error: Option<BatchItemError>,
    }

    /// A response from a batch item.
    #[derive(Deserialize, Debug, Clone)]
    pub struct BatchItemResponse {
        /// The HTTP status code of the response.
        pub status_code: u16,
        /// The request ID of the response.
        pub request_id: String,
        /// The body of the response.
        pub body: Value,
    }

    /// An error from a batch item.
    #[derive(Deserialize, Debug, Clone)]
    pub struct BatchItemError {
        /// The error code.
        pub code: String,
        /// The error message.
        pub message: String,
    }

    /// A batch object.
    #[derive(Deserialize, Debug, Clone)]
    pub struct Batch {
        /// The ID of the batch.
        pub id: String,
        /// The object type, always "batch".
        pub object: String,
        /// The endpoint that this batch is for.
        pub endpoint: String,
        /// Any errors that occurred during batch creation.
        pub errors: Option<Value>,
        /// The ID of the input file.
        pub input_file_id: String,
        /// The completion window for this batch.
        pub completion_window: String,
        /// The status of the batch.
        pub status: String,
        /// The ID of the output file, if available.
        pub output_file_id: Option<String>,
        /// The ID of the error file, if available.
        pub error_file_id: Option<String>,
        /// When the batch was created.
        pub created_at: u64,
        /// When the batch started processing.
        pub in_progress_at: Option<u64>,
        /// When the batch expires.
        pub expires_at: Option<u64>,
        /// When the batch completed.
        pub completed_at: Option<u64>,
        /// When the batch failed.
        pub failed_at: Option<u64>,
        /// When the batch expired.
        pub expired_at: Option<u64>,
        /// The number of requests in the batch.
        pub request_counts: BatchRequestCounts,
        /// Custom metadata for the batch.
        pub metadata: Option<Value>,
    }

    /// The number of requests in a batch.
    #[derive(Deserialize, Debug, Clone)]
    pub struct BatchRequestCounts {
        /// The total number of requests in the batch.
        pub total: u32,
        /// The number of completed requests in the batch.
        pub completed: u32,
        /// The number of failed requests in the batch.
        pub failed: u32,
    }

    /// A list of batches.
    #[derive(Deserialize, Debug, Clone)]
    pub struct BatchList {
        /// The list of batches.
        pub data: Vec<Batch>,
        /// The object type, always "list".
        pub object: String,
        /// Whether there are more batches to fetch.
        pub has_more: bool,
    }

    impl BatchRequestItem {
        /// Create a new batch request item for the chat completions API.
        pub fn new_chat(
            custom_id: impl Into<String>,
            model: impl Into<String>,
            messages: Vec<Value>,
        ) -> Self {
            Self {
                custom_id: custom_id.into(),
                method: "POST".to_string(),
                url: "/v1/chat/completions".to_string(),
                body: serde_json::json!({
                    "model": model.into(),
                    "messages": messages,
                    "max_tokens": 1000,
                }),
            }
        }

        /// Create a new batch request item for the embeddings API.
        pub fn new_embedding(
            custom_id: impl Into<String>,
            model: impl Into<String>,
            input: Vec<String>,
        ) -> Self {
            Self {
                custom_id: custom_id.into(),
                method: "POST".to_string(),
                url: "/v1/embeddings".to_string(),
                body: serde_json::json!({
                    "model": model.into(),
                    "input": input,
                }),
            }
        }

        /// Create a new batch request item for the completions API.
        pub fn new_completion(
            custom_id: impl Into<String>,
            model: impl Into<String>,
            prompt: impl Into<String>,
        ) -> Self {
            Self {
                custom_id: custom_id.into(),
                method: "POST".to_string(),
                url: "/v1/completions".to_string(),
                body: serde_json::json!({
                    "model": model.into(),
                    "prompt": prompt.into(),
                    "max_tokens": 1000,
                }),
            }
        }

        /// Create a new batch request item for the responses API.
        pub fn new_response(
            custom_id: impl Into<String>,
            model: impl Into<String>,
            prompt: impl Into<String>,
        ) -> Self {
            Self {
                custom_id: custom_id.into(),
                method: "POST".to_string(),
                url: "/v1/responses".to_string(),
                body: serde_json::json!({
                    "model": model.into(),
                    "prompt": prompt.into(),
                    "max_tokens": 1000,
                }),
            }
        }
    }

    impl ChatClient {
        fn batches_url(&self) -> url::Url {
            self.base_url.join(&self.batches_path).unwrap()
        }

        /// Create a batch file from a list of batch request items.
        pub async fn create_batch_file(
            &self,
            filename: impl AsRef<str>,
            requests: &[BatchRequestItem],
        ) -> Result<String, BatchError> {
            let files_client = FilesClient::from(self);

            // Create a temporary file
            let temp_path = std::env::temp_dir().join(filename.as_ref());
            let mut file = std::fs::File::create(&temp_path)?;

            // Write each request as a JSON line
            for request in requests {
                let json = serde_json::to_string(request).unwrap(); // todo: remove unwrap
                writeln!(file, "{}", json)?;
            }
            file.flush()?;

            // Upload the file
            let file_obj = files_client
                .upload_file(&temp_path, FilePurpose::Batch)
                .await?;

            // Clean up the temporary file
            std::fs::remove_file(temp_path)?;

            Ok(file_obj.id)
        }

        /// Create a batch from a file ID.
        pub async fn create_batch(
            &self,
            input_file_id: impl AsRef<str>,
            endpoint: impl AsRef<str>,
        ) -> Result<Batch, BatchError> {
            let client = Client::new();
            let url = remove_trailing_slash(self.batches_url());
            let response = client
                .post(url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "input_file_id": input_file_id.as_ref(),
                    "endpoint": endpoint.as_ref(),
                    "completion_window": "24h"
                }))
                .send()
                .await?;

            let response_text = response.text().await?;
            let batch: Result<Batch, serde_json::Error> = serde_json::from_str(&response_text);

            match batch {
                Ok(batch) => Ok(batch),
                Err(e) => {
                    // Try to parse as an OpenAI error
                    let error: Result<OpenAiError, _> = serde_json::from_str(&response_text);
                    match error {
                        Ok(error) => Err(BatchError::OpenAiError(error)),
                        Err(_) => Err(BatchError::JsonParseError(e, response_text)),
                    }
                }
            }
        }

        /// Get the status of a batch.
        pub async fn get_batch_status(&self, batch_id: &str) -> Result<Batch, BatchError> {
            let client = Client::new();
            let url = self.batches_url().join(batch_id).unwrap();
            let response = client
                .get(url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .send()
                .await?;

            let response_text = response.text().await?;
            let batch: Result<Batch, serde_json::Error> = serde_json::from_str(&response_text);

            match batch {
                Ok(batch) => Ok(batch),
                Err(e) => {
                    // Try to parse as an OpenAI error
                    let error: Result<OpenAiError, _> = serde_json::from_str(&response_text);
                    match error {
                        Ok(error) => Err(BatchError::OpenAiError(error)),
                        Err(_) => Err(BatchError::JsonParseError(e, response_text)),
                    }
                }
            }
        }

        /// Wait for a batch to complete.
        pub async fn wait_for_batch(&self, batch_id: &str) -> Result<Batch, BatchError> {
            let mut attempts = 0;
            let max_attempts = 100; // Limit the number of attempts to avoid infinite loops

            loop {
                let batch = self.get_batch_status(batch_id).await?;

                match batch.status.as_str() {
                    "completed" => return Ok(batch),
                    "failed" => return Err(BatchError::BatchFailed(batch_id.to_string())),
                    "expired" => return Err(BatchError::BatchExpired(batch_id.to_string())),
                    "cancelled" => return Err(BatchError::BatchCancelled(batch_id.to_string())),
                    _ => {
                        // Still in progress, wait and try again
                        attempts += 1;
                        if attempts >= max_attempts {
                            return Err(BatchError::BatchTimeout(batch_id.to_string()));
                        }

                        // Exponential backoff with a cap
                        let delay = std::cmp::min(30, 2_u64.pow(attempts)) as u64;
                        sleep(Duration::from_secs(delay)).await;
                    }
                }
            }
        }

        /// Get the results of a batch.
        pub async fn get_batch_results(
            &self,
            batch: &Batch,
        ) -> Result<Vec<BatchResponseItem>, BatchError> {
            let files_client = FilesClient::from(self);

            if batch.status != "completed" {
                return Err(BatchError::Other(format!(
                    "Batch is not completed: {}",
                    batch.status
                )));
            }

            let output_file_id = batch
                .output_file_id
                .as_ref()
                .ok_or_else(|| BatchError::Other("Batch has no output file".to_string()))?;

            let content = files_client.download_file(output_file_id).await?;

            let mut results = Vec::new();
            for line in content.lines() {
                let result: BatchResponseItem = serde_json::from_str(line)
                    .map_err(|e| BatchError::JsonParseError(e, line.to_string()))?;
                results.push(result);
            }

            Ok(results)
        }

        /// Get the errors of a batch.
        pub async fn get_batch_errors(
            &self,
            batch: &Batch,
        ) -> Result<Vec<BatchResponseItem>, BatchError> {
            let files_client = FilesClient::from(self);

            let error_file_id = batch
                .error_file_id
                .as_ref()
                .ok_or_else(|| BatchError::Other("Batch has no error file".to_string()))?;

            let content = files_client.download_file(error_file_id).await?;

            let mut errors = Vec::new();
            for line in content.lines() {
                let error: BatchResponseItem = serde_json::from_str(line)
                    .map_err(|e| BatchError::JsonParseError(e, line.to_string()))?;
                errors.push(error);
            }

            Ok(errors)
        }

        /// Cancel a batch.
        pub async fn cancel_batch(&self, batch_id: &str) -> Result<Batch, BatchError> {
            let client = Client::new();
            let response = client
                .post(
                    self.batches_url()
                        .join(batch_id)
                        .unwrap()
                        .join("cancel")
                        .unwrap(),
                )
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .send()
                .await?;

            let response_text = response.text().await?;
            let batch: Result<Batch, serde_json::Error> = serde_json::from_str(&response_text);

            match batch {
                Ok(batch) => Ok(batch),
                Err(e) => {
                    // Try to parse as an OpenAI error
                    let error: Result<OpenAiError, _> = serde_json::from_str(&response_text);
                    match error {
                        Ok(error) => Err(BatchError::OpenAiError(error)),
                        Err(_) => Err(BatchError::JsonParseError(e, response_text)),
                    }
                }
            }
        }

        /// List all batches.
        pub async fn list_batches(
            &self,
            limit: Option<u32>,
            after: Option<&str>,
        ) -> Result<BatchList, BatchError> {
            let mut url = self.batches_url();

            // Add query parameters
            let mut query_params = Vec::new();
            if let Some(limit) = limit {
                query_params.push(format!("limit={}", limit));
            }
            if let Some(after) = after {
                query_params.push(format!("after={}", after));
            }
            if !query_params.is_empty() {
                url.set_query(Some(&query_params.join("&")));
            }

            let client = Client::new();
            let response = client
                .get(url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .send()
                .await?;

            let response_text = response.text().await?;
            let batch_list: Result<BatchList, serde_json::Error> =
                serde_json::from_str(&response_text);

            match batch_list {
                Ok(batch_list) => Ok(batch_list),
                Err(e) => {
                    // Try to parse as an OpenAI error
                    let error: Result<OpenAiError, _> = serde_json::from_str(&response_text);
                    match error {
                        Ok(error) => Err(BatchError::OpenAiError(error)),
                        Err(_) => Err(BatchError::JsonParseError(e, response_text)),
                    }
                }
            }
        }
    }
}

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
    pub base_url: url::Url,
    /// The subpath to the chat-completions endpoint. By default, this is `chat/completions`.
    pub chat_completions_path: String,
    /// The subpath to the batches endpoint. By default, this is `batches`.
    pub batches_path: String,
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

#[derive(Deserialize, Debug)]
enum ChatResponseOrError {
    #[serde(rename = "error")]
    Error(OpenAiError),

    #[serde(untagged)]
    Response(ChatResponse),
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
    /// OpenAI automatically caches tokens that are used in a previous request.
    /// This reduces input cost.
    pub cached_tokens: u32,
}

/// Includes details about the completion tokens for reasoning models
#[derive(Deserialize, Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct CompletionTokenDetails {
    /// The number of tokens used for reasoning.
    pub reasoning_tokens: u32,
    /// The number of accepted tokens from the reasoning model.
    pub accepted_prediction_tokens: u32,
    /// The number of rejected tokens from the reasoning model.
    /// (These tokens are still counted towards the cost of the request)
    pub rejected_prediction_tokens: u32,
}

impl std::ops::AddAssign for ChatUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.total_tokens += rhs.total_tokens;

        self.prompt_token_details = match (self.prompt_token_details, rhs.prompt_token_details) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (None, Some(rhs)) => Some(rhs),
            (Some(lhs), None) => Some(lhs),
            (None, None) => None,
        };
        self.completion_token_details =
            match (self.completion_token_details, rhs.completion_token_details) {
                (Some(lhs), Some(rhs)) => Some(lhs + rhs),
                (None, Some(rhs)) => Some(rhs),
                (Some(lhs), None) => Some(lhs),
                (None, None) => None,
            };
    }
}

impl std::ops::Add for PromptTokenDetails {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            cached_tokens: self.cached_tokens + rhs.cached_tokens,
        }
    }
}

impl std::ops::Add for CompletionTokenDetails {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            reasoning_tokens: self.reasoning_tokens + rhs.reasoning_tokens,
            accepted_prediction_tokens: self.accepted_prediction_tokens
                + rhs.accepted_prediction_tokens,
            rejected_prediction_tokens: self.rejected_prediction_tokens
                + rhs.rejected_prediction_tokens,
        }
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
    #[error("API returned an unknown response: {0} \nerror: {1} \nrequest: {2}")]
    ApiParseError(String, serde_json::Error, String),

    /// An error occurred when deserializing the response from the API.
    #[error("API returned an error response for request: {1}")]
    ApiError(#[source] OpenAiError, String),

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
            base_url: url::Url::parse("https://api.openai.com/v1/").unwrap(),
            chat_completions_path: "chat/completions".to_string(),
            batches_path: "batches/".to_string(),
            model: model.into(),
            lru: RwLock::new(LruCache::new(NonZeroUsize::new(1024).unwrap())),
            usage: RwLock::new(ChatUsage::default()),
        }
    }

    fn chat_completions_url(&self) -> url::Url {
        self.base_url.join(&self.chat_completions_path).unwrap()
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
    /// #     let my_api = url::Url::parse("https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/").unwrap();
    /// #     ChatClient {
    /// #         base_url: my_api,
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
            let chat_response: ChatResponseOrError = serde_json::from_str(&cached_response)
                .map_err(|e| {
                    ChatError::ApiParseError(cached_response.clone(), e, chat_request_str.clone())
                })?;
            match chat_response {
                ChatResponseOrError::Response(response) => response,
                ChatResponseOrError::Error(error) => {
                    return Err(ChatError::ApiError(error, chat_request_str));
                }
            }
        } else {
            let chat_response = self.chat_uncached(&chat_request).await?;
            let chat_response: ChatResponseOrError =
                serde_json::from_str(&chat_response).map_err(|e| {
                    ChatError::ApiParseError(chat_request_str.clone(), e, chat_request_str.clone())
                })?;
            let chat_response = match chat_response {
                ChatResponseOrError::Response(response) => response,
                ChatResponseOrError::Error(error) => {
                    return Err(ChatError::ApiError(error, chat_request_str));
                }
            };

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
            .post(self.chat_completions_url())
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

#[cfg(test)]
mod batch_tests {
    use super::batch::*;

    use serde_json::json;

    #[test]
    fn test_batch_request_serialization() {
        let request = BatchRequestItem {
            custom_id: "request-1".to_string(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body: json!({
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello world!"}
                ],
                "max_tokens": 1000
            }),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(serialized.contains("custom_id"));
        assert!(serialized.contains("request-1"));
        assert!(serialized.contains("method"));
        assert!(serialized.contains("POST"));
        assert!(serialized.contains("url"));
        assert!(serialized.contains("/v1/chat/completions"));
        assert!(serialized.contains("body"));
        assert!(serialized.contains("gpt-4o"));
        assert!(serialized.contains("helpful assistant"));
        assert!(serialized.contains("Hello world!"));
    }
}
