//! Batch API for processing large numbers of requests asynchronously.
//!
//! The Batch API allows you to send asynchronous groups of requests with 50% lower costs,
//! a separate pool of significantly higher rate limits, and a clear 24-hour turnaround time.
//!
//! # Example
//!
//! ```rust,no_run
//! # use tysm::{batch::{BatchClient, BatchRequestItem}, chat_completions::ChatClient};
//! # use serde_json::json;
//! # use tokio_test::block_on;
//! # use std::collections::HashMap;
//! # block_on(async {
//! let client = ChatClient::from_env("gpt-4o").unwrap();
//! let batch_client = BatchClient::from(&client);
//!
//! // Create batch requests
//! let requests = vec![
//!     BatchRequestItem::new_chat(
//!         "request-1",
//!         "gpt-4o",
//!         vec![
//!             json!({"role": "system", "content": "You are a helpful assistant."}),
//!             json!({"role": "user", "content": "Hello world!"}),
//!         ],
//!     ),
//!     BatchRequestItem::new_chat(
//!         "request-2",
//!         "gpt-4o",
//!         vec![
//!             json!({"role": "system", "content": "You are an unhelpful assistant."}),
//!             json!({"role": "user", "content": "Hello world!"}),
//!         ],
//!     ),
//! ];
//!
//! // Create a batch file
//! let file_id = batch_client.create_batch_file("my_batch.jsonl", &requests).await.unwrap();
//!
//! // Create a batch
//! let batch = batch_client.create_batch(file_id, /* metadata */ HashMap::new()).await.unwrap();
//! println!("Batch created: {}", batch.id);
//!
//! // Check batch status
//! let status = batch_client.get_batch_status(&batch.id).await.unwrap();
//! println!("Batch status: {}", status.status);
//!
//! // Wait for batch to complete
//! let completed_batch = batch_client.wait_for_batch(&batch.id).await.unwrap();
//!
//! // Get batch results
//! let results = batch_client.get_batch_results(&completed_batch).await.unwrap();
//! for result in results {
//!     println!("Result for {}: {}", result.custom_id, result.response.unwrap().body);
//! }
//! # });
//! ```

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;

use std::time::Duration;
use thiserror::Error;
use tokio::time::sleep;

use crate::chat_completions::{ChatClient, ChatError};
use crate::files::{FilePurpose, FilesClient, FilesError};
use crate::utils::remove_trailing_slash;
use crate::OpenAiError;

/// A client for batching requests to the OpenAI API.
pub struct BatchClient {
    /// The API key to use for the ChatGPT API.
    pub api_key: String,
    /// The URL of the ChatGPT API. Customize this if you are using a custom API that is compatible with OpenAI's.
    pub base_url: url::Url,
    /// The subpath to the batches endpoint. By default, this is `batches`.
    pub batches_path: String,
    /// the endpoint whose calls we want to batch
    pub endpoint: String,
    /// The model to use for the ChatGPT API.
    pub model: String,
    /// The client to use for file operations.
    pub files_client: FilesClient,
}

impl From<&ChatClient> for BatchClient {
    fn from(client: &ChatClient) -> Self {
        Self {
            api_key: client.api_key.clone(),
            base_url: client.base_url.clone(),
            batches_path: "batches/".to_string(),
            endpoint: "/v1/chat/completions".to_string(),
            model: client.model.clone(),
            files_client: FilesClient::from(client),
        }
    }
}

/// Errors that can occur when interacting with the Batch API.
#[derive(Error, Debug)]
pub enum BatchError {
    /// An error occurred when sending the request to the API.
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when deserializing JSON.
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
    pub metadata: Option<HashMap<String, String>>,
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

impl BatchClient {
    fn batches_url(&self) -> url::Url {
        self.base_url.join(&self.batches_path).unwrap()
    }

    // I'd rather this return Vec<u8> rather than actually writing to a file, AI!
    /// Create a batch file from a list of batch request items.
    pub async fn create_batch_file(
        &self,
        filename: impl AsRef<str>,
        requests: &[BatchRequestItem],
    ) -> Result<String, BatchError> {
        // Create a temporary file
        let temp_path = std::env::temp_dir().join(filename.as_ref());
        let mut file = std::fs::File::create(&temp_path)?;

        // Write each request as a JSON line
        for request in requests {
            let json = serde_json::to_string(request).unwrap(); // cannot panic
            writeln!(file, "{}", json)?;
        }
        file.flush()?;

        // Upload the file
        let file_obj = self
            .files_client
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
        metadata: HashMap<String, String>,
    ) -> Result<Batch, BatchError> {
        let client = Client::new();
        let url = remove_trailing_slash(self.batches_url());
        let response = client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "input_file_id": input_file_id.as_ref(),
                "endpoint": &self.endpoint,
                "completion_window": "24h",
                "metadata": metadata,
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
        let mut seconds_waited = 0;

        loop {
            let batch = self.get_batch_status(batch_id).await?;

            match batch.status.as_str() {
                "completed" => return Ok(batch),
                "failed" => return Err(BatchError::BatchFailed(batch_id.to_string())),
                "expired" => return Err(BatchError::BatchExpired(batch_id.to_string())),
                "cancelled" => return Err(BatchError::BatchCancelled(batch_id.to_string())),
                _ => {
                    attempts += 1;
                    // Still in progress, wait and try again
                    if seconds_waited >= 86400 {
                        return Err(BatchError::BatchTimeout(batch_id.to_string()));
                    }

                    // Exponential backoff with a cap
                    let delay = std::cmp::min(120, 2_u64.pow(attempts)) as u64;
                    sleep(Duration::from_secs(delay)).await;
                    seconds_waited += delay;
                }
            }
        }
    }

    /// Get the results of a batch.
    pub async fn get_batch_results(
        &self,
        batch: &Batch,
    ) -> Result<Vec<BatchResponseItem>, BatchError> {
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

        let content = self.files_client.download_file(output_file_id).await?;

        let mut results = Vec::new();
        for line in content.lines() {
            let result: BatchResponseItem = serde_json::from_str(line)
                .map_err(|e| BatchError::JsonParseError(e, content.clone()))?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get the errors of a batch.
    pub async fn get_batch_errors(
        &self,
        batch: &Batch,
    ) -> Result<Vec<BatchResponseItem>, BatchError> {
        let error_file_id = batch
            .error_file_id
            .as_ref()
            .ok_or_else(|| BatchError::Other("Batch has no error file".to_string()))?;

        let content = self.files_client.download_file(error_file_id).await?;

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
    ///
    /// This method will automatically handle pagination by repeatedly calling
    /// `list_batches_limited` until all batches have been retrieved.
    pub async fn list_batches(&self) -> Result<Vec<Batch>, BatchError> {
        let mut all_batches = Vec::new();
        let mut last_batch_id = None;

        loop {
            let batch_list = self
                .list_batches_limited(None, last_batch_id.as_deref())
                .await?;

            if batch_list.data.is_empty() {
                break;
            }

            // Get the ID of the last batch for pagination
            if let Some(last_batch) = batch_list.data.last() {
                last_batch_id = Some(last_batch.id.clone());
            }

            all_batches.extend(batch_list.data);

            // If there are no more batches, break
            if !batch_list.has_more {
                break;
            }
        }

        Ok(all_batches)
    }

    /// List all batches.
    async fn list_batches_limited(
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
            .get(remove_trailing_slash(url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await?;

        let response_text = response.text().await?;
        let batch_list: Result<BatchList, serde_json::Error> = serde_json::from_str(&response_text);

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

#[test]
fn test_batch_request_serialization() {
    use serde_json::json;
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
