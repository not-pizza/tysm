use std::sync::RwLock;

use lru::LruCache;
use reqwest::Client;
use schemars::{schema_for, transform::Transform, JsonSchema, Schema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingsRequest {
    model: String,
    input: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<Embedding>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
struct Embedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    total_tokens: u32,
}
use thiserror::Error;

use crate::utils::{api_key, OpenAiApiKeyError};

pub struct EmbeddingsClient {
    /// The API key to use for the ChatGPT API.
    pub api_key: String,
    /// The URL of the ChatGPT API. Customize this if you are using a custom API that is compatible with OpenAI's.
    pub url: String,
    /// The model to use for the ChatGPT API.
    pub model: String,
}

/// Errors that can occur when interacting with the ChatGPT API.
#[derive(Error, Debug)]
pub enum EmbeddingsError {
    /// An error occurred when sending the request to the API.
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when serializing the request to JSON.
    #[error("JSON serialization error: {0}")]
    JsonSerializeError(serde_json::Error, EmbeddingsRequest),

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

impl EmbeddingsClient {
    /// Create a new [`EmbeddingsClient`].
    /// If the API key is in the environment, you can use the [`Self::from_env`] method instead.
    ///
    /// ```rust
    /// use tysm::EmbeddingsClient;
    ///
    /// let client = EmbeddingsClient::new("sk-1234567890", "text-embedding-ada-002");
    /// ```
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        use std::num::NonZeroUsize;

        Self {
            api_key: api_key.into(),
            url: "https://api.openai.com/v1/embeddings".into(),
            model: model.into(),
        }
    }

    /// Create a new [`EmbeddingsClient`].
    /// This will use the `OPENAI_API_KEY` environment variable to set the API key.
    /// It will also look in the `.env` file for an `OPENAI_API_KEY` variable (using dotenv).
    ///
    /// ```rust
    /// # use tysm::EmbeddingsClient;
    /// let client = EmbeddingsClient::from_env("gpt-4o").unwrap();
    /// ```
    pub fn from_env(model: impl Into<String>) -> Result<Self, OpenAiApiKeyError> {
        Ok(Self::new(api_key()?, model))
    }

    pub async fn embed<T: DeserializeOwned + JsonSchema>(
        &self,
        prompt: impl Into<String>,
    ) -> Result<T, EmbeddingsError> {
        let request = EmbeddingsRequest {
            model: self.model.clone(),
            input: prompt.into(),
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| EmbeddingsError::JsonSerializeError(e, request))?;

        let client = Client::new();
        let response = client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let response_text = response.text().await?;
        
        let embeddings_response: EmbeddingsResponse = serde_json::from_str(&response_text)
            .map_err(|e| EmbeddingsError::ApiResponseError(e, response_text.clone(), request_json.clone()))?;

        // Convert the response to the requested type
        serde_json::from_value(serde_json::to_value(embeddings_response)?)
            .map_err(|e| EmbeddingsError::InvalidJson(e, response_text))
    }
}
