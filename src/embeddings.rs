//! Embeddings are a way to represent text in a vector space.
//! This module provides a client for interacting with the OpenAI Embeddings API.

use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct EmbeddingsRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<Embedding>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
enum EmbeddingsResponseOrError {
    #[serde(rename = "error")]
    Error(OpenAiError),
    #[serde(untagged)]
    Response(EmbeddingsResponse),
}

#[derive(Debug, Serialize, Deserialize)]
struct Embedding {
    embedding: Vec<f32>,
    index: usize,
}

/// A vector of floats. Returned as a result of embedding a document.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Vector {
    /// The elements of the vector
    pub elements: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    total_tokens: u32,
}
use thiserror::Error;

use crate::{
    utils::{api_key, OpenAiApiKeyError},
    OpenAiError,
};

/// A client for interacting with the OpenAI Embeddings API.
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

    /// An error occurred when deserializing the response from the API.
    #[error("API returned an unknown response: {0} \nerror: {1} \nrequest: {2}")]
    ApiParseError(String, serde_json::Error, String),

    /// An error occurred when deserializing the response from the API.
    #[error("API returned an error response for request: {1}")]
    ApiError(#[source] OpenAiError, String),

    /// The API returned a response that was not the expected JSON object.
    #[error("API returned a response that was not the expected JSON object: {0} \nresponse: {1}")]
    InvalidJson(serde_json::Error, String),

    /// The API did not return any choices.
    #[error("The wrong amount of embeddings was returned from API")]
    IncorrectNumberOfEmbeddings,
}

impl EmbeddingsClient {
    /// Create a new [`EmbeddingsClient`].
    /// If the API key is in the environment, you can use the [`Self::from_env`] method instead.
    ///
    /// ```rust
    /// use tysm::embeddings::EmbeddingsClient;
    ///
    /// let client = EmbeddingsClient::new("sk-1234567890", "text-embedding-ada-002");
    /// ```
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            url: "https://api.openai.com/v1/embeddings/".into(),
            model: model.into(),
        }
    }

    /// Create a new [`EmbeddingsClient`].
    /// This will use the `OPENAI_API_KEY` environment variable to set the API key.
    /// It will also look in the `.env` file for an `OPENAI_API_KEY` variable (using dotenv).
    ///
    /// ```rust
    /// # use tysm::embeddings::EmbeddingsClient;
    /// let client = EmbeddingsClient::from_env("gpt-4o").unwrap();
    /// ```
    pub fn from_env(model: impl Into<String>) -> Result<Self, OpenAiApiKeyError> {
        Ok(Self::new(api_key()?, model))
    }

    /// Embed a single document into a vector space.
    pub async fn embed_single(&self, document: String) -> Result<Vector, EmbeddingsError> {
        let embeddings = self.embed(vec![document]).await?;
        Ok(embeddings.first().unwrap().clone())
    }

    /// Embed documents into a vector space.
    /// Documents are processed in batches of 100 to stay within API limits.
    pub async fn embed(&self, documents: Vec<String>) -> Result<Vec<Vector>, EmbeddingsError> {
        const BATCH_SIZE: usize = 100;
        let documents_len = documents.len();
        let client = Client::new();
        let mut all_embeddings = Vec::with_capacity(documents_len);

        // Process documents in batches
        for chunk in documents.chunks(BATCH_SIZE) {
            let request = EmbeddingsRequest {
                model: self.model.clone(),
                input: chunk.to_vec(),
            };

            let response = client
                .post(&self.url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await?;

            let response_text = response.text().await?;

            let embeddings_response: EmbeddingsResponseOrError =
                serde_json::from_str(&response_text).map_err(|e| {
                    EmbeddingsError::ApiParseError(
                        response_text.clone(),
                        e,
                        serde_json::to_string(&request).unwrap(),
                    )
                })?;

            let embeddings_response = match embeddings_response {
                EmbeddingsResponseOrError::Response(response) => response,
                EmbeddingsResponseOrError::Error(error) => {
                    return Err(EmbeddingsError::ApiError(
                        error,
                        serde_json::to_string(&request).unwrap(),
                    ));
                }
            };

            if embeddings_response.data.len() != chunk.len() {
                return Err(EmbeddingsError::IncorrectNumberOfEmbeddings);
            }

            all_embeddings.extend(embeddings_response.data.into_iter().map(|e| Vector {
                elements: e.embedding,
            }));
        }

        Ok(all_embeddings)
    }
}

impl Vector {
    /// Calculate the cosine similarity between two vectors.
    pub fn cosine_similarity(&self, other: &Vector) -> f32 {
        /// implement, AI!
    }
}
