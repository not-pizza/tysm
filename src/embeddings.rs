//! Embeddings are a way to represent text in a vector space.
//! This module provides a client for interacting with the OpenAI Embeddings API.

use std::path::PathBuf;
use std::sync::RwLock;

use lru::LruCache;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use xxhash_rust::const_xxh3::xxh3_64 as const_xxh3;

#[derive(Debug, Serialize, Clone)]
struct EmbeddingsRequest<'a> {
    model: String,
    input: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

impl EmbeddingsRequest<'_> {
    fn cache_key(&self, url: &url::Url) -> String {
        let key = format!("{} /:/ {}", serde_json::to_string(&self).unwrap(), url);
        let key = key.as_bytes();
        let id = const_xxh3(key);
        format!("tysm-v1-embeddings_request-{}.zstd", id)
    }
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
    pub base_url: url::Url,
    /// The subpath to the chat-completions endpoint. By default, this is `embeddings`.
    pub embeddings_path: String,
    /// The model to use for the ChatGPT API.
    pub model: String,
    /// The number of documents to send in a single batch.
    pub batch_size: usize,
    /// Some embedding models are trained using a technique that allows them to have their dimensionality lowered without the embedding losing its concept-representing properties. Of OpenAI's models, only text-embedding-3 and later models support this functionality.
    pub dimensions: Option<usize>,
    /// A cache of the few responses. Stores the last 1024 responses by default.
    pub lru: RwLock<LruCache<String, String>>,
    /// The directory in which to cache responses to requests
    pub cache_directory: Option<PathBuf>,
}

/// Errors that can occur when interacting with the ChatGPT API.
#[derive(Error, Debug)]
pub enum EmbeddingsError {
    /// An error occurred when sending the request to the API.
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when deserializing the response from the API.
    #[error("API {url} returned an unknown response: {response} | request: {request}")]
    ApiParseError {
        /// The URL of the API that returned the error.
        url: String,
        /// The response from the API.
        response: String,
        /// The request that was sent to the API.
        request: String,
        /// The error that occurred when deserializing the response.
        #[source]
        error: serde_json::Error,
    },

    /// An error occurred when deserializing the response from the API.
    #[error("API returned an error response for request: {1}")]
    ApiError(#[source] OpenAiError, String),

    /// The API returned a response that was not the expected JSON object.
    #[error("API returned a response that was not the expected JSON object: {0} | response: {1}")]
    InvalidJson(serde_json::Error, String),

    /// The API did not return any choices.
    #[error("The wrong amount of embeddings was returned from API")]
    IncorrectNumberOfEmbeddings,

    /// IO error (usually occurs when reading from the cache).
    #[error("IO error")]
    IoError(#[from] std::io::Error),
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
        use std::num::NonZeroUsize;

        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1/".parse().unwrap(),
            embeddings_path: "embeddings".into(),
            model: model.into(),
            batch_size: 500,
            dimensions: None,
            lru: RwLock::new(LruCache::new(NonZeroUsize::new(1024).unwrap())),
            cache_directory: None,
        }
    }

    /// Sets the number of documents to send in a single batch.
    /// The default batch size is 500. If you have large documents, you may want to set the batch size to a lower value.
    pub fn with_batch_size(self, batch_size: usize) -> Self {
        Self { batch_size, ..self }
    }

    /// Set the cache directory for the client.
    ///
    /// The cache directory will be used to persistently cache all responses to requests.
    pub fn with_cache_directory(mut self, cache_directory: impl Into<PathBuf>) -> Self {
        let cache_directory = cache_directory.into();

        if cache_directory.exists() && cache_directory.is_file() {
            panic!("Cache directory is a file");
        }

        self.cache_directory = Some(cache_directory);
        self
    }

    /// Sets the base URL
    ///
    /// Panics if the argument is not a valid URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        let url = if url.ends_with('/') {
            url
        } else {
            format!("{}/", url)
        };

        let url = url::Url::parse(&url).unwrap();
        self.base_url = url;
        self
    }

    /// Sets the path to the embeddings endpoint.
    ///
    /// By default, this is `embeddings`.
    pub fn with_path(self, path: impl Into<String>) -> Self {
        Self {
            embeddings_path: path.into(),
            ..self
        }
    }

    /// Sets the number of dimensions the embeddings should have.
    ///
    /// Some embedding models are trained using a technique that allows them to have their dimensionality lowered without the embedding losing its concept-representing properties. Of OpenAI's models, only text-embedding-3 and later models support this functionality.
    pub fn with_dimensions(self, dimensions: usize) -> Self {
        Self {
            dimensions: Some(dimensions),
            ..self
        }
    }

    fn embeddings_url(&self) -> url::Url {
        self.base_url.join(&self.embeddings_path).unwrap()
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

    /// Embed a single document into vector space.
    pub async fn embed_single(&self, document: String) -> Result<Vector, EmbeddingsError> {
        let documents = &[document];
        let embeddings = self.embed(documents).await?;
        Ok(embeddings.first().unwrap().1.clone())
    }

    /// Embed documents into vector space.
    ///
    /// Documents are processed in batches to stay within API limits
    pub async fn embed<'a>(
        &self,
        documents: &'a [String],
    ) -> Result<Vec<(&'a String, Vector)>, EmbeddingsError> {
        self.embed_fn(documents, |s| s).await
    }

    /// Embed documents into vector space. A function can be provided to map the documents to strings.
    ///
    /// Documents are processed in batches to stay within API limits.
    pub async fn embed_fn<'a, T: Ord + std::fmt::Debug, S: AsRef<str>>(
        &self,
        documents: &'a [T],
        f: impl Fn(&'a T) -> S,
    ) -> Result<Vec<(&'a T, Vector)>, EmbeddingsError> {
        let documents_len = documents.len();
        let mut all_embeddings = Vec::with_capacity(documents_len);

        // Create indexed documents with their string representations
        let mut indexed_documents: Vec<(usize, &'a T, String)> = documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx, doc, f(doc).as_ref().to_string()))
            .collect();

        // Sort by the document content for stable caching
        indexed_documents.sort_by(|a, b| a.2.cmp(&b.2));

        // Create smart chunks based on document hashes
        let mut chunks: Vec<Vec<(usize, &'a T, &str)>> = Vec::new();
        let mut current_chunk: Vec<(usize, &'a T, &str)> = Vec::new();

        for (idx, doc, doc_str) in indexed_documents.iter() {
            // Calculate hash of the document's debug representation
            let debug_repr = format!("{:?}", doc);
            let hash = const_xxh3(debug_repr.as_bytes());
            
            // Check if we should start a new chunk (1/256 chance or chunk is full)
            let should_split = (hash % 256) == 0 && !current_chunk.is_empty();
            let chunk_full = current_chunk.len() >= self.batch_size;
            
            if should_split || chunk_full {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk);
                    current_chunk = Vec::new();
                }
            }
            
            current_chunk.push((*idx, *doc, doc_str.as_str()));
        }
        
        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        // Process each chunk
        for chunk in chunks {
            let documents_len = chunk.len();
            let input_docs: Vec<&str> = chunk.iter().map(|(_, _, s)| *s).collect();
            
            let request = EmbeddingsRequest {
                model: self.model.clone(),
                input: input_docs,
                dimensions: self.dimensions,
            };

            // Check cache first
            let response_text =
                if let Some(cached_response) = self.embeddings_cached(&request).await {
                    cached_response?
                } else {
                    let response_text = self.embeddings_uncached(&request).await?;
                    self.cache_embeddings_response(&request, &response_text)
                        .await?;

                    response_text
                };

            let embeddings_response: EmbeddingsResponseOrError =
                serde_json::from_str(&response_text).map_err(|e| {
                    EmbeddingsError::ApiParseError {
                        url: self.embeddings_url().to_string(),
                        response: response_text.clone(),
                        request: serde_json::to_string(&request).unwrap(),
                        error: e,
                    }
                })?;

            let embeddings_response = match embeddings_response {
                EmbeddingsResponseOrError::Response(response) => response,
                EmbeddingsResponseOrError::Error(error) => {
                    let request_str = serde_json::to_string(&request).unwrap();
                    let request_str = if request_str.len() > 100 {
                        request_str.chars().take(100).chain("...".chars()).collect()
                    } else {
                        request_str
                    };
                    return Err(EmbeddingsError::ApiError(error, request_str));
                }
            };

            if embeddings_response.data.len() != documents_len {
                return Err(EmbeddingsError::IncorrectNumberOfEmbeddings);
            }

            // Store embeddings with their original indices
            for ((original_idx, doc, _), embedding) in chunk.into_iter().zip(embeddings_response.data) {
                all_embeddings.push((original_idx, doc, Vector {
                    elements: embedding.embedding,
                }));
            }
        }

        // Sort by original index to maintain input order
        all_embeddings.sort_by_key(|(idx, _, _)| *idx);
        
        // Return documents with their embeddings in original order
        Ok(all_embeddings.into_iter().map(|(_, doc, vec)| (doc, vec)).collect())
    }

    async fn embeddings_cached(
        &self,
        request: &EmbeddingsRequest<'_>,
    ) -> Option<Result<String, EmbeddingsError>> {
        let cache_key = request.cache_key(&self.embeddings_url());
        let request_str = serde_json::to_string(request).ok()?;

        // First, check the lru (which we just peek so it's not even really used as a LRU)
        {
            let lru = self.lru.read().ok()?;
            let response = lru.peek(&request_str);
            if let Some(response) = response {
                return Some(Ok(response.clone()));
            }
        }

        // Then, check the cache directory
        let cache_directory = self.cache_directory.as_ref()?;
        if !cache_directory.exists() {
            panic!(
                "Cache directory does not exist: {}",
                cache_directory.display()
            );
        }
        let cache_path = cache_directory.join(cache_key);

        // Read the compressed data from disk
        let compressed_data = tokio::fs::read(&cache_path).await.ok()?;

        // Decompress the data
        let decompressed_data = zstd::decode_all(compressed_data.as_slice()).ok()?;

        // Convert bytes back to string
        let response = String::from_utf8(decompressed_data).ok()?;

        Some(Ok(response))
    }

    async fn embeddings_uncached(
        &self,
        request: &EmbeddingsRequest<'_>,
    ) -> Result<String, EmbeddingsError> {
        let client = Client::new();

        let response = client
            .post(self.embeddings_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?
            .text()
            .await?;

        Ok(response)
    }

    async fn cache_embeddings_response(
        &self,
        request: &EmbeddingsRequest<'_>,
        response: &str,
    ) -> Result<(), EmbeddingsError> {
        let cache_key = request.cache_key(&self.embeddings_url());
        let request_str = serde_json::to_string(request).unwrap();

        if let Some(cache_directory) = &self.cache_directory {
            if !cache_directory.exists() {
                tokio::fs::create_dir_all(&cache_directory).await?;
            }

            let cache_path = cache_directory.join(cache_key);

            // Compress the response with zstd before writing to disk
            let compressed = zstd::encode_all(response.as_bytes(), 3)?;
            tokio::fs::write(&cache_path, compressed).await?;
        }

        self.lru
            .write()
            .ok()
            .unwrap()
            .put(request_str, response.to_string());

        Ok(())
    }
}

impl Vector {
    /// Calculate the cosine similarity between two vectors.
    ///
    /// Panics if the vectors have different dimensions.
    pub fn cosine_similarity(&self, other: &Vector) -> f32 {
        if self.elements.len() != other.elements.len() {
            panic!("Cannot calculate cosine similarity between vectors of different dimensions");
        }

        let dot_product = self.dot_product(other);
        let magnitude_a = self.magnitude();
        let magnitude_b = other.magnitude();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
    }

    /// Calculate the Euclidean distance between two vectors.
    ///
    /// Panics if the vectors have different dimensions.
    pub fn euclidean_distance(&self, other: &Vector) -> f32 {
        if self.elements.len() != other.elements.len() {
            panic!("Cannot calculate Euclidean distance between vectors of different dimensions");
        }

        self.elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Create a normalized (unit) vector with the same direction.
    pub fn normalize(&self) -> Self {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            return self.clone();
        }

        Vector {
            elements: self.elements.iter().map(|x| x / magnitude).collect(),
        }
    }

    /// Calculate the dot product of two vectors.
    ///
    /// Panics if the vectors have different dimensions.
    pub fn dot_product(&self, other: &Vector) -> f32 {
        if self.elements.len() != other.elements.len() {
            panic!("Cannot calculate dot product between vectors of different dimensions");
        }

        self.elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Add another vector to this one.
    ///
    /// Panics if the vectors have different dimensions.
    pub fn add(&self, other: &Vector) -> Self {
        if self.elements.len() != other.elements.len() {
            panic!("Cannot add vectors of different dimensions");
        }

        Vector {
            elements: self
                .elements
                .iter()
                .zip(other.elements.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Subtract another vector from this one.
    ///
    /// Panics if the vectors have different dimensions.
    pub fn subtract(&self, other: &Vector) -> Self {
        if self.elements.len() != other.elements.len() {
            panic!("Cannot subtract vectors of different dimensions");
        }

        Vector {
            elements: self
                .elements
                .iter()
                .zip(other.elements.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Multiply this vector by a scalar.
    pub fn scale(&self, scalar: f32) -> Self {
        Vector {
            elements: self.elements.iter().map(|x| x * scalar).collect(),
        }
    }

    /// Calculate the magnitude (length) of the vector.
    pub fn magnitude(&self) -> f32 {
        self.elements.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Get the dimension (number of elements) of the vector.
    pub fn dimension(&self) -> usize {
        self.elements.len()
    }

    /// Truncate the vector to the given number of dimensions.
    pub fn truncate(&self, dimensions: usize) -> Self {
        Self {
            elements: self.elements.iter().take(dimensions).copied().collect(),
        }
    }
}
