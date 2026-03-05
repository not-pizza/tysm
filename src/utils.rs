/// An error that occurs when the OpenAI API key is not found in the environment.
#[derive(Debug)]
pub struct OpenAiApiKeyError(#[expect(unused)] std::env::VarError);
impl std::fmt::Display for OpenAiApiKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unable to find the OpenAI API key in the environment. Please set the OPENAI_API_KEY environment variable. API keys can be found at <https://platform.openai.com/api-keys>.")
    }
}
impl std::error::Error for OpenAiApiKeyError {}

pub(crate) fn api_key() -> Result<String, OpenAiApiKeyError> {
    #[cfg(feature = "dotenvy")]
    {
        use dotenvy::dotenv;
        dotenv().ok();
    }
    std::env::var("OPENAI_API_KEY").map_err(OpenAiApiKeyError)
}

pub(crate) fn remove_trailing_slash(url: url::Url) -> url::Url {
    let mut url = url;
    let path = url.path().to_string();
    let path = path.trim_end_matches('/');
    url.set_path(path);
    url
}

/// Create a shared reqwest::Client with connection pooling configured for high concurrency.
pub(crate) fn pooled_client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_max_idle_per_host(256)
        .pool_idle_timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("Failed to build HTTP client")
}

/// Compute a shard subdirectory name from a cache key.
///
/// This distributes cache files across 1000 subdirectories so that no single
/// directory tree object becomes too large for GitHub.
pub(crate) fn cache_shard(cache_key: &str) -> String {
    use xxhash_rust::const_xxh3::xxh3_64 as const_xxh3;
    let hash = const_xxh3(cache_key.as_bytes());
    format!("{:03}", hash % 1000)
}

/// Resolve a cache file, checking (in order):
/// 1. Sharded path in `dir` (e.g. `dir/042/key.zstd`)
/// 2. Flat path in `dir` (e.g. `dir/key.zstd`)
///
/// If found in the flat location, moves it to the sharded location.
/// Returns the data if found, or `None`.
pub(crate) async fn read_from_cache_dir(dir: &std::path::Path, cache_key: &str) -> Option<Vec<u8>> {
    let shard = cache_shard(cache_key);
    let sharded_path = dir.join(&shard).join(cache_key);

    // 1. Try sharded path
    if let Ok(data) = tokio::fs::read(&sharded_path).await {
        return Some(data);
    }

    // 2. Try flat path (old layout) and migrate if found
    let flat_path = dir.join(cache_key);
    if let Ok(data) = tokio::fs::read(&flat_path).await {
        // Migrate to sharded location
        let shard_dir = dir.join(&shard);
        let _ = tokio::fs::create_dir_all(&shard_dir).await;
        let _ = tokio::fs::rename(&flat_path, &sharded_path).await;
        return Some(data);
    }

    None
}

/// Write a cache file to the sharded location within `dir`.
pub(crate) async fn write_to_cache_dir(
    dir: &std::path::Path,
    cache_key: &str,
    data: &[u8],
) -> Result<(), std::io::Error> {
    let shard = cache_shard(cache_key);
    let shard_dir = dir.join(&shard);
    tokio::fs::create_dir_all(&shard_dir).await?;
    let sharded_path = shard_dir.join(cache_key);
    tokio::fs::write(&sharded_path, data).await
}
