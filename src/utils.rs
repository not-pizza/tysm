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
