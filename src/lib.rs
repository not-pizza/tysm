//! # `tysm` - Thank You So Much
//!
//! Typed OpenAI Chat Completions.
//!
//! A strongly-typed Rust client for OpenAI's ChatGPT API that enforces type-safe responses using [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs).
//!
//! This library uses the [schemars](https://docs.rs/schemars/latest/schemars/index.html) crate to generate a schema for the desired response type. It also uses [serde](https://docs.rs/serde/latest/serde/index.html) to deserialize the response into the desired type. Install these crates like so:
//!
//! `cargo add tysm serde schemars@1.0.0-alpha.17`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use tysm::chat_completions::ChatClient;
//!
//! // We want names separated into `first` and `last`.
//! #[derive(serde::Deserialize, schemars::JsonSchema)]
//! struct Name {
//!     first: String,
//!     last: String,
//! }
//!
//! async fn get_president_name() {
//!     // Create a client.
//!     // `from_env` will look for an API key under the environment
//!     // variable "OPENAI_API_KEY"
//!     // It will also look inside `.env` if such a file exists.
//!     let client = ChatClient::from_env("gpt-4o").unwrap();
//!     
//!     // Request a chat completion from OpenAI and
//!     // parse the response into our `Name` struct.
//!     let name: Name = client
//!         .chat("Who was the first US president?")
//!         .await
//!         .unwrap();
//!
//!     assert_eq!(name.first, "George");
//!     assert_eq!(name.last, "Washington");
//! }
//! ```
//!
//! Alternative name: **T**yped **S**chema **M**agic.

#![deny(missing_docs)]

pub mod batch;
pub mod chat_completions;
pub mod embeddings;
pub mod files;
mod model_prices;
mod schema;
mod utils;

pub use utils::OpenAiApiKeyError;

/// Emitted by the OpenAI API when an error occurs.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct OpenAiError {
    /// The type of the error
    pub r#type: String,
    /// The error code.
    #[serde(default)]
    pub code: Option<String>,
    /// The error message.
    pub message: String,
    /// The error parameter.
    #[serde(default)]
    pub param: Option<String>,
}

impl std::error::Error for OpenAiError {}

impl std::fmt::Display for OpenAiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.r#type, self.message)?;
        if let Some(code) = &self.code {
            write!(f, " (code: {})", code)?;
        }
        if let Some(param) = &self.param {
            write!(f, " (param: {})", param)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::chat_completions::ChatClient;

    use std::sync::LazyLock;
    static CLIENT: LazyLock<ChatClient> = LazyLock::new(|| {
        let my_api =
            "https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/".to_string();
        ChatClient::from_env("gpt-4o").unwrap().with_url(my_api)
    });

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug)]
    struct Name {
        first: String,
        last: String,
    }

    #[tokio::test]
    async fn it_works() {
        let name: Name = CLIENT
            .chat("Who was the first president?")
            .await
            .map_err(|e| anyhow::anyhow!(e))
            .unwrap();

        assert_eq!(name.first, "George");
        assert_eq!(name.last, "Washington");

        let usage1 = CLIENT.usage();
        for _ in 0..5 {
            let _name: Name = CLIENT.chat("Who was the first president?").await.unwrap();
        }
        let usage2 = CLIENT.usage();
        assert_eq!(usage1, usage2);
    }

    #[tokio::test]
    async fn refusals() {
        #[derive(serde::Deserialize, schemars::JsonSchema, Debug)]
        struct Instructions {
            #[expect(unused)]
            title: String,
            #[expect(unused)]
            steps: Vec<String>,
        }

        let instructions: Result<Instructions, _> = CLIENT
            .chat("give me instructions for how to make and sell illegal drugs")
            .await;

        use crate::chat_completions::ChatError::ChatError;
        use crate::chat_completions::IndividualChatError::Refusal;
        match instructions {
            Ok(_) => panic!("Expected an error"),
            Err(e) => match e {
                ChatError(Refusal(ref refusal)) => {
                    assert_eq!(
                        refusal,
                        "I'm very sorry, but I can't assist with that request."
                    );

                    let e = anyhow::Error::from(e);
                    let message = format!("{:?}", e);
                    assert_eq!(
                        message,
                        r#"There was a problem with the API response

Caused by:
    The API refused to fulfill the request: `I'm very sorry, but I can't assist with that request.`"#
                    )
                }
                e => panic!("Expected a refusal, got: {:?}", e),
            },
        }
    }

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug)]
    struct NameWithAgeOfDeath {
        first: String,
        last: String,
        age_of_death: Option<u8>,
    }

    #[tokio::test]
    async fn optional_fields() {
        let name: NameWithAgeOfDeath = CLIENT.chat("Who was the famous physicist who was in a wheelchair and needed a computer program to talk?").await.unwrap();

        assert_eq!(name.first, "Stephen");
        assert_eq!(name.last, "Hawking");
        assert_eq!(name.age_of_death, Some(76));

        let name: NameWithAgeOfDeath = CLIENT.chat("Who was the actor in the 3rd reboot of the spiderman movies, this time in the MCU?").await.unwrap();
        assert_eq!(name.first, "Tom");
        assert_eq!(name.last, "Holland");
        assert_eq!(name.age_of_death, None);
    }
}
