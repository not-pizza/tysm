//! # `tysm` - Thank You So Much
//!
//! Typed OpenAI Chat Completions.
//!
//! A strongly-typed Rust client for OpenAI's ChatGPT API that enforces type-safe responses using [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs).
//!
//! This library uses the [schemars](https://docs.rs/schemars/latest/schemars/index.html) crate to generate a schema for the desired response type. It also uses [serde](https://docs.rs/serde/latest/serde/index.html) to deserialize the response into the desired type. Install these crates like so:
//!
//! 1. `cargo add serde`.
//! 2. `cargo add --git https://github.com/GREsau/schemars.git schemars`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use tysm::ChatClient;
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

mod chatgpt;
mod schema;

pub use chatgpt::ChatClient;
pub use chatgpt::ChatError;
pub use chatgpt::ChatMessage;
pub use chatgpt::ChatMessageContent;
pub use chatgpt::ChatRequest;
pub use chatgpt::ImageUrl;
pub use chatgpt::OpenAiApiKeyError;

#[cfg(test)]
mod tests {
    use crate::ChatClient;

    use std::sync::LazyLock;
    static CLIENT: LazyLock<ChatClient> = LazyLock::new(|| {
        let my_api = "https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/chat/completions".to_string();
        ChatClient {
            url: my_api,
            ..ChatClient::from_env("gpt-4o").unwrap()
        }
    });

    #[derive(serde::Deserialize, schemars::JsonSchema, Debug)]
    struct Name {
        first: String,
        last: String,
    }

    #[tokio::test]
    async fn it_works() {
        let name: Name = CLIENT.chat("Who was the first president?").await.unwrap();

        assert_eq!(name.first, "George");
        assert_eq!(name.last, "Washington");

        let usage1 = CLIENT.usage();
        for _ in 0..5 {
            let _name: Name = CLIENT.chat("Who was the first president?").await.unwrap();
        }
        let usage2 = CLIENT.usage();
        assert_eq!(usage1, usage2);
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
