use std::collections::HashMap;

use serde_json::json;
use tysm::{
    batch::{BatchClient, BatchRequestItem},
    chat_completions::{ChatClient, ChatMessage, ChatRequest, ResponseFormat},
};

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Response {
    city: String,
    country: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a client
    let client = ChatClient::from_env("gpt-4o").unwrap();

    // Create batch requests
    let requests = vec![
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Italy?",
    ];

    // May take up to 24 hours, because it's a batch request
    let responses = client.batch_chat::<Response>(requests).await?;

    for response in responses {
        println!("{} is the capital of {}", response.city, response.country);
    }

    Ok(())
}
