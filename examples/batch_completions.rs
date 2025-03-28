use tysm::chat_completions::ChatClient;

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
        "What is the capital of Germany?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
        "What is the capital of Greece?",
        "What is the capital of Turkey?",
        "What is the capital of Russia?",
        "What is the capital of China?",
        "What is the capital of Brazil?",
        "What is the capital of Argentina?",
        "What is the capital of Chile?",
        "What is the capital of Mexico?",
        "What is the capital of Canada?",
        "What is the capital of Australia?",
        "What is the capital of New Zealand?",
    ];

    // May take up to 24 hours, because it's a batch request
    println!("Sending batch request... (this may take a while)");
    let responses = client.batch_chat::<Response>(requests).await?;

    println!("---");
    for response in responses {
        println!("{} is the capital of {}", response.city, response.country);
    }
    println!("---");

    println!("Batch request completed!");
    println!(
        "Run this example again, and tysm will automatically reuse the batch from the previous run to save time"
    );

    Ok(())
}
