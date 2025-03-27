use std::env;
use std::error::Error;
use serde_json::json;
use tysm::chat_completions::{ChatClient, batch::BatchRequestItem};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load API key from environment
    dotenv::dotenv().ok();
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    
    // Create a client
    let client = ChatClient::new(api_key, "gpt-4o");
    let batch_client = client.batch();
    
    println!("Creating batch requests...");
    
    // Create batch requests
    let requests = vec![
        BatchRequestItem::new_chat(
            "request-1",
            "gpt-3.5-turbo",
            vec![
                json!({"role": "system", "content": "You are a helpful assistant."}),
                json!({"role": "user", "content": "What is the capital of France?"}),
            ],
        ),
        BatchRequestItem::new_chat(
            "request-2",
            "gpt-3.5-turbo",
            vec![
                json!({"role": "system", "content": "You are a helpful assistant."}),
                json!({"role": "user", "content": "What is the capital of Japan?"}),
            ],
        ),
        BatchRequestItem::new_chat(
            "request-3",
            "gpt-3.5-turbo",
            vec![
                json!({"role": "system", "content": "You are a helpful assistant."}),
                json!({"role": "user", "content": "What is the capital of Italy?"}),
            ],
        ),
    ];
    
    // Create a batch file
    println!("Creating batch file...");
    let file_id = batch_client.create_batch_file("capitals_batch.jsonl", &requests).await?;
    println!("Batch file created with ID: {}", file_id);
    
    // Create a batch
    println!("Creating batch...");
    let batch = batch_client.create_batch(file_id, "/v1/chat/completions").await?;
    println!("Batch created with ID: {}", batch.id);
    
    // Check batch status
    println!("Checking batch status...");
    let status = batch_client.get_batch_status(&batch.id).await?;
    println!("Batch status: {}", status.status);
    
    // Wait for batch to complete
    println!("Waiting for batch to complete...");
    println!("(This may take a while, up to 24 hours)");
    let completed_batch = batch_client.wait_for_batch(&batch.id).await?;
    println!("Batch completed!");
    
    // Get batch results
    println!("Getting batch results...");
    let results = batch_client.get_batch_results(&completed_batch).await?;
    
    // Display results
    println!("\nBatch Results:");
    println!("==============");
    for result in results {
        if let Some(response) = result.response {
            let content = response.body["choices"][0]["message"]["content"].as_str().unwrap_or("No content");
            println!("Result for {}: {}", result.custom_id, content);
        } else if let Some(error) = result.error {
            println!("Error for {}: {} - {}", result.custom_id, error.code, error.message);
        }
    }
    
    Ok(())
}
