use std::path::Path;
use tysm::files::{FilePurpose, FilesClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client using the API key from environment variables
    let client = FilesClient::from_env()?;

    // Example 1: List existing files
    println!("Listing existing files:");
    let files = client.list_files().await?;

    if files.data.is_empty() {
        println!("No files found in your account.");
    } else {
        for file in &files.data {
            println!(
                "- {} (ID: {}, Size: {} bytes, Purpose: {})",
                file.filename, file.id, file.bytes, file.purpose
            );
        }
    }

    // Example 2: Create a temporary file to upload
    let temp_file_path = Path::new("example_data.jsonl");
    if !temp_file_path.exists() {
        println!("\nCreating a temporary file for upload...");
        let content = r#"{"prompt": "What is AI?", "completion": "Artificial Intelligence is the simulation of human intelligence processes by machines."}"#;
        std::fs::write(temp_file_path, content)?;
        println!("Created temporary file: {}", temp_file_path.display());
    }

    // Example 3: Upload a file
    println!("\nUploading a file:");
    let uploaded_file = client
        .upload_file(temp_file_path, FilePurpose::FineTune)
        .await?;
    println!(
        "File uploaded successfully: {} (ID: {})",
        uploaded_file.filename, uploaded_file.id
    );

    // Example 4: Retrieve file information
    println!("\nRetrieving file information:");
    let file_info = client.retrieve_file(&uploaded_file.id).await?;
    println!(
        "File info: {} (ID: {}, Size: {} bytes, Purpose: {})",
        file_info.filename, file_info.id, file_info.bytes, file_info.purpose
    );

    // Example 5: Upload content directly from bytes
    println!("\nUploading content from bytes:");
    let content = r#"{"prompt": "What is ML?", "completion": "Machine Learning is a subset of AI that enables systems to learn from data."}"#.as_bytes().to_vec();
    let uploaded_bytes = client
        .upload_bytes("bytes_example.jsonl", content, FilePurpose::FineTune)
        .await?;
    println!(
        "Bytes uploaded successfully: {} (ID: {})",
        uploaded_bytes.filename, uploaded_bytes.id
    );

    // Example 6: Download file content
    println!("\nDownloading file content:");
    let content = client.download_file(&uploaded_file.id).await?;
    println!("File content: {}", content);

    // Example 7: Delete files (cleanup)
    println!("\nCleaning up - deleting uploaded files:");

    let deleted_file = client.delete_file(&uploaded_file.id).await?;
    println!(
        "Deleted file: {} (Success: {})",
        deleted_file.id, deleted_file.deleted
    );

    let deleted_bytes = client.delete_file(&uploaded_bytes.id).await?;
    println!(
        "Deleted file: {} (Success: {})",
        deleted_bytes.id, deleted_bytes.deleted
    );

    // Clean up the temporary file
    if temp_file_path.exists() {
        std::fs::remove_file(temp_file_path)?;
        println!("Removed temporary file: {}", temp_file_path.display());
    }

    Ok(())
}
