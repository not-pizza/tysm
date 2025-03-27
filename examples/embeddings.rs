use tysm::embeddings::EmbeddingsClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a client using the API key from environment variables
    let client = EmbeddingsClient::from_env("text-embedding-3-small")?;

    // Example documents to embed
    let documents = vec![
        "Artificial intelligence is transforming how we interact with technology.".to_string(),
        "Machine learning models can recognize patterns in large datasets.".to_string(),
        "Natural language processing helps computers understand human language.".to_string(),
    ];

    println!("Embedding multiple documents:");
    // Embed multiple documents
    let embeddings = client.embed(documents.clone()).await?;

    // Print information about the embeddings
    for (i, embedding) in embeddings.iter().enumerate() {
        println!(
            "Document {}: \"{}\" -> Vector with {} dimensions (showing first 5: {:?}...)",
            i + 1,
            documents[i],
            embedding.len(),
            &embedding[..5.min(embedding.len())]
        );
    }

    println!("\nEmbedding a single document:");
    // Embed a single document
    let single_document =
        "Vector databases store and query high-dimensional vectors efficiently.".to_string();
    let single_embedding = client.embed_single(single_document.clone()).await?;

    println!(
        "Single document: \"{}\" -> Vector with {} dimensions (showing first 5: {:?}...)",
        single_document,
        single_embedding.len(),
        &single_embedding[..5.min(single_embedding.len())]
    );

    // Example of calculating similarity between embeddings
    if !embeddings.is_empty() && !single_embedding.is_empty() {
        println!("\nCalculating cosine similarity between embeddings:");

        for (i, doc_embedding) in embeddings.iter().enumerate() {
            let similarity = cosine_similarity(doc_embedding, &single_embedding);
            println!(
                "Similarity between document {} and single document: {:.4}",
                i + 1,
                similarity
            );
        }
    }

    Ok(())
}

// Helper function to calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}
