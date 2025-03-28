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
            embedding.dimension(),
            &embedding.elements[..5.min(embedding.dimension())]
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
        single_embedding.dimension(),
        &single_embedding.elements[..5.min(single_embedding.dimension())]
    );

    // Example of calculating similarity between embeddings
    if !embeddings.is_empty() && !single_embedding.elements.is_empty() {
        println!("\nCalculating cosine similarity between embeddings:");

        for (i, doc_embedding) in embeddings.iter().enumerate() {
            let similarity = doc_embedding.cosine_similarity(&single_embedding);
            println!(
                "Similarity between document {} and single document: {:.4}",
                i + 1,
                similarity
            );
        }
    }

    Ok(())
}
