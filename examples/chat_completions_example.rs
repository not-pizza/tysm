use schemars::JsonSchema;
use serde::Deserialize;
use tysm::chat_completions::ChatClient;

#[derive(Deserialize, Debug, JsonSchema)]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    instructions: Vec<String>,
    prep_time_minutes: u32,
    cook_time_minutes: u32,
    difficulty: String,
}

#[derive(Deserialize, Debug, JsonSchema)]
struct MovieRecommendation {
    title: String,
    year: u32,
    director: String,
    genre: String,
    brief_summary: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client using the API key from environment variables
    let client = ChatClient::from_env("gpt-4o")?;
    
    println!("Example 1: Getting a structured recipe");
    // Get a structured recipe response
    let recipe: Recipe = client
        .chat("Give me a recipe for a quick vegetarian pasta dish")
        .await?;
    
    println!("Recipe: {}", recipe.name);
    println!("Difficulty: {}", recipe.difficulty);
    println!("Prep time: {} minutes", recipe.prep_time_minutes);
    println!("Cook time: {} minutes", recipe.cook_time_minutes);
    
    println!("\nIngredients:");
    for ingredient in recipe.ingredients {
        println!("- {}", ingredient);
    }
    
    println!("\nInstructions:");
    for (i, step) in recipe.instructions.iter().enumerate() {
        println!("{}. {}", i + 1, step);
    }
    
    println!("\n\nExample 2: Using a system prompt");
    // Using a system prompt to guide the response
    let movie: MovieRecommendation = client
        .chat_with_system_prompt(
            "Recommend a movie similar to The Matrix",
            "You are a film critic with expertise in sci-fi movies. Be concise but informative."
        )
        .await?;
    
    println!("Movie Recommendation: {} ({})", movie.title, movie.year);
    println!("Director: {}", movie.director);
    println!("Genre: {}", movie.genre);
    println!("Summary: {}", movie.brief_summary);
    
    // Display token usage
    let usage = client.usage();
    println!("\nToken Usage:");
    println!("Prompt tokens: {}", usage.prompt_tokens);
    println!("Completion tokens: {}", usage.completion_tokens);
    println!("Total tokens: {}", usage.total_tokens);
    
    Ok(())
}
