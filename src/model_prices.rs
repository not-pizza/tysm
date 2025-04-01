//! This module contains information about the pricing of different models.

use std::collections::HashMap;
use std::fmt;

/// The cost of using a model, in dollars per million tokens.
#[derive(Debug, Clone)]
pub struct ModelCost {
    /// The cost of input tokens, in dollars per million tokens.
    pub input: f64,
    /// The cost of output tokens, in dollars per million tokens.
    pub output: f64,
    /// The cost of prompt caching write, in dollars per million tokens.
    pub prompt_caching_write: Option<f64>,
    /// The cost of prompt caching read, in dollars per million tokens.
    pub prompt_caching_read: Option<f64>,
    /// Whether this model has a 50% discount with batch processing.
    pub batch_discount: bool,
}

impl ModelCost {
    /// Create a new ModelCost.
    pub fn new(input: f64, output: f64) -> Self {
        Self {
            input,
            output,
            prompt_caching_write: None,
            prompt_caching_read: None,
            batch_discount: false,
        }
    }

    /// Set the prompt caching costs.
    pub fn with_prompt_caching(mut self, write: f64, read: f64) -> Self {
        self.prompt_caching_write = Some(write);
        self.prompt_caching_read = Some(read);
        self
    }

    /// Set whether this model has a batch discount.
    pub fn with_batch_discount(mut self, batch_discount: bool) -> Self {
        self.batch_discount = batch_discount;
        self
    }
}

/// The costs of using different chat completion models.
#[derive(Debug, Clone)]
pub struct ChatCompletionsCost {
    /// A map from model name to its cost.
    pub models: HashMap<String, ModelCost>,
}

impl ChatCompletionsCost {
    /// Create a new ChatCompletionsCost with default values.
    pub fn new() -> Self {
        let mut models = HashMap::new();

        // Claude 3.7 Sonnet
        models.insert(
            "claude-3-7-sonnet-20240229".to_string(),
            ModelCost::new(3.0, 15.0)
                .with_prompt_caching(3.75, 0.30)
                .with_batch_discount(true),
        );

        // Claude 3.5 Haiku
        models.insert(
            "claude-3-5-haiku-20240307".to_string(),
            ModelCost::new(0.80, 4.0)
                .with_prompt_caching(1.0, 0.08)
                .with_batch_discount(true),
        );

        // Claude 3 Opus
        models.insert(
            "claude-3-opus-20240229".to_string(),
            ModelCost::new(15.0, 75.0)
                .with_prompt_caching(18.75, 1.50)
                .with_batch_discount(true),
        );

        Self { models }
    }

    /// Get the cost of a model.
    pub fn get(&self, model: &str) -> Option<&ModelCost> {
        self.models.get(model)
    }

    /// Calculate the cost of a request in dollars.
    pub fn calculate_cost(
        &self,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
        cached_tokens: Option<u32>,
        batch: bool,
    ) -> Option<f64> {
        let model_cost = self.get(model)?;
        
        let mut cost = 0.0;
        let input_tokens_millions = input_tokens as f64 / 1_000_000.0;
        let output_tokens_millions = output_tokens as f64 / 1_000_000.0;
        
        // Calculate input cost
        if let Some(cached_tokens) = cached_tokens {
            let cached_tokens_millions = cached_tokens as f64 / 1_000_000.0;
            let new_tokens = input_tokens.saturating_sub(cached_tokens);
            let new_tokens_millions = new_tokens as f64 / 1_000_000.0;
            
            if let (Some(write_cost), Some(read_cost)) = (model_cost.prompt_caching_write, model_cost.prompt_caching_read) {
                cost += new_tokens_millions * write_cost;
                cost += cached_tokens_millions * read_cost;
            } else {
                cost += input_tokens_millions * model_cost.input;
            }
        } else {
            cost += input_tokens_millions * model_cost.input;
        }
        
        // Calculate output cost
        cost += output_tokens_millions * model_cost.output;
        
        // Apply batch discount if applicable
        if batch && model_cost.batch_discount {
            cost *= 0.5;
        }
        
        Some(cost)
    }
}

impl Default for ChatCompletionsCost {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_costs() {
        let costs = ChatCompletionsCost::new();
        
        // Test Claude 3.7 Sonnet
        let sonnet = costs.get("claude-3-7-sonnet-20240229").unwrap();
        assert_eq!(sonnet.input, 3.0);
        assert_eq!(sonnet.output, 15.0);
        assert_eq!(sonnet.prompt_caching_write, Some(3.75));
        assert_eq!(sonnet.prompt_caching_read, Some(0.30));
        assert!(sonnet.batch_discount);
        
        // Test Claude 3.5 Haiku
        let haiku = costs.get("claude-3-5-haiku-20240307").unwrap();
        assert_eq!(haiku.input, 0.80);
        assert_eq!(haiku.output, 4.0);
        assert_eq!(haiku.prompt_caching_write, Some(1.0));
        assert_eq!(haiku.prompt_caching_read, Some(0.08));
        assert!(haiku.batch_discount);
        
        // Test Claude 3 Opus
        let opus = costs.get("claude-3-opus-20240229").unwrap();
        assert_eq!(opus.input, 15.0);
        assert_eq!(opus.output, 75.0);
        assert_eq!(opus.prompt_caching_write, Some(18.75));
        assert_eq!(opus.prompt_caching_read, Some(1.50));
        assert!(opus.batch_discount);
    }

    #[test]
    fn test_cost_calculation() {
        let costs = ChatCompletionsCost::new();
        
        // Test regular cost calculation
        let cost = costs.calculate_cost("claude-3-5-haiku-20240307", 1_000, 500, None, false).unwrap();
        let expected = (1_000.0 / 1_000_000.0) * 0.80 + (500.0 / 1_000_000.0) * 4.0;
        assert!((cost - expected).abs() < 0.0001);
        
        // Test with caching
        let cost = costs.calculate_cost("claude-3-5-haiku-20240307", 1_000, 500, Some(400), false).unwrap();
        let expected = (600.0 / 1_000_000.0) * 1.0 + (400.0 / 1_000_000.0) * 0.08 + (500.0 / 1_000_000.0) * 4.0;
        assert!((cost - expected).abs() < 0.0001);
        
        // Test with batch discount
        let cost = costs.calculate_cost("claude-3-5-haiku-20240307", 1_000, 500, None, true).unwrap();
        let expected = ((1_000.0 / 1_000_000.0) * 0.80 + (500.0 / 1_000_000.0) * 4.0) * 0.5;
        assert!((cost - expected).abs() < 0.0001);
    }
}
