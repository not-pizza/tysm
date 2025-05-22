//! This module contains the pricing of different models.

/// The cost of using a model, in dollars per million tokens.
#[derive(Debug, Clone)]
pub(crate) struct ModelCost {
    pub(crate) name: &'static str,
    /// The cost of input tokens, in dollars per million tokens.
    pub(crate) input: f64,
    /// The cost of cached input tokens, in dollars per million tokens.
    pub(crate) cached_input: Option<f64>,
    /// The cost of output tokens, in dollars per million tokens.
    pub(crate) output: f64,
}

pub(crate) const CHAT_COMPLETIONS: &[ModelCost] = &[
    // Anthropic
    ModelCost {
        name: "claude-3-7-sonnet",
        input: 3.0,
        cached_input: None,
        output: 15.0,
    },
    ModelCost {
        name: "claude-3-5-haiku",
        input: 0.80,
        cached_input: None,
        output: 4.0,
    },
    ModelCost {
        name: "claude-3-opus",
        input: 15.0,
        cached_input: None,
        output: 75.0,
    },
    ModelCost {
        name: "claude-opus-4",
        input: 15.0,
        cached_input: None,
        output: 75.0,
    },
    ModelCost {
        name: "claude-sonnet-4",
        input: 3.0,
        cached_input: None,
        output: 15.0,
    },
    ModelCost {
        name: "claude-haiku-4",
        input: 0.80,
        cached_input: None,
        output: 4.0,
    },
    // OpenAI
    // Copied from https://platform.openai.com/docs/pricing on 2025-04-17
    ModelCost {
        name: "gpt-4.1",
        input: 2.00,
        cached_input: Some(0.50),
        output: 8.00,
    },
    ModelCost {
        name: "gpt-4.1-mini",
        input: 0.40,
        cached_input: Some(0.10),
        output: 1.60,
    },
    ModelCost {
        name: "gpt-4.1-nano",
        input: 0.10,
        cached_input: Some(0.025),
        output: 0.40,
    },
    ModelCost {
        name: "gpt-4.5-preview",
        input: 75.00,
        cached_input: Some(37.50),
        output: 150.00,
    },
    ModelCost {
        name: "gpt-4o",
        input: 2.50,
        cached_input: Some(1.25),
        output: 10.00,
    },
    ModelCost {
        name: "gpt-4o-audio-preview",
        input: 2.50,
        cached_input: None,
        output: 10.00,
    },
    ModelCost {
        name: "gpt-4o-realtime-preview",
        input: 5.00,
        cached_input: Some(2.50),
        output: 20.00,
    },
    ModelCost {
        name: "gpt-4o-mini",
        input: 0.15,
        cached_input: Some(0.075),
        output: 0.60,
    },
    ModelCost {
        name: "gpt-4o-mini-audio-preview",
        input: 0.15,
        cached_input: None,
        output: 0.60,
    },
    ModelCost {
        name: "gpt-4o-mini-realtime-preview",
        input: 0.60,
        cached_input: Some(0.30),
        output: 2.40,
    },
    ModelCost {
        name: "o1",
        input: 15.00,
        cached_input: Some(7.50),
        output: 60.00,
    },
    ModelCost {
        name: "o1-pro",
        input: 150.00,
        cached_input: None,
        output: 600.00,
    },
    ModelCost {
        name: "o3",
        input: 10.00,
        cached_input: Some(2.50),
        output: 40.00,
    },
    ModelCost {
        name: "o4-mini",
        input: 1.10,
        cached_input: Some(0.275),
        output: 4.40,
    },
    ModelCost {
        name: "o3-mini",
        input: 1.10,
        cached_input: Some(0.55),
        output: 4.40,
    },
    ModelCost {
        name: "o1-mini",
        input: 1.10,
        cached_input: Some(0.55),
        output: 4.40,
    },
    ModelCost {
        name: "gpt-4o-mini-search-preview",
        input: 0.15,
        cached_input: None,
        output: 0.60,
    },
    ModelCost {
        name: "gpt-4o-search-preview",
        input: 2.50,
        cached_input: None,
        output: 10.00,
    },
    ModelCost {
        name: "computer-use-preview",
        input: 3.00,
        cached_input: None,
        output: 12.00,
    },
];

pub(crate) fn cost(model: &str, usage: crate::chat_completions::ChatUsage) -> Option<f64> {
    let model_cost = CHAT_COMPLETIONS
        .iter()
        .find(|mc| model.starts_with(mc.name))?;
    let (cached_prompt_tokens, uncached_prompt_tokens) =
        if let Some(details) = usage.prompt_token_details {
            (
                details.cached_tokens,
                usage.prompt_tokens - details.cached_tokens,
            )
        } else {
            (0, usage.prompt_tokens)
        };

    let usage_cost = model_cost.input * uncached_prompt_tokens as f64 / 1_000_000.0
        + model_cost.cached_input.unwrap_or(model_cost.input) * cached_prompt_tokens as f64
            / 1_000_000.0
        + model_cost.output * usage.completion_tokens as f64 / 1_000_000.0;
    Some(usage_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost() {
        let usage = crate::chat_completions::ChatUsage {
            prompt_tokens: 2000000,
            completion_tokens: 1000000,
            prompt_token_details: Some(crate::chat_completions::PromptTokenDetails {
                cached_tokens: 1000000,
            }),
            completion_token_details: None,
            total_tokens: 2000000,
        };
        let cost = cost("gpt-4o", usage);
        // input: 2.50,
        // cached_input: Some(1.25),
        // output: 10.00,
        assert_eq!(cost, Some(2.50 + 1.25 + 10.00));
    }
}
