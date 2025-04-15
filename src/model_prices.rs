//! This module contains the pricing of different models.

/// The cost of using a model, in dollars per million tokens.
#[derive(Debug, Clone)]
pub(crate) struct ModelCost {
    pub(crate) name: &'static str,
    /// The cost of input tokens, in dollars per million tokens.
    pub(crate) input: f64,
    /// The cost of output tokens, in dollars per million tokens.
    pub(crate) output: f64,
}

pub(crate) const CHAT_COMPLETIONS: &[ModelCost] = &[
    ModelCost {
        name: "claude-3-7-sonnet",
        input: 3.0,
        output: 15.0,
    },
    ModelCost {
        name: "claude-3-5-haiku",
        input: 0.80,
        output: 4.0,
    },
    ModelCost {
        name: "claude-3-opus",
        input: 15.0,
        output: 75.0,
    },
    ModelCost {
        name: "gpt-4.5",
        input: 75.0,
        output: 150.00,
    },
    ModelCost {
        name: "gpt-4o",
        input: 2.50,
        output: 10.0,
    },
    ModelCost {
        name: "gpt-4o-mini",
        input: 0.150,
        output: 0.600,
    },
    ModelCost {
        name: "gpt-o1",
        input: 15.00,
        output: 60.00,
    },
    ModelCost {
        name: "o3-mini",
        input: 1.10,
        output: 4.40,
    },
    ModelCost {
        name: "gpt-4.1",
        input: 2.0,
        output: 8.0,
    },
    ModelCost {
        name: "gpt-4.1-mini",
        input: 0.4,
        output: 1.6,
    },
    ModelCost {
        name: "gpt-4.1-nano",
        input: 0.1,
        output: 0.4,
    },
];

pub(crate) fn cost(model: &str, usage: crate::chat_completions::ChatUsage) -> Option<f64> {
    let model_cost = CHAT_COMPLETIONS
        .iter()
        .find(|mc| model.starts_with(mc.name))?;
    let usage_cost = model_cost.input * usage.prompt_tokens as f64 / 1_000_000.0
        + model_cost.output * usage.completion_tokens as f64 / 1_000_000.0;
    Some(usage_cost)
}
