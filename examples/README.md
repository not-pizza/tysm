# Examples

This directory contains examples demonstrating how to use the different APIs provided by the `tysm` crate.

## Prerequisites

Before running these examples, make sure you have:

1. An OpenAI API key
2. Set the API key in your environment as `OPENAI_API_KEY`

You can set the API key in your environment with:

```bash
export OPENAI_API_KEY=your-api-key-here
```

Or create a `.env` file in the project root with:

```
OPENAI_API_KEY=your-api-key-here
```

## Running the Examples

To run an example, use the following command from the project root:

```bash
cargo run --example <example_name>
```

Where `<example_name>` is one of:
- `chat_completions_example`
- `embeddings_example`
- `files_example`

## Examples Overview

### Chat Completions API (`chat_completions_example.rs`)

Demonstrates how to:
- Create a chat client
- Get structured responses from the API
- Use system prompts to guide responses
- Track token usage

### Embeddings API (`embeddings_example.rs`)

Demonstrates how to:
- Create an embeddings client
- Generate embeddings for multiple documents
- Generate an embedding for a single document
- Calculate similarity between embeddings

### Files API (`files_example.rs`)

Demonstrates how to:
- List existing files
- Upload files from a file path
- Upload content directly from bytes
- Retrieve file information
- Download file content
- Delete files

## Notes

- The Files API examples will create and delete temporary files in your OpenAI account.
- These examples use minimal error handling for clarity. In production code, you should implement more robust error handling.
