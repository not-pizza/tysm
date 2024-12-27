# tysm - Thank You So Much

**Typed OpenAI Chat Completions in Rust**

[crates.io](https://crates.io/crates/tysm) | [docs.rs](https://docs.rs/tysm/latest/tysm/) | [blog post](https://chadnauseam.com/coding/ai/openai-structured-outputs-are-really-useful)

## Table of Contents
- [tysm - Thank You So Much](#tysm---thank-you-so-much)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [Features](#features)
  - [Setup](#setup)
    - [Automatic Caching](#automatic-caching)
    - [Custom API endpoints](#custom-api-endpoints)
  - [Feature flags](#feature-flags)
  - [License](#license)
  - [Backstory](#backstory)
  - [Footguns](#footguns)

A strongly-typed Rust client for OpenAI's ChatGPT API that enforces type-safe responses using [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs).

## Usage

```rust
use tysm::ChatClient;

/// We want names separated into `first` and `last`.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Name {
    first: String,
    last: String,
}

async fn get_president_name() {
    // Create a client.
    // `from_env` will look for an API key under the environment
    // variable "OPENAI_API_KEY"
    // It will also look inside `.env` if such a file exists.
    let client = ChatClient::from_env("gpt-4o").unwrap();
    
    // Request a chat completion from OpenAI and
    // parse the response into our `Name` struct.
    let name: Name = client
        .chat("Who was the first US president?")
        .await
        .unwrap();

    assert_eq!(name.first, "George");
    assert_eq!(name.last, "Washington");
}
```

## Features

- Type-safe API responses
- Concise interface
- Automatic local caching of API responses

## Setup

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys).
2. Create a `.env` file, and make it look like this:

   ```
   OPENAI_API_KEY=<your api key here>
   ```
3. Add `.env` to your `.gitignore` so you don't accidentally commit it.
4. Add the crate and the necessary dependencies to your Rust project with:
    1. `cargo add tysm serde`. 
    2. `cargo add --git https://github.com/GREsau/schemars.git schemars`


### Automatic Caching

I'm a big fan of memoization. By default, the last 1024 responses will be stored inside the `Client`. For this reason it can be useful to make a client just once using LazyLock (which is part of the standard library since 1.80).

```rust
use std::sync::LazyLock;
use tysm::ChatClient;

// Create a lazily-initialized `CLIENT` variable to avoid recreating a `ChatClient` every time we want to hit the API.
static CLIENT: LazyLock<ChatClient> = LazyLock::new(|| ChatClient::from_env("gpt-4o").unwrap());

fn see() {
    #[derive(tysm::Deserialize, tysm::JsonSchema)]
    struct Name {
        first: String,
        last: String,
    }

    for _ in 0..10_000 {
        // The built-in cache prevents us from going bankrupt
        let _name: Name = CLIENT.chat("Who was the first US president?").await.unwrap();
    }
}
```

### Custom API endpoints

Sometimes people want to use a different completions API. For example, I maintain a wrapper around OpenAI's API that adds a global cache. To switch endpoints, just do this:

```rust
let my_api = "https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/chat/completions".to_string();
let client = ChatClient {
    url: my_api,
    ..ChatClient::from_env("gpt-4o").unwrap()
};
```

By the way, feel free to use this endpoint if you want, but I don't promise to maintain it forever.

## Feature flags

The following feature flags are available:

1. `dotenv` - (enabled by default) Enables automatic loading of environment variables from a `.env` file. 

Example of disabling dotenv:
```toml
[dependencies]
tysm = { version = "0.2", default-features = false }
```

## License

This project is licensed under the MIT License.

## Backstory

The name stands for "thank you so much", which is what I say I ask ChatGPT a question and get a great answer! If you prefer, it could also stand for "**Ty**ped **S**chema **M**agic".

I like making ChatGPT-wrappers. Unfortunately the rust ecosystem for calling ChatGPT is more anemic than you would think, and it's not very complicated, so I always end up writing my own code for calling it. It's just an API endpoint after all. In my various git repos, I'd estimate I have about 5 implementations of this.

I was in the middle of writing my 6th on a lazy christmas eve when I realized that I'm too lazy to keep doing that. So I decided to solve the problem for myself once and for all.

I almost never use streaming or anything fancy like that so this library doesn't support it. I designed it with my future lazy self in mind - which is why it has dotenv built in and has built-in caching.

The whole library is basically one file right now, so hopefully it will be easy for you to move on from once you outgrow it.

## Footguns

1. the trait bound `Books: schemars::JsonSchema` is not satisfied

```
error[E0277]: the trait bound `MyStruct: schemars::JsonSchema` is not satisfied
note: required by a bound in `ChatClient::chat`
   --> ~/coding/typed-openai/src/chatgpt.rs:198:64
    |
198 |     pub async fn chat<T: DeserializeOwned + JsonSchema>(
    |                                             ^^^^^^^^^^ required by this bound in `ChatClient::chat`
```

You probably forgot to add the in-development version of Schemars to your project. Try replacing the `schemars` entry in your Cargo.toml with this:

```toml
schemars = { git = "https://github.com/GREsau/schemars.git", version = "1.0.0-alpha.17", features = [
    "preserve_order",
] }
```
