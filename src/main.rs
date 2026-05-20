use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "al-sum", about = "List, search and summarize Al transcript documents")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List available documents
    List,
    /// Summarize documents. Without --doc: print [id] title for each. With --doc: print title + full summary.
    Summarize {
        /// Document ID to summarize in full (omit for title overview of all docs)
        #[arg(long)]
        doc: Option<String>,
    },
    /// Search documents (hybrid text + semantic)
    Search {
        query: String,
        #[arg(long, default_value = "10")]
        limit: u32,
    },
}

struct Config {
    al_url: String,
    al_psk: String,
    openai_base_url: String,
    openai_api_key: String,
    openai_model: String,
}

impl Config {
    fn from_env() -> Result<Self> {
        Ok(Self {
            al_url: std::env::var("AL_URL").context("AL_URL not set")?,
            al_psk: std::env::var("AL_PSK").context("AL_PSK not set")?,
            openai_base_url: std::env::var("OPENAI_BASE_URL").context("OPENAI_BASE_URL not set")?,
            openai_api_key: std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY not set")?,
            openai_model: std::env::var("OPENAI_MODEL").context("OPENAI_MODEL not set")?,
        })
    }
}

#[derive(Deserialize)]
struct Document {
    doc_id: String,
    started_at: f64,
    ended_at: f64,
    entry_count: u64,
    snippet: Option<String>,
}

#[derive(Deserialize)]
struct DocumentsResponse {
    documents: Vec<Document>,
}

#[derive(Deserialize)]
struct Entry {
    text: String,
}

#[derive(Deserialize)]
struct DocumentDetail {
    doc_id: String,
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
struct HybridHit {
    doc_id: String,
    snippet: Option<String>,
    started_at: f64,
    ended_at: f64,
    entry_count: u64,
    score: Option<f64>,
}

#[derive(Deserialize)]
struct HybridResponse {
    hits: Vec<HybridHit>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

fn format_time(ts: f64) -> String {
    use chrono::TimeZone;
    let secs = ts as i64;
    let dt = chrono::Local.timestamp_opt(secs, 0).single().unwrap_or_default();
    dt.format("%Y-%m-%d %H:%M").to_string()
}

async fn fetch_documents(client: &Client, cfg: &Config) -> Result<Vec<Document>> {
    let resp = client
        .get(format!("{}/documents?limit=100", cfg.al_url))
        .header("Authorization", format!("Bearer {}", cfg.al_psk))
        .send()
        .await?
        .error_for_status()?
        .json::<DocumentsResponse>()
        .await?;
    Ok(resp.documents)
}

async fn fetch_document(client: &Client, cfg: &Config, id: &str) -> Result<DocumentDetail> {
    client
        .get(format!("{}/document/{}", cfg.al_url, id))
        .header("Authorization", format!("Bearer {}", cfg.al_psk))
        .send()
        .await?
        .error_for_status()?
        .json::<DocumentDetail>()
        .await
        .context("failed to parse document detail")
}

async fn llm(client: &Client, cfg: &Config, system: &str, user: &str) -> Result<String> {
    let req = ChatRequest {
        model: cfg.openai_model.clone(),
        messages: vec![
            ChatMessage { role: "system".into(), content: system.into() },
            ChatMessage { role: "user".into(), content: user.into() },
        ],
        stream: false,
    };
    let resp = client
        .post(format!("{}/v1/chat/completions", cfg.openai_base_url))
        .header("Authorization", format!("Bearer {}", cfg.openai_api_key))
        .json(&req)
        .send()
        .await?
        .error_for_status()?
        .json::<ChatResponse>()
        .await?;
    resp.choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .context("no choices in response")
}

async fn title_for(client: &Client, cfg: &Config, text: &str) -> Result<String> {
    llm(
        client,
        cfg,
        "You are a concise title generator. Reply with only a short title (5-8 words) for the transcript. No punctuation at the end.",
        &format!("Generate a title for this transcript:\n\n{}", text),
    )
    .await
}

async fn summary_for(client: &Client, cfg: &Config, text: &str) -> Result<String> {
    llm(
        client,
        cfg,
        "You are a concise summarizer. Summarize the transcript in Markdown. Use ## and below for any section headers (the caller provides the H1 title). Capture the main topics discussed.",
        &format!("Summarize this transcript:\n\n{}", text),
    )
    .await
}

async fn cmd_list(client: &Client, cfg: &Config) -> Result<()> {
    let docs = fetch_documents(client, cfg).await?;
    if docs.is_empty() {
        println!("No documents found.");
        return Ok(());
    }
    for doc in docs {
        let snippet = doc.snippet.unwrap_or_default();
        let snippet = if snippet.len() > 80 {
            format!("{}…", &snippet[..80])
        } else {
            snippet
        };
        println!(
            "[{}] {} → {}  ({} entries)  {}",
            doc.doc_id,
            format_time(doc.started_at),
            format_time(doc.ended_at),
            doc.entry_count,
            snippet,
        );
    }
    Ok(())
}

async fn cmd_summarize(client: &Client, cfg: &Config, doc_id: Option<String>) -> Result<()> {
    match doc_id {
        Some(id) => {
            let detail = fetch_document(client, cfg, &id).await?;
            if detail.entries.is_empty() {
                println!("(empty)");
                return Ok(());
            }
            let text = detail.entries.iter().map(|e| e.text.as_str()).collect::<Vec<_>>().join(" ");
            let title = title_for(client, cfg, &text).await?;
            let summary = summary_for(client, cfg, &text).await?;
            // Strip any H1 the model may have added despite instructions
            let summary = summary
                .lines()
                .skip_while(|l| l.starts_with("# "))
                .collect::<Vec<_>>()
                .join("\n")
                .trim_start()
                .to_string();
            println!("# {title}");
            println!();
            println!("{summary}");
        }
        None => {
            let docs = fetch_documents(client, cfg).await?;
            if docs.is_empty() {
                println!("No documents found.");
                return Ok(());
            }
            for doc in docs {
                let detail = fetch_document(client, cfg, &doc.doc_id).await?;
                if detail.entries.is_empty() {
                    println!("[{}] (empty)", doc.doc_id);
                    continue;
                }
                let text = detail.entries.iter().map(|e| e.text.as_str()).collect::<Vec<_>>().join(" ");
                let title = title_for(client, cfg, &text).await?;
                println!("[{}] {}", doc.doc_id, title);
            }
        }
    }
    Ok(())
}

async fn cmd_search(client: &Client, cfg: &Config, query: &str, limit: u32) -> Result<()> {
    let hits = client
        .get(format!("{}/search/hybrid", cfg.al_url))
        .header("Authorization", format!("Bearer {}", cfg.al_psk))
        .query(&[("q", query), ("limit", &limit.to_string())])
        .send()
        .await?
        .error_for_status()?
        .json::<HybridResponse>()
        .await?
        .hits;

    if hits.is_empty() {
        println!("No results.");
        return Ok(());
    }

    for h in hits {
        let score = h.score.map(|s| format!(" [{:.2}]", s)).unwrap_or_default();
        let snippet = h.snippet.unwrap_or_default();
        println!(
            "[doc:{}] {} → {}  ({} entries){}",
            h.doc_id,
            format_time(h.started_at),
            format_time(h.ended_at),
            h.entry_count,
            score,
        );
        println!("  {}", snippet);
        println!();
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::from_env()?;
    let client = Client::new();

    match cli.command {
        Command::List => cmd_list(&client, &cfg).await,
        Command::Summarize { doc } => cmd_summarize(&client, &cfg, doc).await,
        Command::Search { query, limit } => cmd_search(&client, &cfg, &query, limit).await,
    }
}
