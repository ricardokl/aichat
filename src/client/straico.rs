use super::prompt_format::*;
use super::*;

use anyhow::{anyhow, bail, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

const API_BASE: &str = "https://api.straico.com/v1";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct StraicoConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl StraicoClient {
    config_get_fn!(api_key, get_api_key);

    pub const PROMPTS: [PromptAction<'static>; 1] =
        [("api_key", "API Key:", true, PromptKind::String)];
}

impl_client_trait!(
    StraicoClient,
    (
        prepare_chat_completions,
        straico_chat_completions,
        straico_chat_completions_streaming
    ),
    (noop_prepare_embeddings, noop_embeddings),
    (noop_prepare_rerank, noop_rerank),
);

fn prepare_chat_completions(
    self_: &StraicoClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key().ok();
    let url = format!("{API_BASE}/prompt/completion");
    let body = straico_build_chat_completions_body(data, &self_.model)?;
    let mut request_data = RequestData::new(url, body);
    if let Some(api_key) = api_key {
        request_data.bearer_auth(api_key);
    }
    Ok(request_data)
}

pub async fn straico_chat_completions(
    builder: RequestBuilder,
    model: &Model,
) -> Result<ChatCompletionsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    debug!("non-stream-data: {data}");
    straico_extract_chat_completions(&data, &model)
}

async fn straico_chat_completions_streaming(
    _: RequestBuilder,
    _: &mut SseHandler,
    _: &Model,
) -> Result<()> {
    bail!("Straico does not support streaming")
}

fn straico_build_chat_completions_body(data: ChatCompletionsData, model: &Model) -> Result<Value> {
    let ChatCompletionsData {
        messages,
        temperature: _,
        top_p: _,
        functions: _,
        stream: _,
    } = data;

    let prompt = generate_prompt(&messages, smart_prompt_format(model.name()))?;

    Ok(json!({
        "message": prompt,
        "models": [model.name()],
    }))

    // Ok(body)
}

fn straico_extract_chat_completions(data: &Value, model: &Model) -> Result<ChatCompletionsOutput> {
    let model_data: &Value = &data["data"]["completions"][model.name()]["completion"];
    let text = &model_data["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("Invalid response data: {data}"))?;
    let id = &model_data["id"];

    let output = ChatCompletionsOutput {
        text: text.to_string(),
        tool_calls: vec![],
        id: Some(id.to_string()),
        input_tokens: None,
        output_tokens: None,
    };

    Ok(output)
}
