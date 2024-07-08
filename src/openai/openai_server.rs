use super::requests::ChatCompletionRequest;
use super::requests::Messages;
use super::responses::{APIError, ChatCompletionResponse, ChatCompletionUsageResponse};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::new_streaming_conn;
use super::utils::get_created_time_secs;
use super::OpenAIServerData;
use actix_web::{post, web, Either, HttpResponse};
use tokenizers::Encoding;
use uuid::Uuid;

// fn verify_model(data: &OpenAIServerData<'_>, model_name: &String) -> Result<(), APIError> {
//     let current_name = {
//         let model = data.model.lock().unwrap();
//         model.get_pipeline().name().to_string()
//     };
//     if &current_name != model_name {
//         Err(APIError::new(format!(
//             "Model name `{model_name}` is invalid."
//         )))
//     } else {
//         Ok(())
//     }
// }

// Get prompt, roles
async fn get_gen_prompt(
    data: &OpenAIServerData<'_>,
    request: &web::Json<ChatCompletionRequest>,
) -> Result<String, APIError> {
    let mut model = data.model.lock().await;
    let conversation = model
        .get_mut_pipeline()
        .get_conversation(data.record_conversation);

    match &request.messages {
        Messages::Literal(msg) => {
            return Ok(msg.clone());
        }
        Messages::Map(messages) => {
            for message in messages {
                let role = message
                    .get("role")
                    .ok_or(APIError::new("Message key `role` not found.".to_string()))?;
                let content = message
                    .get("content")
                    .ok_or(APIError::new(
                        "Message key `content` not found.".to_string(),
                    ))?
                    .clone();

                if role == "system" {
                    conversation.set_system_message(content);
                } else {
                    conversation.append_message(role.to_string(), content)
                }
            }
        }
    }

    Ok(conversation.get_prompt())
}

async fn check_length(
    request: &web::Json<ChatCompletionRequest>,
    prompt: String,
    data: &OpenAIServerData<'_>,
) -> Result<Encoding, APIError> {
    let token_ids = {
        let model = data.model.lock().await;
        model
            .get_pipeline()
            .tokenizer()
            .tokenizer()
            .encode(prompt, false)
            .map_err(APIError::from)?
    };

    let max_gen_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    if token_ids.len() + max_gen_tokens > data.pipeline_config.max_model_len {
        Err(APIError::new(format!(
            "This model's maximum context length is {} tokens. \
            However, you requested {} tokens ({} in the messages, \
            {} in the completion). \nPlease clear the chat history or reduce the length of the \
            messages.",
            data.pipeline_config.max_model_len,
            max_gen_tokens + token_ids.len(),
            token_ids.len(),
            max_gen_tokens
        )))
    } else {
        Ok(token_ids)
    }
}

#[post("/v1/chat/completions")]
async fn chat_completions(
    data: web::Data<OpenAIServerData<'static>>,
    request: web::Json<ChatCompletionRequest>,
) -> Either<Result<web::Json<ChatCompletionResponse>, APIError>, HttpResponse> {
    // let model_name = &request.model;
    // let res = verify_model(&data, model_name);
    // if res.is_err() {
    //     return Either::Left(Err(res.err().unwrap()));
    // }

    if request.logit_bias.as_ref().is_some()
        && request.logit_bias.as_ref().is_some_and(|x| !x.is_empty())
    {
        return Either::Left(Err(APIError::new_str(
            "`logit_bias` is not currently supported.",
        )));
    }

    let prompt = get_gen_prompt(&data, &request).await;
    if prompt.is_err() {
        return Either::Left(Err(prompt.err().unwrap()));
    }
    let prompt = prompt.unwrap();

    let token_ids = check_length(&request, prompt.clone(), &data).await;
    if token_ids.is_err() {
        return Either::Left(Err(token_ids.err().unwrap()));
    }
    let mut token_ids: Encoding = token_ids.unwrap();
    if token_ids.len() % 2 == 0 {
        //padding to avoid block allocation issue
        token_ids.pad(
            token_ids.len() + 1,
            0,
            0,
            "\n",
            tokenizers::PaddingDirection::Right,
        );
    }
    println!("\n\n\nPrompt {:?}", prompt);

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let sampling_params = SamplingParams::new(
        request.n.unwrap_or(1),
        request.best_of,
        request.presence_penalty.unwrap_or(0.0),
        request.frequency_penalty.unwrap_or(0.0),
        request.repetition_penalty.unwrap_or(1.1),
        request.temperature.unwrap_or(0.7),
        request.top_p.unwrap_or(1.0),
        request.top_k.unwrap_or(-1),
        request.use_beam_search.unwrap_or(false),
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        request.stop.clone(),
        request.stop_token_ids.clone().unwrap_or_default(),
        request.ignore_eos.unwrap_or(false),
        request
            .max_tokens
            .unwrap_or(data.pipeline_config.default_max_tokens),
        None,
        None,
        request.skip_special_tokens.unwrap_or(true),
    );
    if sampling_params.is_err() {
        return Either::Left(Err(sampling_params.err().unwrap()));
    }
    let sampling_params = sampling_params.unwrap();

    let created = get_created_time_secs();

    if request.stream.is_some_and(|x| x) {
        let (sender, receiver) = new_streaming_conn();
        let _ = tokio::spawn(async move {
            let mut model = data.model.lock().await;
            let result = model
                .generate(
                    token_ids,
                    request_id,
                    created,
                    sampling_params,
                    request.logprobs.unwrap_or(false),
                    Some(&sender),
                )
                .await
                .unwrap();
            //chat completion statistics
            let usage = ChatCompletionUsageResponse {
                completion_tokens: result
                    .iter()
                    .map(|(_, usage)| usage.completion_tokens)
                    .sum(),
                prompt_tokens: result.iter().map(|(_, usage)| usage.prompt_tokens).sum(),
                total_tokens: result.iter().map(|(_, usage)| usage.total_tokens).sum(),
                prompt_time_costs: result
                    .iter()
                    .map(|(_, usage)| usage.prompt_time_costs)
                    .sum(),
                completion_time_costs: result
                    .iter()
                    .map(|(_, usage)| usage.completion_time_costs)
                    .sum(),
            };
            println!(
                "\r\n Prefilling: {} prompt tokens processed in {} seconds",
                usage.prompt_tokens,
                usage.prompt_time_costs / 1000
            );

            println!(
                "\r\n Decoding: {} tokens processed in {} seconds ({} tokens/s)",
                usage.completion_tokens,
                usage.completion_time_costs / 1000,
                usage.completion_tokens * 1000
                    / if usage.completion_time_costs > 0 {
                        usage.completion_time_costs
                    } else {
                        1
                    }
            );
        });

        return Either::Right(
            HttpResponse::Ok()
                .append_header(("content-type", "text/event-stream"))
                .streaming(receiver),
        );
    }
    let result = {
        let mut model = data.model.lock().await;
        let model_res = model
            .generate(
                token_ids,
                request_id.clone(),
                created,
                sampling_params,
                request.logprobs.unwrap_or(false),
                None,
            )
            .await;
        if model_res.is_err() {
            return Either::Left(Err(model_res.err().unwrap()));
        }
        model_res.unwrap()
    };

    let choices = result
        .iter()
        .flat_map(|(choices, _)| choices.clone())
        .collect::<Vec<_>>();
    let usage = ChatCompletionUsageResponse {
        completion_tokens: result
            .iter()
            .map(|(_, usage)| usage.completion_tokens)
            .sum(),
        prompt_tokens: result.iter().map(|(_, usage)| usage.prompt_tokens).sum(),
        total_tokens: result.iter().map(|(_, usage)| usage.total_tokens).sum(),
        prompt_time_costs: result
            .iter()
            .map(|(_, usage)| usage.prompt_time_costs)
            .sum(),
        completion_time_costs: result
            .iter()
            .map(|(_, usage)| usage.completion_time_costs)
            .sum(),
    };

    Either::Left(Ok(web::Json(ChatCompletionResponse {
        id: request_id,
        choices,
        created,
        model: request.model.clone(),
        object: "chat.completion",
        usage,
    })))
}
