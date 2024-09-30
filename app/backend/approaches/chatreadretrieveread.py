from typing import Any, Coroutine, List, Literal, Optional, Union, overload
import asyncio

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper

class ChatReadRetrieveReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],
        embedding_deployment: Optional[str],
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    @property
    def system_message_chat_conversation(self):
        return """You are a helpful assistant that answers technical questions about Rotavaours and Evaporation. Be brief in your answers.
Concepts to remember:
Application: A method used on an instrument to determine the amount of a given analyte or to describe how to use the instrument on a given sample type with specific parameters. Methods, procedures, and results of such applications are explained in application notes.\\n - Configuration/Instrument Configuration: An instrument with a particular article number that includes a set of features, components, or accessories. A bundle refers to an instrument sold with another instrument, usually with a specific article number.\\n – Rotavapor: Instruments such as Rotavapor R-300, Rotavapor R-100 and
Rotavapor R-80
are rotary evaporators.
- Chillers: Instruments such as Recirculating Chiller F-100, Recirculating Chiller F-105, Recirculating Chiller F-305, Recirculating Chiller F-308 and Recirculating Chiller F-314
- Pumps: Instruments such as  Vacuum Pump V-100, Vacuum Pump V-300, Vacuum Pump V-600, Vacuum Pump V-80 and Vacuum Pump V-180.
- Vacuum controllers : Instruments  such as Interface I-80, Interface I-180, Interface I-100, Interface I-300 and Interface I-300 Pro.
- Dynamic Line: For high end rotary evaporator system solution consisting of instruments such as Rotavapor R-300, Recirculating Chiller F-305, Recirculating Chiller F-308, Recirculating Chiller F-314, Vacuum Pump V-300 and Vacuum Pump V-600, Interface I-300 and Interface I-300 Pro.
- Essential Line: Consisting of instruments such Rotavapor R-80,  Recirculating Chiller F-100, Recirculating Chiller F-105,   Vacuum Pump V-80, Vacuum Pump V-180, Interface I-80and Interface I-180 for distillation in up to 1 l evaporation flasks and instruments such as Rotavapor R-100, Recirculating Chiller F-100, Recirculating Chiller F-105, Vacuum Pump V-100 and Interface I-100 for distillation in up to 4 l evaporation flasks.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information, just say "I was not able to find any information in the provided resources. If your question is considered relevant and there should be an answer available, I will receive training and updates in the coming weeks." Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
   For tabular information, return it as an HTML table. Do not return markdown format. Always use plain text for equations. If the question is not in English, answer in the language used in the question.
   Each source has a name followed by the actual information. Always include the source name for each fact you use in the response. Use square brackets to reference the source, for example [example1.txt]. Don't combine sources, list each source separately, for example [example1.txt][example2.pdf].
{follow_up_questions_prompt}
{injected_prompt}
        """

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]

        use_semantic_captions = True if overrides.get("retrieval_mode") == "text" else False
        if use_semantic_captions:
            top = 3
            minimum_search_score = 0
            minimum_reranker_score = 0
        else:
            top = overrides.get("top", 3)
            minimum_search_score = overrides.get("minimum_search_score", 0.0)
            minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)

        filter = self.build_filter(overrides, auth_claims)
        use_semantic_ranker = True if overrides.get("semantic_ranker") and has_text else False

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        user_query_request = "Generate search query for: " + original_user_query

        tools: List[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": "search_sources",
                    "description": "Retrieve sources from the Azure AI Search index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                            }
                        },
                        "required": ["search_query"],
                    },
                },
            }
        ]

        query_response_token_limit = 100
        query_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=self.query_prompt_template,
            tools=tools,
            few_shots=self.query_prompt_few_shots,
            past_messages=messages[:-1],
            new_user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=query_messages,
            model="gpt4" if overrides.get('use_gpt4') else "chat",
            temperature=0.0,
            max_tokens=query_response_token_limit,
            n=1,
            tools=tools,
        )

        query_text = self.get_search_query(chat_completion, original_user_query)

        vectors: list[VectorQuery] = []
        if has_vector:
            vectors.append(await self.compute_text_embedding(query_text))

        if not has_text:
            query_text = None

        results = await self.search(
            top,
            query_text,
            filter,
            vectors,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        async def generate_response(user_query, DOC, i):
            doc = DOC[i].content
            response = await self.openai_client.chat.completions.create(
                model="gpt4" if overrides.get('use_gpt4') else "chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that is expert in technical text understanding. You pay attention to the smallest detail in the text. Keep it concise and to the point."
                    },
                    {
                        "role": "user",
                        "content": f"Answer the user query based on the following context. If the context does not contain enough information to answer the query, return NONE : \nQUERY: {user_query} \n CONTEXT: {doc}"
                    }
                ],
                temperature=0)
            response_str = str(response.choices[0].message.content).strip()
            return response_str

        async def generate_responses_async(user_query, DOC):
            tasks = [generate_response(user_query, DOC, i) for i in range(len(DOC))]
            responses = await asyncio.gather(*tasks)

            RESPONSE = []
            INDICES = []
            for index, response_str in enumerate(responses):
                if response_str.lower() != 'none':
                    RESPONSE.append(response_str)
                    INDICES.append(index)

            return RESPONSE, INDICES

        RESPONSE, INDICES = await generate_responses_async(query_text, results)
        
        for i in INDICES:
            idx = INDICES.index(i)
            results[i].content = RESPONSE[idx]
        final_result = [results[i] for i in INDICES]
        results = final_result

        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        content = "\n".join(sources_content)

        system_message = self.get_system_prompt(
            overrides.get("prompt_template"),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        response_token_limit = 1024
        messages_for_completion = build_messages(
            model=self.chatgpt_model,
            system_prompt=system_message,
            past_messages=messages[:-1],
            new_user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        data_points = {"text": sources_content}

        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate search query",
                    [str(message) for message in query_messages],
                    {"model": "gpt4o"}
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "has_vector": has_vector,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in messages_for_completion],
                    {"model": "gpt4o"},
                ),
            ],
        }

        # Generate the initial response
        chat_completion = await self.openai_client.chat.completions.create(
            model="gpt4" if overrides.get('use_gpt4') else "chat",
            messages=messages_for_completion,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=False,
        )

        initial_response = chat_completion.choices[0].message.content

        # Check if the response indicates no information was found
        if 'any information' in initial_response.lower():
            # If no information was found, create a new chat completion
            chat_coroutine = self.openai_client.chat.completions.create(
                model="gpt4" if overrides.get('use_gpt4') else "chat",
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are an expert in evaporation. Think step by step and respond to the user's question. Begin your response with: 'Sorry, no information was found in our documents. Meanwhile, here is an answer from OpenAI GPT4o that you may find useful!\n\n'
Here is the question: {query_text}"""
                    }
                ],
                temperature=overrides.get("temperature", 0.3),
                max_tokens=1024,
                n=1,
                stream=should_stream,
            )

            # Update extra_info to reflect this action
            extra_info["thoughts"].append(
                ThoughtStep(
                    "No information in documents",
                    "Initial response indicated no information was found. Creating a new chat completion with GPT knowledge.",
                    {"model": "gpt4o"},
                )
            )
       
        else:
            # If information was found, use the original response
            chat_coroutine = self.openai_client.chat.completions.create(
                model="gpt4" if overrides.get('use_gpt4') else "chat",
                messages=messages_for_completion,
                temperature=overrides.get("temperature", 0.3),
                max_tokens=response_token_limit,
                n=1,
                stream=should_stream,
            )

        return (extra_info, chat_coroutine) 