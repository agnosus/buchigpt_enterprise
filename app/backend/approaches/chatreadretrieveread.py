from typing import Any, Coroutine, List, Literal, Optional, Union, overload

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
)

from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper
from core.modelhelper import get_token_limit
import asyncio

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
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
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
        return """You are a helpful assistant that answers technical questions about Kjeldahl and distillation. Be brief in your answers.

Concepts to remember:
- Application: A method used on an instrument to determine the amount of a given analyte or to describe how to use the instrument on a given sample type with specific parameters. Methods, procedures, and results of such applications are explained in application notes.
- Configuration/Instrument Configuration: An instrument with a particular article number that includes a set of features, components, or accessories. A bundle refers to an instrument sold with another instrument, usually with a specific article number.
- Digesters (Digestion Units): Instruments such as KjelDigester K-446 and KjelDigester K-449, which are block digesters. SpeedDigesters include SpeedDigester K-425, SpeedDigesters K-436, and SpeedDigesters K-439.
- Scrubber K-415: An instrument for fume removal during digestion, available in multiple configurations (DuoScrub, TripleScrub, TripleScrubECO, QuadScrubECO).
- Distillation Units: Instruments divided into low-mid-range distillation (product line K-365) and high-end Kjeldahl distillation:
  - Kjel Line: For nitrogen-containing analytes, consisting of EasyKjel, BasicKjel, and MultiKjel.
  - Dist Line: Consisting of EasyDist, BasicDist, and MultiDist, each with different analyte capabilities.
  - High-End Kjeldahl: Includes the KjelMaster K-375 (a distillation unit with integrated titration for nitrogen-containing analytes) which can be coupled with KjelSampler K-376 / K-377 (an autosampler instrument that can transfer samples to the KjelMaster K-375).

Answer ONLY with the facts listed in the list of sources below. If there isn't enough information, say you don't know and that you are being trained and will be updated soon. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.

For tabular information, return it as an HTML table. Do not return markdown format. If the question is not in English, answer in the language used in the question.

Each source has a name followed by the actual information. Always include the source name for each fact you use in the response. Use square brackets to reference the source, for example [info1.txt]. Don't combine sources, list each source separately, for example [info1.txt][info2.pdf].

{follow_up_questions_prompt}
{injected_prompt}

        """

    @overload
    async def run_until_final_call(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...

    @overload
    async def run_until_final_call(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...

    async def run_until_final_call(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)

        filter = self.build_filter(overrides, auth_claims)
        use_semantic_ranker = True if overrides.get("semantic_ranker") and has_text else False

        original_user_query = history[-1]["content"]
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

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        query_messages = self.get_messages_from_history(
            system_prompt=self.query_prompt_template,
            model_id=self.chatgpt_model,
            history=history,
            user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - len(user_query_request),
            few_shots=self.query_prompt_few_shots,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=query_messages,  # type: ignore
            # Azure OpenAI takes the deployment name as the model name
            model="gpt4" if overrides.get('use_gpt4') else "chat",
            temperature=0.0,  # Minimize creativity for search query generation
            max_tokens=100,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
            tools=tools,
            tool_choice="auto",
        )

        query_text = self.get_search_query(chat_completion, original_user_query)

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if has_vector:
            vectors.append(await self.compute_text_embedding(query_text))

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
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

        # GAHAintervention: each result is checked using gpt to see if it can answer the question 
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

        # end of intervention


        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        content = "\n".join(sources_content)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        system_message = self.get_system_prompt(
            overrides.get("prompt_template"),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        response_token_limit = 1024
        messages_token_limit = self.chatgpt_token_limit - response_token_limit
        messages = self.get_messages_from_history(
            system_prompt=system_message,
            model_id=self.chatgpt_model,
            history=history,
            # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=messages_token_limit,
        )

        data_points = {"text": sources_content}

        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate search query",
                    [str(message) for message in query_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
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
                    [str(message) for message in messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        chat_coroutine = self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model="gpt4" if overrides.get('use_gpt4') else "chat",
            messages=messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
        )
        return (extra_info, chat_coroutine)
