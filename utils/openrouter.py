import asyncio
from collections.abc import Callable
import aiohttp
from tqdm import tqdm
import os

async def send_async_requests(inputs: list[dict], request_generator: Callable, batch_size: int = 10, max_retries: int = 3, verbose: bool | None = None) -> list[object]:
    """
    Send many HTTP requests concurrently, with batching, exponential back-off
    and automatic retry.

    Params
        @inputs: A list of dicts, where each element corresponds to a single request.
        @request_generator: A function with two arguments (session, input), which takes a single input dictionary and returns a parsed 
         asyncio session response or raises.
        @batch_size: Max number of prompts to group in a single batch. Prompts in a batch are sent concurrently.
        @max_retries: Max number of retries on failed prompt calls (so total attempts <= max_retries + 1).
        @verbose: If True, prints progress. If None, defaults to True if multiple batches are used.

    Returns:
        A list in exactly the same order as inputs.

    Raises
        RuntimeError if any request is still failing after the allowed retries.

    Example
        # Example 1: GET request returning JSON
        async def req_gen(session, input):
            url = 'https://en.wikipedia.org/wiki/' + input['q'] + '/'
            headers = {'Content-Type': 'text/html'}
            async with session.get(url, headers = headers) as response:
                return await response.text()
                
        res = await send_async_requests(inputs = [{"q": "test"}, {"q": "dog"}], request_generator = req_gen)

        # Example 2: POST request returning JSON
        async def req_gen(session, input):
            url = 'https://api.openai.com/v1/chat/completions'
            headers = {'Authorization': 'Bearer ' + os.getenv('OPENAI_API_KEY')}
            payload = {'model': 'gpt-4.1', 'messages': input['p']}
            async with session.post(url, headers = headers, json = payload) as response:
                return await response.json()
                
        res = await send_async_requests(
            inputs = [{"p": [{'role': 'user', 'content': 'hello!'}]}, {"p": [{'role': 'user', 'content': 'bye!'}]}], 
            request_generator = req_gen
        )
    """

    # Retry helpers
    async def retry_requests(session, tasks, attempt: int = 0):
        """
        Recursive function to retry until they all exceed or we exceed max_retries.

        Params:
            @session: Open aiohttp session shared by the caller
            @tasks: List of (idx, payload) pairs on first call; the payload is the original element from `inputs`.
             On retries we keep a triple (idx, payload, exc) to surface the last err message if give up.
            @attempt: Current retry depth.

        Returns:
            List of (idx, result) pairs that succeeded (order *not* guaranteed)
        """
        # Stop condition
        if attempt > max_retries:
            summary = "\n".join(f"idx {idx}: {exc!r}" for idx, _payload, exc in tasks)
            raise RuntimeError(f"Maximum retries exceeded. Still failing:\n{summary}")
        
        # Exponential backoff before this retry round
        if attempt:
            backoff = 2 ** attempt
            print(f"Retry {attempt} - sleeping {backoff}s for {len(tasks)} requests")
            await asyncio.sleep(backoff)

        # Fire all pending requests concurrently.
        coroutines = (
            request_generator(session, payload) # Build the HTTP call
            for _idx, payload, *_ in tasks # Works for 2- or 3-tuples
        )
        results = await asyncio.gather(*coroutines, return_exceptions = True)

        # Partition results into `completed` and `pending` for the next try. #
        completed = [] # List of 2-tuples containing (index, return obj)
        pending = [] # List of 3-tuples containing (index, input dict, exception)

        for (idx, payload, *_), result in zip(tasks, results):
            if isinstance(result, Exception):
                if attempt == 0:                       # first failure → log once
                    print(f"[idx {idx}] first failure: {result!r}")
                pending.append((idx, payload, result)) # keep the exception!
            else:
                completed.append((idx, result))

        # If anything failed, retry just those; otherwise bubble success to the caller
        if pending:
            completed.extend(
                await retry_requests(session, pending, attempt + 1)
            )
            
        return completed
    
    # Batching
    indexed = list(enumerate(inputs))
    chunks = [indexed[i:i + batch_size] for i in range(0, len(indexed), batch_size)]
    show_bar = verbose is True or (verbose is None and len(chunks) > 1)

    all_pairs: list[tuple[int, object]] = []

    # Sequential loop
    for chunk in tqdm(chunks, total = len(chunks), disable = not show_bar, leave = True):
        async with aiohttp.ClientSession() as session:   # new session each batch
            pairs = await retry_requests(session, chunk)
        all_pairs.extend(pairs)

    # Re-ordering
    flat = {idx: resp for idx, resp in all_pairs}
    if len(flat) != len(inputs):
        raise RuntimeError('Output length mismatch; some requests never succeeded')

    return [flat[i] for i in range(len(inputs))]


async def get_openrouter_responses(prompts: list[list], params: dict, batch_size: int = 3, max_retries: int = 3, api_key: str = os.getenv('OPENROUTER_API_KEY'), verbose = None):
    """
    Asynchronously send a list of LLM prompts to the Openrouter endpoint

    Params:
        @prompts: A lists of prompts, where each prompt is a list of messages to send in the request.
        @params: Anything other than the messages to pass into the request body, such as model or temperature.
        @batch_size: Max number of prompts to group in a single batch. Prompts in a batch are sent concurrently.
        @max_retries: Max number of retries on failed prompt calls.
        @api_key: The Openrouter API key.
        
    Example:
        prompts_list = [
            [{'role': 'system', 'content': 'You are a math teacher.'}, {'role': 'user', 'content': 'What is 1+1?'}],
            [{'role': 'system', 'content': 'You are a math teacher.'}, {'role': 'user', 'content': 'What is 1+2?'}],
            [{'role': 'system', 'content': 'You are a math teacher.'}, {'role': 'user', 'content': 'What is 1+3?'}],
            [{'role': 'system', 'content': 'You are a math teacher.'}, {'role': 'user', 'content': 'What is 1+4?'}]
        ]
        
        await get_openrouter_responses(
            prompts_list,
            {
                'model': 'z-ai/glm-4.5',
                'temperature': 0,
                'top_p': 1,
                'topk_k': 1,
                'frequency_penalty': 0,
                'presence_penalty': 0,
                'repetition_penalty': 1,
                'provider': {
                    'order': ['deepinfra/fp8'],
                    'allow_fallbacks': False
                }
            }
        )
    """

    url = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {'Authorization': 'Bearer ' + api_key}
    params = params or {}

    async def _request(session, prompt: list[dict]) -> object:
        payload = {'messages': prompt, **params}
        async with session.post(url, headers = headers, json = payload) as resp:
            return await resp.json()

    return await send_async_requests(
        inputs = prompts,
        request_generator = _request,
        batch_size = batch_size,
        max_retries = max_retries,
        verbose = verbose,
    )