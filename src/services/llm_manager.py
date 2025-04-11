
import runpod
import openai 
import logging

from src.config import RUNPOD_API_KEY
from src.config import Config

config = Config()
logger = logging.getLogger(__name__)


class LLMManager:
    def __call__(self, *args, **kwds):
        pass
    
    def _generate_prompt(self, query: str, context: str) -> str:
        prompt = f"""
        You are an helpful assistant named Eve developed by ESA and PiSchool that help researchers and students understanding topics reguarding Earth Observation.
        Givent the following context: {context}.
        
        Please reply in a precise and accurate manner to this query: {query}
        
        Answer:
        """
        return prompt
    
    def generate_answer(
        self,
        query: str,
        context: str,
        llm="llama-3.1",
        max_new_tokens=150,
    ):

        
        prompt = self._generate_prompt(query=query, context=context)

        if llm == "openai":
            messages_pompt = []
            messages_pompt += [{"role": "user", "content": prompt}]

            response = openai.chat.completions.create(
                messages=messages_pompt,
                model="gpt-4",
                temperature=0.3,
                max_tokens=500,
            )
            message = response.choices[0].message.content
            return message

        elif llm == "eve-instruct-v0.1":
            #TODO: this shall make context smaller. Maybe needs fixings
            context = (
                context if context != "" else context[: (1024 - max_new_tokens) * 4]
            )

            runpod.api_key = RUNPOD_API_KEY
            endpoint = runpod.Endpoint(config.get_instruct_llm_id())
            print("PROMPT: ", prompt)

            try:
                response = endpoint.run_sync(
                    {
                        "input": {
                            "prompt": f"{prompt}",
                            "sampling_params": {"max_tokens": max_new_tokens},
                        }
                    },
                    timeout=config.get_instruct_llm_timeout(),
                )
                return " ".join(response[0]["choices"][0]["tokens"])
            except TimeoutError as e:
                logger.error(f"Job timed out. {str(e)}")
            except Exception as e:
                logging.error(f"{str(e)}")
