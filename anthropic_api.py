"""
Generic Anthropic API client for handling Claude API interactions.
"""
from typing import List, Dict, Union, Optional, Generator
import pandas as pd
from pathlib import Path
from enum import Enum
from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlockParam, DocumentBlockParam


class ModelType(Enum):
    HAIKU = "claude-3-5-haiku-20241022"
    SONNET = "claude-3-7-sonnet-20250219"


class AnthropicAPI:
    def __init__(self, api_key: str):
        """Initialize the API client with Anthropic API key."""
        self.client = Anthropic(api_key=api_key)

    def message_iterator(
        self, 
        system_prompt: str, 
        initial_content: List[Union[TextBlockParam, DocumentBlockParam]],
        model: ModelType,
        max_tokens: int = 8192
    ) -> Generator[str, Optional[str], None]:
        """
        Coroutine that handles message exchange with Claude API.
        
        This is a bidirectional generator: it yields API responses to the caller,
        and the caller can send back the next message using generator.send().
        
        Example usage:
            gen = api.message_iterator(system_prompt, initial_content, model)
            
            # Get first response
            response = next(gen)
            
            # Send follow-up and get next response
            next_response = gen.send("follow up question")
            
            # Decide not to continue
            gen.close()  # or just don't call send() again
        
        Args:
            system_prompt: The system prompt to guide Claude's behavior
            initial_content: Initial content blocks to send to Claude
            model: The Claude model to use for this conversation
            max_tokens: Maximum tokens for Claude response
        
        Yields:
            Each response from Claude
            
        Input:
            String to send as next user message when .send() is called;
            or None to end the conversation
        """
        messages: List[MessageParam] = [
            MessageParam(
                role="user",
                content=initial_content
            )
        ]
        
        # Main conversation loop
        while True:
            # Call the API - exceptions will propagate to caller
            response = self.client.messages.create(
                model=model.value,
                max_tokens=max_tokens,
                temperature=0,
                system=[
                    TextBlockParam(
                        type="text",
                        text=system_prompt
                    )
                ],
                messages=messages
            )
            
            print(
                f"Input size: {response.usage.input_tokens} tokens\n",
                f"Response size: {response.usage.output_tokens} tokens\n"
            )
            
            # Extract text from all text blocks in the response
            response_texts = []
            for block in response.content:
                if block.type == "text":
                    response_texts.append(block.text)
            
            # Combine all text blocks into a single response
            response_text = "\n".join(response_texts) if response_texts else ""
            
            # Add assistant's response to conversation history
            messages.append(MessageParam(
                role="assistant",
                content=[TextBlockParam(
                    type="text",
                    text=response_text
                )]
            ))
            
            # Yield the response and get the next message from caller
            next_message = yield response_text
            
            # If caller doesn't provide a next message, exit the conversation
            if next_message is None:
                return
                
            # Add user's next message to conversation history
            messages.append(MessageParam(
                role="user",
                content=[
                    TextBlockParam(
                        type="text",
                        text=next_message
                    )
                ]
            ))

    def save_dataframes(self, data: Dict[str, pd.DataFrame], output_path: Path) -> None:
        """Save extracted DataFrames to CSV files."""
        for name, table in data.items():
            try:
                csv_path = output_path.with_name(f"{output_path.stem}_{name}.csv")
                table.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"Error saving {name} to CSV: {e}") 