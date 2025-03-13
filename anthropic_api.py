"""
Generic Anthropic API client for handling Claude API interactions.
"""
from typing import List, Dict, Union, Optional, Generator, Any, TypeAlias, Sequence
import pandas as pd
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import base64
import pdfplumber
from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlockParam, DocumentBlockParam, Base64PDFSourceParam, CacheControlEphemeralParam
from anthropic._exceptions import OverloadedError, RateLimitError


class ModelType(Enum):
    HAIKU = "claude-3-5-haiku-20241022"
    SONNET = "claude-3-7-sonnet-20250219"


BlockParams: TypeAlias = Sequence[Union[TextBlockParam, DocumentBlockParam]]


@dataclass
class Thread(ABC):
    """Base class for managing a conversation thread with Claude."""
    api_key: str
    user_prompt: str
    name: str
    max_tokens: int
    
    def __post_init__(self):
        """Initialize the Anthropic client."""
        self.client = Anthropic(api_key=self.api_key)
    
    @staticmethod
    def load_pdf_text(pdf_path: Path) -> str:
        """Extract text content from PDF using pdfplumber."""
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text

    @staticmethod
    def load_pdf_data(pdf_path: Path) -> str:
        """Load PDF and encode as base64."""
        with pdf_path.open("rb") as pdf_file:
            binary_data = pdf_file.read()
            base64_encoded_data = base64.b64encode(binary_data)
            return base64_encoded_data.decode("utf-8")
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt to guide Claude's behavior. To be implemented by subclasses."""
        pass

    @property
    @abstractmethod
    def model(self) -> ModelType:
        """Model to use for the conversation. To be implemented by subclasses."""
        pass
    
    def message_iterator(
        self, 
        system_prompt: str, 
        initial_content: BlockParams,
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
        
        retry_count, max_retries = 0, 3
        # Main conversation loop
        while True:
            # Call the API - exceptions will propagate to caller
            try:
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
            except (OverloadedError, RateLimitError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\nRate limit or overloaded error hit: {e.__class__.__name__}, waiting 15 seconds before retry {retry_count}/{max_retries}")
                    import time
                    time.sleep(15)
                    continue
                raise  # Re-raise if we've hit max retries
            
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
    
    def iterate_thread(self, initial_content: Optional[BlockParams] = None) -> Generator[str, Optional[str], None]:
        """
        Process the conversation thread, yielding each response.
        
        Args:
            initial_content: Optional list of content blocks. If not provided,
                the thread will prepare its own content.
                
        Returns:
            Generator yielding Claude's responses and accepting messages via send()
        """
        # Add the user prompt to initial content
        content = [
            *(initial_content or []),
            TextBlockParam(type="text", text=self.user_prompt)
        ]
        
        # Create message iterator
        message_gen = self.message_iterator(
            system_prompt=self.system_prompt,
            initial_content=content,
            model=self.model,
            max_tokens=self.max_tokens
        )
        
        # Get first response
        response_text = next(message_gen)
        
        # Enter the send/yield loop
        while True:
            # Yield the response and get the user's next message
            user_message = yield response_text
            
            # If no message was sent, we're done
            if user_message is None:
                break
                
            # Forward the user message to the API iterator
            try:
                response_text = message_gen.send(user_message)
            except StopIteration:
                # API iterator is done
                break 