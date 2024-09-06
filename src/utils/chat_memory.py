import uuid
from typing import Dict, List, Tuple
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatMessage
from cachetools import TTLCache

MEMORY_TOKEN_LIMIT = 1024
class ChatMemoryManager:
    def __init__(self, max_size=1000, ttl=3600):
        self.memories: Dict[str, ChatMemoryBuffer] = {}
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)

    def get_chat_memory(self, session_id: str) -> ChatMemoryBuffer:
        if session_id not in self.memories:
            self.memories[session_id] = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)
        return self.memories[session_id]

    def add_message(self, session_id: str, role: str, message: str):
        chat_memory = self.get_chat_memory(session_id)
        # Map 'human' to 'user' and 'ai' to 'assistant'
        role_mapping = {
            'human': 'user',
            'ai': 'assistant'
        }
        mapped_role = role_mapping.get(role, role)
        chat_message = ChatMessage(role=mapped_role, content=message)
        chat_memory.put(chat_message)
        self.cache[session_id] = True  # Update cache to keep session alive
        
    # def add_message(self, session_id: str, role: str, message: str):
    #     chat_memory = self.get_chat_memory(session_id)
    #     chat_memory.put(f"{role}: {message}")  # Combine role and message
    #     self.cache[session_id] = True  # Update cache to keep session alive

    # def get_chat_history(self, session_id: str) -> List[Tuple[str, str]]:
    #     chat_memory = self.get_chat_memory(session_id)
    #     return chat_memory.get()

    def get_chat_history(self, session_id: str) -> List[Tuple[str, str]]:
        chat_memory = self.get_chat_memory(session_id)
        return [(msg.role, msg.content) for msg in chat_memory.get()]
    
    def clear_chat_history(self, session_id: str):
        if session_id in self.memories:
            del self.memories[session_id]
        if session_id in self.cache:
            del self.cache[session_id]

    def apply_chat_memory(self, engine, session_id: str):
        chat_memory = self.get_chat_memory(session_id)
        if isinstance(engine, ReActAgent):
            engine.memory = chat_memory
            return engine
        else:
            return engine.as_chat_engine(chat_memory=chat_memory)
            # raise ValueError(f"Unsupported engine type: {type(engine)}")


chat_memory_manager = ChatMemoryManager()

def generate_session_id() -> str:
    return str(uuid.uuid4())