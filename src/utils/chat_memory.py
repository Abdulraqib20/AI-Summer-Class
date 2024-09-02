# import uuid
# from typing import Dict, List, Tuple
# from llama_index.core.memory import ChatMemoryBuffer
# from llama_index.core.chat_engine.types import BaseChatEngine

# MEMORY_TOKEN_LIMIT = 1500  

# class ChatMemoryManager:
#     def __init__(self):
#         self.chat_memories: Dict[str, ChatMemoryBuffer] = {}

#     def create_chat_memory(self, session_id: str) -> None:
#         if session_id not in self.chat_memories:
#             self.chat_memories[session_id] = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)

#     def get_chat_memory(self, session_id: str) -> ChatMemoryBuffer:
#         if session_id not in self.chat_memories:
#             self.create_chat_memory(session_id)
#         return self.chat_memories[session_id]

#     def add_message(self, session_id: str, role: str, message: str) -> None:
#         chat_memory = self.get_chat_memory(session_id)
#         chat_memory.put(role, message)

#     def get_chat_history(self, session_id: str) -> List[Tuple[str, str]]:
#         chat_memory = self.get_chat_memory(session_id)
#         return chat_memory.get_all()

#     def clear_chat_history(self, session_id: str) -> None:
#         if session_id in self.chat_memories:
#             del self.chat_memories[session_id]

#     def apply_chat_memory(self, chat_engine: BaseChatEngine, session_id: str) -> BaseChatEngine:
#         chat_memory = self.get_chat_memory(session_id)
#         # return chat_engine.chat(chat_history=chat_memory)
#         return chat_engine.as_chat_engine(chat_history=chat_memory)

# chat_memory_manager = ChatMemoryManager()

# def generate_session_id() -> str:
#     return str(uuid.uuid4())


import uuid
from typing import Dict, List, Tuple
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import BaseChatEngine

class ChatMemoryManager:
    def __init__(self):
        self.chat_memories: Dict[str, ChatMemoryBuffer] = {}

    def create_chat_memory(self, session_id: str) -> None:
        if session_id not in self.chat_memories:
            self.chat_memories[session_id] = ChatMemoryBuffer.from_defaults(token_limit=1500)

    def get_chat_memory(self, session_id: str) -> ChatMemoryBuffer:
        if session_id not in self.chat_memories:
            self.create_chat_memory(session_id)
        return self.chat_memories[session_id]

    def add_message(self, session_id: str, message: str) -> None:
        chat_memory = self.get_chat_memory(session_id)
        chat_memory.put(message)

    def get_chat_history(self, session_id: str) -> List[Tuple[str, str]]:
        chat_memory = self.get_chat_memory(session_id)
        return chat_memory.get_all()

    def clear_chat_history(self, session_id: str) -> None:
        if session_id in self.chat_memories:
            del self.chat_memories[session_id]

    def apply_chat_memory(self, chat_engine, session_id: str):
        chat_memory = self.get_chat_memory(session_id)
        chat_engine.memory = chat_memory
        return chat_engine

chat_memory_manager = ChatMemoryManager()

def generate_session_id() -> str:
    return str(uuid.uuid4())