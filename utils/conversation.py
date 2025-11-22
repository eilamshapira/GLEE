"""
Simplified conversation templates for LiteLLM.
Replaces the FastChat implementation to remove dependency on FastChat.
"""

class Conversation:
    def __init__(self, name, roles=("user", "assistant")):
        self.name = name
        self.roles = roles
        self.messages = []
        self.system_message = ""

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_openai_api_messages(self):
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        
        for role, message in self.messages:
            # Map roles to OpenAI format
            # litellm expects 'user' and 'assistant'
            api_role = role
            if role == self.roles[0]:
                api_role = "user"
            elif role == self.roles[1]:
                api_role = "assistant"
            
            if message:
                messages.append({"role": api_role, "content": message})
        
        return messages

def get_conv_template(name):
    """
    Get a conversation template.
    We use a standard template for all models, relying on LiteLLM to handle
    model-specific formatting and role mapping.
    """
    return Conversation(name, roles=("user", "assistant"))
