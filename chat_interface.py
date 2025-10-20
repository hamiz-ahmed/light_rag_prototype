"""
Chat Interface Module

Provides a command-line chat interface for querying the RAG system.
"""

import readline
import sys
from typing import Optional
from rag_processor import RAGProcessor


class ChatInterface:
    """
    Command-line chat interface for RAG system queries.
    """

    def __init__(self, rag_processor: RAGProcessor):
        """
        Initialize the chat interface.

        Args:
            rag_processor: Configured RAG processor instance
        """
        self.rag_processor = rag_processor
        self.history = []

        # Setup readline for better input handling
        self._setup_readline()

    def _setup_readline(self):
        """Setup readline for enhanced input experience."""
        try:
            # Enable tab completion (basic)
            readline.parse_and_bind('tab: complete')

            # Set up history file
            history_file = ".rag_chat_history"
            try:
                readline.read_history_file(history_file)
            except (FileNotFoundError, PermissionError):
                pass

            # Save history on exit
            import atexit
            def save_history():
                try:
                    readline.write_history_file(history_file)
                except (PermissionError, OSError):
                    pass  # Silently fail if we can't write history
            
            atexit.register(save_history)

        except (ImportError, Exception):
            # readline not available on some platforms or other issues
            pass

    def start_chat(self):
        """Start the interactive chat session."""
        print("\n" + "="*60)
        print("🤖 LightRAG Chat Interface")
        print("="*60)
        print("Type your questions about the documentation.")
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear conversation history")
        print("  /quit     - Exit the chat")
        print("="*60 + "\n")

        while True:
            try:
                # Get user input
                user_input = self._get_input()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        break
                    continue

                # Process query
                self._process_query(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except EOFError:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

    def _get_input(self) -> str:
        """Get user input with prompt."""
        try:
            return input("You: ").strip()
        except KeyboardInterrupt:
            raise
        except EOFError:
            raise

    def _handle_command(self, command: str) -> bool:
        """
        Handle chat commands.

        Args:
            command: The command string

        Returns:
            True if should exit, False otherwise
        """
        command = command.lower()

        if command == '/quit':
            print("Goodbye! 👋")
            return True
        elif command == '/help':
            self._show_help()
        elif command == '/history':
            self._show_history()
        elif command == '/clear':
            self._clear_history()
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands.")

        return False

    def _show_help(self):
        """Show help message."""
        print("\nAvailable commands:")
        print("  /help     - Show this help message")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear conversation history")
        print("  /quit     - Exit the chat")
        print()

    def _show_history(self):
        """Show conversation history."""
        if not self.history:
            print("No conversation history.")
            return

        print("\nConversation History:")
        print("-" * 40)
        for i, (question, answer) in enumerate(self.history, 1):
            print(f"{i}. Q: {question}")
            print(f"   A: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print()
        print("-" * 40)

    def _clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        print("Conversation history cleared.")

    def _process_query(self, question: str):
        """
        Process a user query and display the response.

        Args:
            question: User's question
        """
        print("Thinking... 🤔")

        # Query the RAG system
        try:
            response = self.rag_processor.query(question)
            
            # Check if response is None or empty
            if not response or response == "None":
                response = "I couldn't find relevant information to answer your question. Please try rephrasing or ask something else."
                
        except Exception as e:
            print(f"ERROR: Query failed: {e}")
            import traceback
            traceback.print_exc()
            response = f"Sorry, an error occurred while processing your query: {e}"

        # Display response
        print(f"\nAssistant: {response}\n")

        # Add to history
        self.history.append((question, response))

        # Limit history to last 50 conversations
        if len(self.history) > 50:
            self.history = self.history[-50:]
